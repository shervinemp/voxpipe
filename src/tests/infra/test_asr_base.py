"""Microphone capture and recogniser plumbing, tested with a fake sound device and VAD model."""
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from voxpipe.asr.models.base import ModelBase
from voxpipe.asr.vad import Silero
from voxpipe.core.exceptions import ASRError

from ._fakes import FakeVadModel, loud, quiet


class PortAudioError(Exception):
    pass


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = self.stopped = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def deliver(self, chunk):
        self.callback(chunk, len(chunk), None, None)


class FakeSoundDevice:
    PortAudioError = PortAudioError

    def __init__(self):
        self.devices = [
            {"name": "Speakers (Realtek)", "max_input_channels": 0},
            {"name": "Microphone (USB Audio)", "max_input_channels": 1},
            {"name": "Headset Microphone", "max_input_channels": 1},
        ]
        self.streams = []
        self.fail_open = 0

    def query_devices(self, device=None, kind=None):
        if device is None and kind is None:
            return self.devices
        if device is None:
            return self.devices[1]
        return self.devices[device]

    def InputStream(self, **kwargs):  # noqa: N802 - mirrors sounddevice's API
        if self.fail_open:
            self.fail_open -= 1
            raise PortAudioError("device busy")
        self.streams.append(FakeStream(**kwargs))
        return self.streams[-1]


class Echo(ModelBase):
    """Transcribes a segment as the number of chunks it holds."""

    def _transcribe(self, audio):
        if getattr(self, "broken", False):
            self.broken = False
            raise RuntimeError("model crashed")
        return f"{len(audio) // 512} chunks"


class TestModelBase(unittest.TestCase):
    def setUp(self):
        self.sd = FakeSoundDevice()
        patcher = patch("voxpipe.asr.models.base.sd", self.sd)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self, sound_device=None, **kwargs):
        vad = Silero(model=FakeVadModel(), leading_silence_duration=0.0, trailing_silence_duration=0.3)
        return Echo(vad, sound_device, **kwargs)

    def test_stream_matches_the_vad(self):
        asr = self.make()
        options = self.sd.streams[0].kwargs
        self.assertEqual((options["samplerate"], options["blocksize"], options["channels"]), (16_000, 512, 1))
        self.assertIsNone(options["device"], "None selects the system default input")
        self.assertIs(asr.vad, asr._vad)

    def test_device_by_index_or_name(self):
        self.assertEqual(self.make(2)._input_stream.kwargs["device"], 2)
        self.assertEqual(self.make("usb audio")._input_stream.kwargs["device"], 1)
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(self.make("no such mic")._input_stream.kwargs["device"])

    def test_name_lookup_skips_output_devices(self):
        self.sd.devices[0]["name"] = "Realtek Microphone Output"
        self.assertEqual(self.make("microphone")._input_stream.kwargs["device"], 1)

    def test_no_input_device_is_an_asr_error(self):
        self.sd.query_devices = MagicMock(side_effect=PortAudioError("no default input"))
        with self.assertRaises(ASRError):
            self.make()

    def test_iteration_transcribes_each_utterance(self):
        asr = self.make()
        stream = self.sd.streams[0]
        for chunk in loud(20) + quiet(20) + loud(8) + quiet(20):
            stream.deliver(chunk)
        asr.stop()
        trailing = asr.vad._trailing_chunks
        self.assertEqual(list(asr), [f"{20 + trailing} chunks", f"{8 + trailing} chunks"])
        self.assertTrue(stream.stopped and stream.closed)

    def test_audio_is_copied_out_of_the_callback_buffer(self):
        asr = self.make()
        buffer = loud(1)[0]
        for _ in range(20):
            self.sd.streams[0].deliver(buffer)
        buffer[:] = 0  # PortAudio reuses its buffer after the callback returns
        asr.mute()
        asr.stop()
        self.assertEqual(list(asr), ["20 chunks"])

    def test_mute_ends_the_utterance_and_ignores_later_audio(self):
        asr = self.make()
        stream = self.sd.streams[0]
        self.assertFalse(asr.is_muted)
        for chunk in loud(12):
            stream.deliver(chunk)
        asr.mute()
        self.assertTrue(asr.is_muted)
        for chunk in loud(30):
            stream.deliver(chunk)
        asr.unmute()
        for chunk in loud(7):
            stream.deliver(chunk)
        asr.mute()
        asr.stop()
        self.assertEqual(list(asr), ["12 chunks", "7 chunks"])

    def test_failed_transcription_yields_an_empty_transcript_and_recognition_goes_on(self):
        asr = self.make()
        asr.broken = True
        stream = self.sd.streams[0]
        for chunk in loud(10):
            stream.deliver(chunk)
        asr.mute()
        asr.unmute()
        for chunk in loud(6):
            stream.deliver(chunk)
        asr.mute()
        asr.stop()
        with self.assertLogs(level="ERROR"):
            self.assertEqual(list(asr), ["", "6 chunks"], "the consumer still learns that an utterance ended")

    def test_stop_unblocks_a_waiting_consumer(self):
        asr = self.make()
        asr.start()
        heard = []
        consumer = threading.Thread(target=lambda: heard.extend(asr), daemon=True)
        consumer.start()
        time.sleep(0.05)
        asr.stop()
        consumer.join(timeout=2)
        self.assertFalse(consumer.is_alive())
        self.assertEqual(heard, [])

    def test_context_manager_starts_and_stops(self):
        with self.make() as asr:
            stream = asr._input_stream
            self.assertTrue(stream.started)
        self.assertTrue(stream.stopped)

    def test_lost_device_is_reopened_by_name(self):
        Echo.RECONNECT_INTERVAL = 0.02
        self.addCleanup(setattr, Echo, "RECONNECT_INTERVAL", ModelBase.RECONNECT_INTERVAL)
        asr = self.make("usb audio", reconnect_timeout=0.05)
        self.addCleanup(asr.stop)
        first = asr._input_stream
        with self.assertLogs(level="WARNING"):
            asr.start()
            first.deliver(quiet(1)[0])  # audio flows, then stops
            self.sd.fail_open = 2  # the device is busy for the first two attempts
            deadline = time.monotonic() + 3
            while asr._input_stream is first and time.monotonic() < deadline:
                time.sleep(0.01)
        self.assertIsNot(asr._input_stream, first)
        self.assertTrue(first.closed)
        self.assertTrue(asr._input_stream.started)
        self.assertEqual(asr._input_stream.kwargs["device"], 1)

    def test_gives_up_on_a_device_that_never_returns(self):
        Echo.RECONNECT_INTERVAL = 0.005
        Echo.MAX_RECONNECT_ATTEMPTS = 3
        self.addCleanup(setattr, Echo, "RECONNECT_INTERVAL", ModelBase.RECONNECT_INTERVAL)
        self.addCleanup(setattr, Echo, "MAX_RECONNECT_ATTEMPTS", ModelBase.MAX_RECONNECT_ATTEMPTS)
        asr = self.make("usb audio", reconnect_timeout=0.01)
        with self.assertLogs(level="ERROR"):
            asr.start()
            asr._input_stream.deliver(quiet(1)[0])
            self.sd.devices = []  # unplugged for good
            heard = []
            consumer = threading.Thread(target=lambda: heard.extend(asr), daemon=True)
            consumer.start()
            consumer.join(timeout=3)
        self.assertFalse(consumer.is_alive(), "iteration ends so the pipeline can shut down")
        asr.stop()


class TestProviders(unittest.TestCase):
    def test_registry(self):
        from voxpipe.asr.model import ASRProviders
        from voxpipe.core.exceptions import ProviderError

        self.assertEqual(ASRProviders.names(), ["CrisperWhisper", "ParakeetV2"])
        with self.assertRaises(ProviderError):
            ASRProviders.create("Whisperer")

    def test_tts_registry(self):
        from voxpipe.core.exceptions import ProviderError
        from voxpipe.tts.model import Kokoro, TTSProviders

        self.assertEqual(TTSProviders.names(), ["Kokoro"])
        self.assertIs(TTSProviders.get("Kokoro"), Kokoro)
        with self.assertRaises(ProviderError):
            TTSProviders.get("Robot")


class TestStreamLifecycle(unittest.TestCase):
    def setUp(self):
        self.sd = FakeSoundDevice()
        patcher = patch("voxpipe.asr.models.base.sd", self.sd)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self, **kwargs):
        vad = Silero(model=FakeVadModel(), leading_silence_duration=0.0, trailing_silence_duration=0.3)
        return Echo(vad, **kwargs)

    def test_a_stream_that_cannot_be_opened_is_an_asr_error(self):
        self.sd.fail_open = 1
        with self.assertRaisesRegex(ASRError, "Could not open audio input"):
            self.make()

    def test_stop_during_a_reconnect_leaves_no_stream_running(self):
        asr = self.make()
        asr._stopped.set()  # stop() got in while the watchdog was opening the replacement
        original = self.sd.streams[0]

        real_open = asr._open_stream
        asr._stopped.clear()

        def open_then_stop(device):
            stream = real_open(device)
            asr._stopped.set()
            return stream

        asr._open_stream = open_then_stop
        self.assertTrue(asr._reconnect())

        replacement = self.sd.streams[-1]
        self.assertIsNot(replacement, original)
        self.assertTrue(replacement.stopped and replacement.closed, "the stream opened too late is closed, not kept")
        self.assertIs(asr._input_stream, original)

    def test_audio_delivered_after_stop_is_ignored(self):
        asr = self.make()
        stream = self.sd.streams[0]
        asr.stop()
        for chunk in loud(10):
            stream.deliver(chunk)
        self.assertEqual(list(asr), [])

    def test_the_watchdog_counts_from_start_when_no_audio_ever_arrives(self):
        asr = self.make()
        self.assertIsNone(asr._last_chunk_time)
        with patch.object(asr, "_watch_device"):
            asr.start()
        self.addCleanup(asr.stop)
        self.assertIsNotNone(asr._last_chunk_time, "a stream that never delivers must still be noticed as lost")


if __name__ == "__main__":
    unittest.main()
