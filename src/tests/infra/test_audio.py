"""AudioPlayer and TTS queue tests against a fake sound device."""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from voxpipe.tts.audio import AudioPlayer
from voxpipe.tts.base import TTSBase


class FakeSoundDevice:
    """Stands in for the ``sounddevice`` module: clips "play" until finished or stopped."""

    def __init__(self):
        self.default = SimpleNamespace(device=(0, 1))
        self.played = []
        self.stops = 0
        self._done = threading.Event()
        self._started = threading.Event()

    def query_devices(self, device=None, kind=None):
        return {"name": "fake speakers", "max_output_channels": 2}

    def play(self, audio, samplerate=None, device=None):
        self._done.clear()
        self.played.append(audio)
        self._started.set()

    def wait(self):
        self._done.wait(timeout=5)

    def stop(self):
        self.stops += 1
        self._done.set()

    # --- test controls ---

    def finish_clip(self):
        """Let the clip being played run to its end."""
        self._done.set()

    def wait_for_clip(self, count):
        deadline = time.monotonic() + 2
        while len(self.played) < count and time.monotonic() < deadline:
            time.sleep(0.005)
        return len(self.played) >= count


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


class TestAudioPlayer(unittest.TestCase):
    def setUp(self):
        self.sd = FakeSoundDevice()
        patcher = patch("voxpipe.tts.audio.sd", self.sd)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.player = AudioPlayer()
        self.addCleanup(self.player.stop)

    def test_uses_default_output_device(self):
        self.assertEqual(self.player.output_device, 1)
        self.assertEqual(AudioPlayer(output_device=0).output_device, 0)

    def test_plays_clips_in_order(self):
        self.player.start()
        for clip in ("a", "b", "c"):
            self.assertTrue(self.player.enqueue(clip, 24_000))
        for count in (1, 2, 3):
            self.assertTrue(self.sd.wait_for_clip(count))
            self.assertTrue(self.player.is_playing)
            self.sd.finish_clip()
        self.assertEqual(self.sd.played, ["a", "b", "c"])
        self.assertTrue(wait_until(lambda: not self.player.is_playing))

    def test_stop_playback_cuts_the_clip_and_drops_the_queue(self):
        self.player.start()
        for clip in ("a", "b", "c"):
            self.player.enqueue(clip, 24_000)
        self.assertTrue(self.sd.wait_for_clip(1))
        generation = self.player.generation

        self.player.stop_playback()

        self.assertEqual(self.player.generation, generation + 1)
        self.assertGreaterEqual(self.sd.stops, 1)
        self.assertTrue(wait_until(lambda: not self.player.is_playing))
        self.assertEqual(self.sd.played, ["a"], "queued clips must not play after an interruption")

    def test_playback_resumes_after_an_interruption(self):
        self.player.start()
        self.player.enqueue("old", 24_000)
        self.assertTrue(self.sd.wait_for_clip(1))
        self.player.stop_playback()
        self.player.enqueue("new", 24_000)
        self.assertTrue(self.sd.wait_for_clip(2))
        self.assertEqual(self.sd.played, ["old", "new"])

    def test_clips_from_an_interrupted_generation_are_refused(self):
        stale = self.player.generation
        self.player.stop_playback()
        self.assertFalse(self.player.enqueue("late", 24_000, stale))
        self.assertTrue(self.player.enqueue("fresh", 24_000, self.player.generation))
        self.assertTrue(self.player.is_playing, "a queued clip counts as playing")

    def test_a_full_queue_makes_the_producer_wait_rather_than_lose_a_clip(self):
        player = AudioPlayer(max_queued=1)
        self.addCleanup(player.stop)
        player.start()
        player.enqueue("a", 1)
        self.assertTrue(self.sd.wait_for_clip(1))
        player.enqueue("b", 1)  # fills the queue while "a" plays

        queued = []
        producer = threading.Thread(target=lambda: queued.append(player.enqueue("c", 1)), daemon=True)
        producer.start()
        producer.join(timeout=0.2)
        self.assertTrue(producer.is_alive(), "the third clip waits for room")

        for count in (2, 3):
            self.sd.finish_clip()
            self.assertTrue(self.sd.wait_for_clip(count))
        producer.join(timeout=2)
        self.assertEqual(queued, [True])
        self.assertEqual(self.sd.played, ["a", "b", "c"])

    def test_an_interruption_releases_a_waiting_producer(self):
        player = AudioPlayer(max_queued=1)
        self.addCleanup(player.stop)
        player.start()
        player.enqueue("a", 1)
        self.assertTrue(self.sd.wait_for_clip(1))
        player.enqueue("b", 1)
        stale = player.generation

        queued = []
        producer = threading.Thread(target=lambda: queued.append(player.enqueue("c", 1, stale)), daemon=True)
        producer.start()
        player.stop_playback()
        producer.join(timeout=2)
        self.assertEqual(queued, [False])

    def test_a_full_queue_with_no_player_running_drops_the_clip(self):
        player = AudioPlayer(max_queued=2)
        self.assertTrue(player.enqueue("a", 1))
        self.assertTrue(player.enqueue("b", 1))
        with self.assertLogs("voxpipe.tts.audio", level="WARNING"):
            self.assertFalse(player.enqueue("c", 1), "waiting would never end")

    def test_is_playing_holds_from_queueing_until_the_clip_has_finished(self):
        seen = []
        original_play = self.sd.play

        def play(audio, **kwargs):
            seen.append(self.player.is_playing)  # the clip has left the queue but is not audible yet
            original_play(audio, **kwargs)

        self.sd.play = play
        self.player.start()
        self.player.enqueue("a", 24_000)
        self.assertTrue(self.sd.wait_for_clip(1))
        self.assertEqual(seen, [True])
        self.assertTrue(self.player.is_playing)
        self.sd.finish_clip()
        self.assertTrue(wait_until(lambda: not self.player.is_playing))

    def test_a_failure_while_waiting_for_a_clip_does_not_end_the_player(self):
        self.player.start()
        with patch.object(self.sd, "wait", side_effect=RuntimeError("device vanished")):
            with self.assertLogs("voxpipe.tts.audio", level="ERROR"):
                self.player.enqueue("a", 24_000)
                self.assertTrue(wait_until(lambda: not self.player.is_playing))
        self.assertTrue(self.player._thread.is_alive())
        self.player.enqueue("b", 24_000)
        self.assertTrue(self.sd.wait_for_clip(2))

    def test_playback_error_does_not_end_the_player(self):
        self.player.start()
        with patch.object(self.sd, "play", side_effect=RuntimeError("device gone")):
            with self.assertLogs("voxpipe.tts.audio", level="ERROR"):
                self.player.enqueue("a", 24_000)
                self.assertTrue(wait_until(lambda: not self.player.is_playing))
        self.player.enqueue("b", 24_000)
        self.assertTrue(self.sd.wait_for_clip(1))

    def test_stop_ends_the_thread_and_is_idempotent(self):
        self.player.start()
        self.player.start()
        thread = self.player._thread
        self.player.stop()
        self.player.stop()
        self.assertFalse(thread.is_alive())


class FakePlayer:
    """The slice of AudioPlayer that TTSBase relies on."""

    def __init__(self):
        self.generation = 0
        self.clips = []
        self.started = self.stopped = False

    @property
    def is_playing(self):
        return False

    def enqueue(self, audio, sample_rate, generation=None):
        if generation is not None and generation != self.generation:
            return False
        self.clips.append(audio)
        return True

    def stop_playback(self):
        self.generation += 1

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class RecordingTTS(TTSBase):
    """Synthesises text into itself, optionally pausing mid-synthesis."""

    @classmethod
    def from_config(cls, config):
        return cls(FakePlayer())

    def __init__(self, player):
        super().__init__(player)
        self.hold = threading.Event()
        self.hold.set()
        self.synthesising = threading.Event()
        self.options = []

    def synthesize(self, text, **options):
        self.options.append(options)
        self.synthesising.set()
        self.hold.wait(timeout=5)
        if text == "boom":
            raise RuntimeError("synthesis failed")
        return ([] if text == "..." else [text]), 24_000


class TestTTSQueue(unittest.TestCase):
    def setUp(self):
        self.player = FakePlayer()
        self.tts = RecordingTTS(self.player)
        self.addCleanup(self.tts.stop)

    def _settle(self):
        self.assertTrue(wait_until(lambda: not self.tts.is_speaking))

    def test_sentences_are_spoken_in_order(self):
        for sentence in ("one", "two", "three"):
            self.tts(sentence)
        self._settle()
        self.assertEqual(self.player.clips, [["one"], ["two"], ["three"]])

    def test_options_reach_synthesize(self):
        self.tts("hi", voice="bf_emma", speed=1.2)
        self._settle()
        self.assertEqual(self.tts.options, [{"voice": "bf_emma", "speed": 1.2}])

    def test_empty_audio_is_not_queued(self):
        self.tts("...")
        self._settle()
        self.assertEqual(self.player.clips, [])

    def test_is_speaking_covers_synthesis_in_progress(self):
        self.tts.hold.clear()
        self.tts("slow")
        self.assertTrue(self.tts.synthesising.wait(timeout=2))
        self.assertTrue(self.tts.is_speaking)
        self.tts.hold.set()
        self._settle()

    def test_interrupt_discards_the_sentence_being_synthesised_and_those_behind_it(self):
        self.tts.hold.clear()
        for sentence in ("one", "two", "three"):
            self.tts(sentence)
        self.assertTrue(self.tts.synthesising.wait(timeout=2))

        self.tts.interrupt()
        self.tts.hold.set()
        self._settle()

        self.assertEqual(self.player.clips, [], "audio finished after the interruption must not be played")
        self.assertEqual(len(self.tts.options), 1, "sentences behind it are not even synthesised")

        self.tts("fresh")
        self._settle()
        self.assertEqual(self.player.clips, [["fresh"]])

    def test_interrupt_flag_on_call(self):
        self.tts("old")
        self._settle()
        generation = self.player.generation
        self.tts("new", interrupt=True)
        self._settle()
        self.assertEqual(self.player.generation, generation + 1)
        self.assertEqual(self.player.clips[-1], ["new"])

    def test_failed_synthesis_does_not_stop_later_sentences(self):
        with self.assertLogs(level="ERROR"):
            self.tts("boom")
            self.tts("after")
            self._settle()
        self.assertEqual(self.player.clips, [["after"]])

    def test_context_manager_starts_and_stops_the_player(self):
        with RecordingTTS(FakePlayer()) as tts:
            self.assertTrue(tts.audio_player.started)
        self.assertTrue(tts.audio_player.stopped)


class TestKokoroTextCleaning(unittest.TestCase):
    def test_espeak_is_only_located_on_windows(self):
        from voxpipe.tts import model

        with patch.object(model.os, "name", "posix"), patch.dict(model.os.environ, {}, clear=True):
            model._locate_espeak()
            self.assertNotIn("PHONEMIZER_ESPEAK_LIBRARY", model.os.environ)

    def test_markup_and_emoji_patterns(self):
        from voxpipe.tts.model import _MARKUP, _SYMBOLS_AND_EMOJI

        cleaned = _SYMBOLS_AND_EMOJI.sub("", _MARKUP.sub("", "**Sure** _thing_ `now` \U0001F600"))
        self.assertEqual(cleaned.strip(), "Sure thing now")


class TestTTSLifecycle(unittest.TestCase):
    def setUp(self):
        self.player = FakePlayer()
        self.tts = RecordingTTS(self.player)
        self.addCleanup(self.tts.stop)

    def _settle(self):
        self.assertTrue(wait_until(lambda: not self.tts.is_speaking))

    def test_speaking_starts_the_threads_on_first_use(self):
        self.assertFalse(self.player.started)
        self.tts("hello")
        self._settle()
        self.assertTrue(self.player.started, "text in, speech out must work without an explicit start()")
        self.assertEqual(self.player.clips, [["hello"]])

    def test_it_can_be_stopped_and_used_again(self):
        self.tts("one")
        self._settle()
        self.tts.stop()
        self.tts("two")
        self._settle()
        self.assertEqual(self.player.clips, [["one"], ["two"]])

    def test_stop_forgets_sentences_that_were_never_synthesised(self):
        self.tts.hold.clear()
        for sentence in ("one", "two", "three"):
            self.tts(sentence)
        self.assertTrue(self.tts.synthesising.wait(timeout=2))
        self.tts.stop()
        self.assertFalse(self.tts.is_speaking, "cancelled sentences must not count as speech for ever")
        self.tts.hold.set()
        time.sleep(0.05)  # the sentence that was mid-synthesis finishes; it belongs to the old run
        self.assertFalse(self.tts.is_speaking)
        self.assertGreaterEqual(self.tts._pending, 0)

    def test_text_tied_to_an_interrupted_run_of_speech_is_dropped(self):
        run = self.tts.generation
        self.tts("before", generation=run)
        self._settle()
        self.tts.interrupt()
        self.tts("late", generation=run)
        self.tts("fresh")
        self._settle()
        self.assertEqual(self.player.clips, [["before"], ["fresh"]])


if __name__ == "__main__":
    unittest.main()
