"""Voice activity detection tests with a fake Silero model."""
import threading
import unittest

import numpy as np

from voxpipe.asr.vad import Silero
from voxpipe.core.config import load_config

from ._fakes import HOP, FakeVadModel, loud, quiet


def collect(vad, chunks, flush=False):
    """Feed chunks, close the stream, and return every segment produced."""
    for chunk in chunks:
        vad.feed(chunk)
    if flush:
        vad.flush()
    vad.close()
    return list(vad)


class TestSilero(unittest.TestCase):
    def make(self, **kwargs):
        kwargs.setdefault("leading_silence_duration", 0.1)  # 3 chunks
        kwargs.setdefault("trailing_silence_duration", 0.3)  # 10 chunks
        return Silero(model=FakeVadModel(), **kwargs)

    def test_silence_produces_nothing(self):
        self.assertEqual(collect(self.make(), quiet(100)), [])

    def test_speech_followed_by_silence_is_one_segment(self):
        vad = self.make()
        segments = collect(vad, quiet(20) + loud(30) + quiet(20))
        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment.dtype, np.float32)
        self.assertEqual(segment.ndim, 1)
        # 3 chunks of lead-in, all 30 chunks of speech, and the trailing silence that closed it.
        self.assertEqual(len(segment), (3 + 30 + vad._trailing_chunks) * HOP)

    def test_no_speech_is_clipped_without_lead_in(self):
        vad = self.make(leading_silence_duration=0.0)
        (segment,) = collect(vad, quiet(20) + loud(30) + quiet(20))
        self.assertEqual(int((segment > 0.1).sum()), 30 * HOP)

    def test_segment_keeps_leading_audio_without_duplicating_the_onset_chunk(self):
        vad = self.make(leading_silence_duration=0.5)  # 15 chunks of lead-in
        marked = quiet(10) + [np.full((HOP, 1), 0.2 + i / 100, dtype=np.float32) for i in range(8)] + quiet(15)
        (segment,) = collect(vad, marked)
        speech_values = [round(float(v), 2) for v in segment[::HOP] if v > 0.1]
        self.assertEqual(speech_values, [round(0.2 + i / 100, 2) for i in range(8)], "each chunk exactly once")
        self.assertEqual(float(segment[0]), 0.0, "quiet lead-in precedes the speech")

    def test_short_noise_is_ignored(self):
        vad = self.make()
        self.assertEqual(collect(vad, quiet(10) + loud(Silero.MIN_SPEECH_CHUNKS - 1) + quiet(30)), [])

    def test_two_utterances_are_two_segments(self):
        segments = collect(self.make(), loud(20) + quiet(20) + loud(20) + quiet(20))
        self.assertEqual(len(segments), 2)

    def test_brief_pause_does_not_split_an_utterance(self):
        segments = collect(self.make(), loud(20) + quiet(5) + loud(20) + quiet(20))
        self.assertEqual(len(segments), 1)

    def test_flush_ends_the_segment_at_once(self):
        """Push-to-talk release: the utterance is complete without waiting for trailing silence."""
        vad = self.make()
        (segment,) = collect(vad, loud(20), flush=True)
        self.assertEqual(len(segment), 20 * HOP)

    def test_flush_discards_a_half_detected_onset(self):
        vad = self.make()
        for chunk in loud(3):
            vad.feed(chunk)
        vad.flush()
        for chunk in loud(3):
            vad.feed(chunk)
        vad.close()
        self.assertEqual(list(vad), [], "3 + 3 chunks either side of a flush are not one 6-chunk onset")

    def test_flush_keeps_utterances_from_different_presses_apart(self):
        vad = self.make(leading_silence_duration=1.0)
        for chunk in loud(20, value=0.3):
            vad.feed(chunk)
        vad.flush()
        for chunk in loud(20, value=0.7):
            vad.feed(chunk)
        vad.flush()
        vad.close()
        first, second = list(vad)
        self.assertTrue(np.allclose(first, 0.3))
        self.assertTrue(np.allclose(second, 0.7), "no audio from the previous press leaks into the next")

    def test_max_segment_duration_splits_long_speech(self):
        vad = self.make(max_segment_duration=0.64)  # 20 chunks
        segments = collect(vad, loud(70) + quiet(20))
        self.assertGreaterEqual(len(segments), 3)
        self.assertTrue(all(len(s) <= 20 * HOP + vad._trailing_chunks * HOP for s in segments))

    def test_onset_callback_fires_once_per_utterance(self):
        onsets = []
        vad = self.make(on_speech_onset=lambda: onsets.append(1))
        collect(vad, loud(20) + quiet(20) + loud(20) + quiet(20))
        self.assertEqual(len(onsets), 2)

    def test_audio_level_callback(self):
        levels = []
        vad = self.make()
        vad.on_audio_level = lambda rms, probability: levels.append((round(rms, 2), round(probability, 2)))
        collect(vad, quiet(1) + loud(1))
        self.assertEqual(levels, [(0.0, 0.05), (0.5, 0.9)])

    def test_failing_level_callback_is_harmless(self):
        vad = self.make()
        vad.on_audio_level = lambda rms, probability: 1 / 0
        self.assertEqual(len(collect(vad, loud(20) + quiet(20))), 1)

    def test_stereo_and_flat_chunks(self):
        vad = self.make()
        stereo = [np.full((HOP, 2), 0.5, dtype=np.float32) for _ in range(20)]
        flat = [np.zeros(HOP, dtype=np.float32) for _ in range(20)]
        (segment,) = collect(vad, stereo + flat)
        self.assertEqual(segment.ndim, 1)

    def test_model_receives_context_plus_chunk(self):
        vad = self.make()
        collect(vad, loud(2))
        frames = vad._model.frames
        self.assertEqual(frames[0].shape, (1, FakeVadModel.CONTEXT_SIZE + HOP))
        self.assertTrue(np.all(frames[0][0, : FakeVadModel.CONTEXT_SIZE] == 0), "first context is silence")
        self.assertTrue(np.all(frames[1][0, : FakeVadModel.CONTEXT_SIZE] == 0.5), "context carries over")

    def test_unreadable_chunk_is_skipped(self):
        vad = self.make()
        with self.assertLogs("voxpipe.asr.vad", level="WARNING"):
            segments = collect(vad, loud(10) + ["not audio"] + loud(10) + quiet(20))
        self.assertEqual(len(segments), 1)

    def test_backlog_is_bounded_and_feed_never_blocks(self):
        vad = self.make()
        with self.assertLogs("voxpipe.asr.vad", level="WARNING"):
            for chunk in quiet(Silero.MAX_BACKLOG_CHUNKS + 50):
                vad.feed(chunk)
        self.assertEqual(vad._queue.qsize(), Silero.MAX_BACKLOG_CHUNKS)
        vad.flush()  # control messages still get through a full backlog
        vad.close()
        self.assertEqual(vad._queue.qsize(), Silero.MAX_BACKLOG_CHUNKS + 2)

    def test_full_backlog_with_live_audio_does_not_deadlock(self):
        """The consumer once blocked forever putting its own end-of-segment marker on a full queue."""
        vad = self.make()
        backlog = quiet(10) + loud(30) + quiet(Silero.MAX_BACKLOG_CHUNKS - 40)
        for chunk in backlog:
            vad.feed(chunk)

        stop = threading.Event()

        def microphone():
            while not stop.is_set():
                vad.feed(quiet()[0])
                stop.wait(0.001)

        feeder = threading.Thread(target=microphone, daemon=True)
        feeder.start()
        segments = []
        consumer = threading.Thread(target=lambda: segments.append(next(iter(vad))), daemon=True)
        consumer.start()
        consumer.join(timeout=5)
        stop.set()
        feeder.join(timeout=2)

        self.assertFalse(consumer.is_alive(), "consumer deadlocked on a full queue")
        self.assertEqual(len(segments), 1)

    def test_audio_gate(self):
        class WakeWord:
            def __init__(self):
                self.seen = 0

            def process(self, chunk):
                self.seen += 1
                return "active" if self.seen == 5 else "idle"

        vad = self.make(audio_gate=WakeWord(), leading_silence_duration=0.0)
        for chunk in loud(4) + loud(1) + loud(20) + quiet(20):
            vad.feed(chunk)
        segments = iter(vad)
        first = next(segments)
        self.assertEqual(len(first), (20 + vad._trailing_chunks) * HOP, "audio before the gate opened is dropped")

        for chunk in loud(20) + quiet(20):  # no wake word this time
            vad.feed(chunk)
        vad.close()
        self.assertEqual(list(segments), [], "the gate closes again after each utterance")

    def test_from_config(self):
        asr_config = load_config().asr.model_copy(
            update={"vad_threshold": 0.7, "leading_silence_ms": 320, "trailing_silence_ms": 640, "max_segment_duration": 3.2}
        )
        vad = Silero.from_config(asr_config, model=FakeVadModel())
        self.assertEqual(vad.vad_threshold, 0.7)
        self.assertEqual(vad._leading_chunks, 10)
        self.assertEqual(vad._trailing_chunks, 21)
        self.assertEqual(vad._max_segment_chunks, 100)
        self.assertEqual((vad.sample_rate, vad.hop_size), (16_000, HOP))


class TestStreamingModel(unittest.TestCase):
    """The model is recurrent: it must be fed its own state back, chunk after chunk."""

    def make(self, **kwargs):
        kwargs.setdefault("leading_silence_duration", 0.0)
        kwargs.setdefault("trailing_silence_duration", 0.3)
        return Silero(model=FakeVadModel(), **kwargs)

    @staticmethod
    def run_through(vad, *groups):
        for group in groups:
            for item in group:
                vad.feed(item) if item is not None else vad.flush()
        vad.close()
        return list(vad)

    def test_state_and_context_are_carried_from_chunk_to_chunk(self):
        vad = self.make()
        self.run_through(vad, loud(8), quiet(12))  # the fake fails any call that brings the wrong state
        session = vad._model._model
        self.assertEqual(len(session.frames), 20, "one model run per chunk")
        self.assertEqual(session.fresh_starts, 1, "only the very first chunk starts from a blank state")

    def test_a_flush_starts_the_model_afresh(self):
        vad = self.make()
        self.run_through(vad, loud(8), [None], loud(8))
        session = vad._model._model
        self.assertEqual(session.fresh_starts, 2)
        self.assertFalse(session.frames[8][0, :64].any(), "no audio context is carried across the flush")

    def test_a_model_without_a_session_is_refused(self):
        from voxpipe.core.exceptions import ASRError

        class NoSession:
            SAMPLE_RATE, HOP_SIZE, CONTEXT_SIZE = 16_000, HOP, 64

        with self.assertRaises(ASRError):
            Silero(model=NoSession())

    def test_a_failing_onset_listener_does_not_end_recognition(self):
        def listener():
            raise RuntimeError("output device unplugged")

        vad = self.make(on_speech_onset=listener)
        with self.assertLogs("voxpipe.asr.vad", level="ERROR"):
            segments = self.run_through(vad, loud(8), quiet(12))
        self.assertEqual(len(segments), 1)

    def test_a_segment_cut_for_length_keeps_the_gate_open(self):
        class OpensOnce:
            def process(self, chunk):
                return "active"

        vad = self.make(max_segment_duration=0.64, audio_gate=OpensOnce())  # 20 chunks per segment
        segments = self.run_through(vad, loud(1), loud(70), quiet(12))
        self.assertEqual([len(s) // HOP for s in segments][:3], [20, 20, 20], "one utterance, cut three times")
        self.assertGreater(len(segments), 3)


if __name__ == "__main__":
    unittest.main()
