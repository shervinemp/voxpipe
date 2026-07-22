import threading
import unittest
from unittest.mock import MagicMock


class TestPipelineCallback(unittest.TestCase):

    def _make(self, **kw):
        from voxpipe.pipeline.pipeline import Pipeline
        p = Pipeline.__new__(Pipeline)
        p.logger = MagicMock()
        p.events = MagicMock()
        
        p.memory = None
        p._response_parts = []
        p._llm_busy = False
        p._interrupt_event = threading.Event()
        p._interrupted_at = None
        p._match_command = MagicMock(return_value=False)
        p.session = MagicMock()
        p.session.return_value = iter([])
        p.tts = None
        for k, v in kw.items():
            setattr(p, k, v)
        return p

    def test_callback_gate_filters_noise(self):
        pipe = self._make()
        pipe._callback("...")
        pipe.session.assert_not_called()

    def test_callback_valid_text_calls_session(self):
        pipe = self._make()
        pipe.session.return_value = iter(["hello world"])
        pipe._callback("hello world")
        pipe.session.assert_called_once()

    def test_callback_match_command_skips_llm(self):
        pipe = self._make(_match_command=MagicMock(return_value=True))
        pipe._callback("stop")
        pipe.session.assert_not_called()

    def test_stale_interrupt_cleared_before_llm(self):
        """A stale interrupt from a prior speech onset must not cancel the utterance."""
        pipe = self._make()
        pipe.session.return_value = iter(["hello world"])
        pipe._interrupt_event.set()

        pipe._callback("hello world")

        self.assertFalse(pipe._interrupt_event.is_set(),
                         "interrupt still set after _callback entry")
        pipe.session.assert_called_once()

    def test_fresh_interrupt_during_stream_breaks_loop(self):
        """A new speech onset during TTS must still interrupt (barge-in)."""
        pipe = self._make()
        pipe.session.return_value = iter(["sentence one. ", "sentence two."])
        pipe._interrupt_event = MagicMock()
        pipe._interrupt_event.is_set.side_effect = [False, True]
        pipe._interrupt_event.clear.return_value = None

        pipe._callback("test")

        self.assertEqual(pipe._response_parts, ["sentence one."])

    def test_barge_in_during_tts_stops_playback(self):
        """Barge-in must stop TTS and break the callback loop mid-stream."""
        pipe = self._make()
        pipe.session.return_value = iter(["part one. ", "part two. ", "part three."])
        pipe._interrupt_event = MagicMock()
        pipe._interrupt_event.is_set.side_effect = [False, True, False]
        pipe._interrupt_event.clear.return_value = None

        pipe._callback("test")

        self.assertEqual(len(pipe._response_parts), 1,
                         "only first sentence should have been processed before interrupt")
        self.assertEqual(pipe._response_parts[0], "part one.")

    def test_multiple_sentences_without_interrupt(self):
        """Without stale interrupt, multiple sentences flow through."""
        pipe = self._make()
        pipe.session.return_value = iter(["first. ", "second. ", "third."])

        pipe._callback("test")

        self.assertEqual(pipe._response_parts, ["first.", "second.", "third."])
