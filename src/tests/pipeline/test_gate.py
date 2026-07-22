"""Tests for transcript gate filtering."""
import unittest


class TestTranscriptGate(unittest.TestCase):
    from voxpipe.pipeline.gate import qualify_transcript
    _gate = staticmethod(qualify_transcript)

    def test_returns_none_for_empty(self):
        self.assertEqual(self._gate(""), (None, None))

    def test_returns_none_for_noise(self):
        self.assertEqual(self._gate("..."), (None, None))
        self.assertEqual(self._gate("   "), (None, None))

    def test_returns_text_for_valid(self):
        text, ann = self._gate("hello world")
        self.assertEqual(text, "hello world")
        self.assertIsNone(ann)

    def test_single_word_passes(self):
        text, ann = self._gate("yes")
        self.assertEqual(text, "yes")

    def test_complete_sentence_passes(self):
        text, ann = self._gate("I think that")
        self.assertEqual(text, "I think that")
        self.assertIsNone(ann)

    def test_no_annotation_for_complete_sentence(self):
        text, ann = self._gate("I think that is correct.")
        self.assertEqual(text, "I think that is correct.")
        self.assertIsNone(ann)
