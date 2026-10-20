"""Tests for the transcript gate."""
import unittest

from voxpipe.pipeline.transcript import qualify_transcript


class TestTranscriptGate(unittest.TestCase):
    def test_noise_is_dropped(self):
        for transcript in ("", "   ", "...", "?!", None):
            with self.subTest(transcript=transcript):
                self.assertIsNone(qualify_transcript(transcript))

    def test_speech_passes_tidied_up(self):
        for transcript, expected in [
            ("hello world", "hello world"),
            ("  yes ", "yes"),
            ("I think that is correct.", "I think that is correct."),
        ]:
            with self.subTest(transcript=transcript):
                self.assertEqual(qualify_transcript(transcript), expected)

    def test_numbers_are_speech_too(self):
        for transcript in ("42", "2024.", "9 1 1", "10:30", "$5"):
            with self.subTest(transcript=transcript):
                self.assertEqual(qualify_transcript(transcript), transcript)


if __name__ == "__main__":
    unittest.main()
