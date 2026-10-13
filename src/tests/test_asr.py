import unittest

from ._integration import KOKORO, PARAKEET, requires_model, transcribe, tts

pytestmark = requires_model("Kokoro", KOKORO + PARAKEET)


class TestASR(unittest.TestCase):
    def test_parakeet_v2_transcribe(self):
        """Synthesise speech in memory, transcribe it, and check the words survive."""
        original = "this is a test sentence for the voice detection system"
        samples, sample_rate = tts().synthesize(original)

        transcript = transcribe(samples, sample_rate)

        expected = set(original.split())
        similarity = len(set(transcript.lower().split()) & expected) / len(expected)
        self.assertGreater(similarity, 0.8, transcript)
