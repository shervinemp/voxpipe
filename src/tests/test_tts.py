import os
import tempfile
import unittest

import soundfile as sf

from ._integration import requires_model, tts

pytestmark = requires_model("Kokoro")


class TestTTS(unittest.TestCase):
    def test_generate_audio(self):
        samples, sample_rate = tts().synthesize("This is a test sentence for the voice detection system.")

        self.assertGreater(len(samples), 0)
        self.assertEqual(sample_rate, 24000)

        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "speech.wav")
            sf.write(output_path, samples, sample_rate)
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_voice_and_speed_options(self):
        normal, _ = tts().synthesize("The quick brown fox jumps over the lazy dog.")
        fast, _ = tts().synthesize("The quick brown fox jumps over the lazy dog.", speed=1.5)
        self.assertLess(len(fast), len(normal))

    def test_unpronounceable_text_is_silence(self):
        samples, _ = tts().synthesize("*** \U0001F600 ***")
        self.assertEqual(len(samples), 0)
