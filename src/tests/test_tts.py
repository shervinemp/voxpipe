import unittest
from unittest.mock import patch
import numpy as np

from voxpipe.tts.model import TTSProviders


class TestTTS(unittest.TestCase):
    @patch("voxpipe.tts.model.AudioPlayer")
    def test_generate_audio(self, mock_audio_player):
        for provider_name in dir(TTSProviders):
            if provider_name.startswith("__"):
                continue

            with self.subTest(provider=provider_name):
                tts_cls = getattr(TTSProviders, provider_name)

                tts_cls.download()
                tts = tts_cls()
                text = "This is a test sentence for the voice detection system."

                samples, sample_rate = tts._synthesize(
                    text, voice="af_heart", language="en-us",
                    speed=1.0, interrupt=False,
                )

                self.assertIsNotNone(samples)
                self.assertGreater(len(samples), 0)
                self.assertEqual(sample_rate, 24000)

                import os, soundfile as sf
                output_path = os.path.join(
                    os.path.dirname(__file__),
                    f"test_audio_{provider_name.lower()}.wav",
                )
                sf.write(output_path, samples, sample_rate)
                self.assertTrue(os.path.exists(output_path))
