import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from voxpipe.asr.models import ParakeetV2
from voxpipe.tts.model import TTSProviders
from voxpipe.core.config import config


class TestASR(unittest.TestCase):
    @patch("voxpipe.asr.models.parakeetv2.ParakeetV2._inputstream")
    @patch("voxpipe.asr.models.parakeetv2.Silero")
    def test_parakeet_v2_transcribe(self, mock_silero, mock_input_stream):
        """Generate TTS audio in memory, transcribe with ASR, check similarity."""
        mock_vad = MagicMock()
        mock_silero.return_value = mock_vad
        mock_input_stream.return_value = MagicMock()
        asr = ParakeetV2()

        original = "this is a test sentence for the voice detection system"

        with patch("voxpipe.tts.model.AudioPlayer"):
            tts_cls = getattr(TTSProviders, config.get("tts.provider"))
            tts_cls.download()
            tts = tts_cls()
            samples, sr = tts._synthesize(original, voice="af_heart", language="en-us", speed=1.0, interrupt=False)

        # ONNX ASR expects mono
        if samples.ndim > 1 and samples.shape[1] > 1:
            samples = np.mean(samples, axis=1, keepdims=False).astype(samples.dtype)

        transcript = asr._model.recognize(samples, sample_rate=sr)

        self.assertIsNotNone(transcript)
        self.assertGreater(len(transcript), 0)

        similarity = len(
            set(transcript.lower().split()) & set(original.split())
        ) / len(set(original.split()))
        self.assertGreater(similarity, 0.8)
