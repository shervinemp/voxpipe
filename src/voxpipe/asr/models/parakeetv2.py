"""NVIDIA Parakeet TDT v2 speech recognition through onnx-asr."""

from typing import Any, Optional

import numpy as np

from ..vad import Silero
from .base import ModelBase, SoundDevice


class ParakeetV2(ModelBase):
    MODEL_NAME = "nemo-parakeet-tdt-0.6b-v2"

    def __init__(self, sound_device: SoundDevice = None, *, vad: Optional[Silero] = None, model: Any = None):
        if model is None:
            from onnx_asr import load_model

            model = load_model(self.MODEL_NAME, quantization="int8")
        self._model = model
        super().__init__(vad or Silero(), sound_device)
        self._warm_up()

    def _warm_up(self) -> None:
        """Run one silent second through the model so the first utterance does not pay for cold start."""
        try:
            self._transcribe(np.zeros(self.vad.sample_rate, dtype=np.float32))
        except Exception:  # noqa: BLE001 - warm-up is best effort
            self.logger.warning("ASR warm-up failed (may affect first-utterance latency).", exc_info=True)

    def _transcribe(self, audio: np.ndarray) -> str:
        return self._model.recognize(audio, sample_rate=self.vad.sample_rate)
