"""CrisperWhisper 2.0 speech recognition: verbatim transcripts with fillers and vocal events."""

from typing import Any, Optional

import numpy as np

from ...core.models import CrisperWhisperConfig
from ..vad import Silero
from .base import ModelBase, SoundDevice


class CrisperWhisper(ModelBase):
    """Wraps the ``crisperwhisper`` package.

    Produces verbatim transcripts with fillers, repetitions and vocal events,
    or clean intended text, depending on ``settings.mode``.

    Note: the model weights are released under the Nyra Health
    Non-Commercial Research License; commercial use requires a license.
    """

    def __init__(
        self,
        sound_device: SoundDevice = None,
        *,
        vad: Optional[Silero] = None,
        settings: Optional[CrisperWhisperConfig] = None,
        model: Any = None,
    ):
        self.settings = settings or CrisperWhisperConfig()
        if model is None:
            from crisperwhisper import CrisperWhisperModel

            model = CrisperWhisperModel(self.settings.model, backend=self.settings.backend)
        self._model = model
        super().__init__(vad or Silero(), sound_device)

    def _transcribe(self, audio: np.ndarray) -> str:
        result = self._model.transcribe(
            audio,
            sr=self.vad.sample_rate,
            language=self.settings.language,
            mode=self.settings.mode,
        )
        return result.text
