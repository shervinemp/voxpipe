from typing import Any, Callable, Generator, Iterable

import numpy as np

from .base import ModelBase
from .parakeetv2 import Silero
from ...core.utils import get_logger


class CrisperWhisper(ModelBase):
    """Speech recognition via the CrisperWhisper 2.0 pipeline.

    Wraps the `crisperwhisper` package (HuggingFace ID
    `nyralabs/CrisperWhisper2.0_small`, configurable via
    `asr.crisperwhisper.model`). Produces verbatim transcripts with
    fillers, repetitions and vocal events, or clean intended text.

    Note: the model weights are released under the Nyra Health
    Non-Commercial Research License; commercial use requires a license.
    """

    def __init__(self, sound_device: int | str = 0):
        from crisperwhisper import CrisperWhisperModel

        self.logger = get_logger(__name__)

        # Resolve string device name to integer index
        if isinstance(sound_device, str):
            import sounddevice as sd
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if sound_device.lower() in dev["name"].lower():
                    sound_device = i
                    break
            else:
                self.logger.warning(
                    f"Sound device '{sound_device}' not found. Using default (0)."
                )
                sound_device = 0

        from voxpipe.core.config import config as _cfg
        self._model_id = _cfg.get("asr.crisperwhisper.model")
        self._backend = _cfg.get("asr.crisperwhisper.backend", "auto")
        self._mode = _cfg.get("asr.crisperwhisper.mode", "verbatim")
        self._language = _cfg.get("asr.crisperwhisper.language", "en")

        self._model = CrisperWhisperModel(self._model_id, backend=self._backend)

        # Read VAD settings from config before initializing Silero
        vad_threshold = _cfg.get("asr.vad_threshold", 0.4)
        trailing_ms = _cfg.get("asr.trailing_silence_ms", 800)
        leading_ms = _cfg.get("asr.leading_silence_ms", 1000)
        max_segment = _cfg.get("asr.max_segment_duration", 0.0)
        self._vad = Silero(
            vad_threshold=vad_threshold,
            trailing_silence_duration=trailing_ms / 1000.0,
            leading_silence_duration=leading_ms / 1000.0,
            max_segment_duration=max_segment,
        )

        super().__init__(sound_device)

    def _consume(self, chunk: Iterable[float]):
        if self._is_muted.is_set():
            return
        self._vad(chunk)

    def _produce(self) -> Generator[str, None, None]:
        for e in self._vad:
            result = self._model.transcribe(
                e,
                sr=self._vad._model.SAMPLE_RATE,
                language=self._language,
                mode=self._mode,
            )
            text = result.text
            self.logger.debug(
                "CrisperWhisper returned %r (len=%d)",
                text[:40] if text else text,
                len(text) if text else 0,
            )
            yield text

    def _inputstream(self, sound_device: int, callback: Callable):
        import sounddevice as sd

        return sd.InputStream(
            samplerate=self._vad._model.SAMPLE_RATE,
            blocksize=self._vad._model.HOP_SIZE,
            device=sound_device,
            channels=1,
            callback=callback,
        )

    def disable_w_passthrough(self, value: Any = None):
        value = np.zeros(
            (self._input_stream.blocksize, self._input_stream.channels),
            dtype=np.float32,
        )
        super().disable_w_passthrough(value)
        self._vad.flush()
