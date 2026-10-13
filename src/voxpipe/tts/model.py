"""Text-to-speech providers."""

import os
import re
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from ..core.config import get_config
from ..core.downloader import ensure_downloaded
from ..core.exceptions import ProviderError
from ..core.models import AppConfig, TTSConfig
from .audio import AudioPlayer
from .base import TTSBase

_ESPEAK_ENV_VAR = "PHONEMIZER_ESPEAK_LIBRARY"
_WINDOWS_ESPEAK_LIBRARIES = (
    r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
    r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll",
)

_MARKUP = re.compile(r"[*_~`´<>]")
_SYMBOLS_AND_EMOJI = re.compile(r"[\U00002600-\U000027BF\U0001F300-\U0010FFFF]")


def _locate_espeak() -> None:
    """Point the phonemizer at eSpeak NG on Windows, where it is not on the library search path.

    Elsewhere the phonemizer finds the shared library itself; an explicit
    ``PHONEMIZER_ESPEAK_LIBRARY`` always wins.
    """
    if os.name != "nt" or os.environ.get(_ESPEAK_ENV_VAR):
        return
    for library in _WINDOWS_ESPEAK_LIBRARIES:
        if os.path.exists(library):
            os.environ[_ESPEAK_ENV_VAR] = library
            directory = os.path.dirname(library)
            if directory not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
            return


class Kokoro(TTSBase):
    """Kokoro ONNX speech synthesis."""

    sample_rate: int = 24_000

    def __init__(
        self,
        weights_dir: str,
        *,
        voice: str = "af_heart",
        language: str = "en-us",
        speed: float = 1.0,
        audio_player: Optional[AudioPlayer] = None,
    ):
        paths = ensure_downloaded("Kokoro", local_dir=weights_dir)
        _locate_espeak()
        from kokoro_onnx import Kokoro as KokoroONNX
        from kokoro_onnx.tokenizer import Tokenizer

        self.kokoro = KokoroONNX(model_path=paths["model"], voices_path=paths["voices"])
        self.tokenizer = Tokenizer()
        self.voice = voice
        self.language = language
        self.speed = speed
        super().__init__(audio_player or AudioPlayer())

    @classmethod
    def from_config(cls, config: TTSConfig) -> "Kokoro":
        return cls(config.weights_dir)

    @classmethod
    def download(cls, config: TTSConfig) -> None:
        """Fetch the model files into ``tts.weights_dir`` if they are not already there."""
        ensure_downloaded("Kokoro", local_dir=config.weights_dir)

    def synthesize(
        self, text: str, *, voice: Optional[str] = None, language: Optional[str] = None, speed: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        silence = np.array([], dtype=np.float32), self.sample_rate
        text = _SYMBOLS_AND_EMOJI.sub("", _MARKUP.sub("", text)).strip()
        if not text:
            return silence
        phonemes = self.tokenizer.phonemize(text, lang=language or self.language)
        if not phonemes.strip():
            self.logger.warning("Empty phonemes. Skipping TTS.")
            return silence

        self.logger.debug("TTS synthesizing %d chars -> %d phonemes", len(text), len(phonemes))
        samples, sample_rate = self.kokoro.create(
            phonemes, voice=voice or self.voice, speed=speed or self.speed, is_phonemes=True, trim=False
        )
        self.logger.debug("TTS generated %d samples (%.1fs)", len(samples), len(samples) / sample_rate)
        return samples, sample_rate


class TTSProviders:
    """Registry of text-to-speech providers, by the name used in ``tts.provider``."""

    _registry: Dict[str, Type[TTSBase]] = {"Kokoro": Kokoro}

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._registry)

    @classmethod
    def get(cls, name: str) -> Type[TTSBase]:
        try:
            return cls._registry[name]
        except KeyError:
            raise ProviderError(f"Unknown TTS provider: {name!r}. Expected one of: {', '.join(cls.names())}.") from None

    @classmethod
    def create(cls, name: Optional[str] = None, config: Optional[AppConfig] = None) -> TTSBase:
        """Build the provider ``name`` (default: the configured one) from the configuration."""
        tts_config = (config or get_config()).tts
        return cls.get(name or tts_config.provider).from_config(tts_config)
