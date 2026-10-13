"""Registry of speech recognition providers."""

from typing import Callable, Dict, List, Optional

from ..core.config import get_config
from ..core.exceptions import ProviderError
from ..core.models import AppConfig, ASRConfig
from .models import CrisperWhisper, ParakeetV2
from .models.base import ModelBase
from .vad import Silero


def _parakeet(config: ASRConfig) -> ModelBase:
    return ParakeetV2(config.sound_device, vad=Silero.from_config(config))


def _crisperwhisper(config: ASRConfig) -> ModelBase:
    return CrisperWhisper(config.sound_device, vad=Silero.from_config(config), settings=config.crisperwhisper)


class ASRProviders:
    """Builds the recogniser named by ``asr.provider``."""

    _factories: Dict[str, Callable[[ASRConfig], ModelBase]] = {
        "ParakeetV2": _parakeet,
        "CrisperWhisper": _crisperwhisper,
    }

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._factories)

    @classmethod
    def create(cls, name: Optional[str] = None, config: Optional[AppConfig] = None) -> ModelBase:
        """Build the provider ``name`` (default: the configured one) from the configuration."""
        asr_config = (config or get_config()).asr
        name = name or asr_config.provider
        try:
            factory = cls._factories[name]
        except KeyError:
            raise ProviderError(f"Unknown ASR provider: {name!r}. Expected one of: {', '.join(cls.names())}.") from None
        return factory(asr_config)
