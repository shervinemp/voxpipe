"""Speech recognition backends."""

from .base import ModelBase
from .crisperwhisper import CrisperWhisper
from .parakeetv2 import ParakeetV2

__all__ = ["ModelBase", "ParakeetV2", "CrisperWhisper"]
