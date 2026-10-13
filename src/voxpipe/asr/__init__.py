"""Speech recognition: microphone capture, voice activity detection and ASR providers."""

from .model import ASRProviders
from .vad import AudioGate, Silero

__all__ = ["ASRProviders", "AudioGate", "Silero"]
