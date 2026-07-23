class VoiceControlError(Exception):
    """Base exception for all voice-control errors."""


VoxpipeError = VoiceControlError


class ConfigError(VoiceControlError):
    """Configuration loading or validation error."""


class ModelError(VoiceControlError):
    """Model download, load, or verification error."""


class ProviderError(VoiceControlError):
    """Provider initialization or configuration error."""


class ASRError(VoiceControlError):
    """Speech recognition or VAD error."""


class LLMError(VoiceControlError):
    """Language model inference error."""


class TTSError(VoiceControlError):
    """Text-to-speech synthesis or playback error."""


class StorageError(VoiceControlError):
    """Storage backend (graph/vector) error."""


class ToolError(VoiceControlError):
    """Tool call execution or validation error."""
