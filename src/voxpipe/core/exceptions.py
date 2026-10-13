"""Exception hierarchy for voxpipe."""


class VoxpipeError(Exception):
    """Base class for every error raised by voxpipe."""


class ConfigError(VoxpipeError):
    """Configuration loading or validation error."""


class ModelError(VoxpipeError):
    """Model download, load, or verification error."""


class ProviderError(VoxpipeError):
    """Provider initialization or configuration error."""


class ASRError(VoxpipeError):
    """Speech recognition or VAD error."""


class LLMError(VoxpipeError):
    """Language model inference error."""


class TTSError(VoxpipeError):
    """Text-to-speech synthesis or playback error."""


class StorageError(VoxpipeError):
    """Storage backend error."""


class ToolError(VoxpipeError):
    """Tool call execution or validation error."""
