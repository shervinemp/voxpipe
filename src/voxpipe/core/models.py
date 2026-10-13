"""Typed configuration schema.

Every section rejects unknown keys so a typo in ``config.yaml`` fails at
startup with the offending key named, instead of being silently ignored.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationHistoryConfig(_Section):
    enabled: bool = False
    db_path: str = "data/conversations.db"
    max_entries: int = Field(default=1000, ge=1)
    top_k: int = Field(default=2, ge=0, description="Recalled snippets per turn; 0 disables recall.")
    ttl_days: float = Field(default=30, ge=0, description="0 keeps entries forever.")


class LocalModelEntry(_Section):
    model_path: Optional[str] = None
    n_ctx: int = Field(default=4096, ge=256)
    max_tokens: int = Field(default=512, ge=1)
    decoder: str = "general"
    type_k: Optional[str] = None
    type_v: Optional[str] = None

    @model_validator(mode="after")
    def _reply_fits_the_context(self) -> "LocalModelEntry":
        if self.max_tokens >= self.n_ctx:
            raise ValueError("max_tokens must be smaller than n_ctx, or no room is left for the prompt")
        return self


class LiteLLMConfig(_Section):
    """Settings handed to LiteLLMProvider, which owns their validation."""

    provider: str
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: float = 60.0
    n_ctx: int = 16_384
    max_tokens: int = 1_024


class LLMConfig(_Section):
    backend: Literal["local", "litellm"] = "local"
    model: str = Field(description="Key into `local` selecting the GGUF model for the local backend.")
    max_tool_iterations: int = Field(
        default=1, ge=0, description="0 = tools disabled, N = up to N tool passes plus a final answer pass"
    )
    max_turns: int = Field(
        default=10, ge=1, description="Turns (a user message and its answer) kept in the model's context."
    )
    tool_timeout: float = Field(default=10.0, gt=0, description="Seconds allowed for one pass of tool calls.")
    local: dict[str, LocalModelEntry] = Field(default_factory=dict)
    litellm: Optional[LiteLLMConfig] = None


class TTSConfig(_Section):
    provider: str
    weights_dir: str


class HotkeyConfig(_Section):
    enable: bool = True
    push_to_talk: Optional[str] = None
    press_to_reset: Optional[str] = None


class CrisperWhisperConfig(_Section):
    model: str = "nyralabs/CrisperWhisper2.0_small"
    backend: str = "auto"
    mode: Literal["verbatim", "intended"] = "verbatim"
    language: str = "en"


class ASRConfig(_Section):
    provider: str
    weights_dir: str
    sound_device: Optional[int | str] = Field(
        default=None, description="Input device index or name fragment; None uses the system default."
    )
    vad_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    trailing_silence_ms: int = Field(default=800, ge=100, le=10_000)
    leading_silence_ms: int = Field(default=1000, ge=0, le=10_000)
    max_segment_duration: float = Field(default=0.0, ge=0.0, le=300.0)
    crisperwhisper: CrisperWhisperConfig = Field(default_factory=CrisperWhisperConfig)


class AppConfig(_Section):
    llm: LLMConfig
    tts: TTSConfig
    asr: ASRConfig
    hotkeys: HotkeyConfig = Field(default_factory=HotkeyConfig)
    conversation_history: ConversationHistoryConfig = Field(default_factory=ConversationHistoryConfig)
