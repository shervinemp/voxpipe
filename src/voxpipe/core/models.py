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


class TopicTrackingConfig(_Section):
    """Keep the active conversation apart from speech that belongs elsewhere.

    ``backend`` selects how relatedness is judged: ``lexical`` needs no extra
    packages, ``embedding`` uses a small fastembed model, and ``laya`` uses
    the Laya decision model.  ``threshold`` is in the scorer's own 0..1 scale;
    None picks a sensible default for the backend.
    """

    enabled: bool = False
    backend: Literal["lexical", "embedding", "laya"] = "lexical"
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ttl_sentences: int = Field(
        default=10, ge=1, description="Utterances an unused side buffer survives before it is dropped."
    )
    max_unrelated: int = Field(
        default=4, ge=1, description="Side buffers kept at once; the least recently used is dropped."
    )
    history_sentences: int = Field(
        default=8, ge=1, description="Recent utterances of a buffer compared with the new sentence."
    )
    promote_after: int = Field(
        default=0,
        ge=0,
        description="Consecutive utterances in a side buffer before it becomes the active topic; 0 never promotes.",
    )
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Optional[str] = None
    laya_model: Literal["auto", "english", "multilingual", "typed-decisions"] = "auto"
    laya_device: Optional[str] = None
    laya_preload: bool = False
    laya_max_len: int = Field(default=1024, ge=64, le=8192)


class AttentionConfig(_Section):
    """Addressee-aware floor control for busy, multi-person environments.

    ``busy_mode`` is the easy preset: it enables the addressee gate, waits for
    a possible other-speaker reply before speaking (``defer_ms``), and uses
    silence-biased thresholds.  Every setting can still be overridden
    individually; ``enabled`` turns the gate on without the busy preset.
    """

    busy_mode: bool = False
    enabled: bool = False
    defer_ms: Optional[int] = Field(
        default=None, ge=0, le=5000, description="Wait this long for a possible reply before speaking."
    )
    speak_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    silence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Optional[str] = None
    examples_path: Optional[str] = "data/addressee_examples.jsonl"
    decision_log_path: Optional[str] = None
    shadow: bool = Field(default=False, description="Decide but never speak; collect labels first.")
    other_ttl_sentences: int = Field(default=50, ge=1)
    background_ttl_sentences: int = Field(default=10, ge=1)
    max_examples: int = Field(default=1000, ge=1)
    k_neighbors: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _thresholds_make_sense(self) -> "AttentionConfig":
        speak = self.speak_threshold if self.speak_threshold is not None else 0.70
        silence = self.silence_threshold if self.silence_threshold is not None else 0.35
        if silence >= speak:
            raise ValueError("silence_threshold must be lower than speak_threshold")
        return self


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
    topic_tracking: TopicTrackingConfig = Field(default_factory=TopicTrackingConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
