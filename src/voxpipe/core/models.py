from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class ConversationHistoryConfig(BaseModel):
    enabled: bool = False
    db_path: str = "data/conversations.db"
    max_entries: int = 1000
    top_k: int = 2
    ttl_days: int = 30


class LocalModelEntry(BaseModel):
    model_path: Optional[str] = None
    n_ctx: int = 512
    max_tokens: int = 128
    decoder: str = "general"
    type_k: Optional[str] = None
    type_v: Optional[str] = None


class LLMConfig(BaseModel):
    backend: str = "local"
    model: str
    max_tool_iterations: int = Field(default=1, ge=0, description="0 = tools disabled, 1 = single call, N = N+1 max passes")
    local: dict[str, LocalModelEntry] = Field(default_factory=dict)
    litellm: dict = Field(default_factory=dict)


class TTSConfig(BaseModel):
    provider: str
    weights_dir: str


class HotkeyConfig(BaseModel):
    enable: bool = True
    push_to_talk: Optional[str] = None
    press_to_reset: Optional[str] = None


class ASRConfig(BaseModel):
    provider: str
    weights_dir: str
    vad_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    trailing_silence_ms: int = Field(default=800, ge=100, le=10_000)
    leading_silence_ms: int = Field(default=1000, ge=0, le=10_000)
    max_segment_duration: float = Field(default=0.0, ge=0.0, le=300.0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    def model_post_init(self, __context):
        extra_keys = set(self.__pydantic_extra__ or {})
        if extra_keys:
            import logging
            logging.getLogger(__name__).warning(
                f"Unknown config keys ignored: {extra_keys}. "
                "Check for typos in your config.yaml."
            )

    llm: LLMConfig
    tts: TTSConfig
    asr: ASRConfig
    hotkeys: HotkeyConfig = Field(default_factory=HotkeyConfig)
    conversation_history: ConversationHistoryConfig = Field(default_factory=ConversationHistoryConfig)
