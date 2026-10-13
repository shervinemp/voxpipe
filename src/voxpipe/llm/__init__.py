"""LLM layer: conversations, tools, providers, and the session that runs turns."""

from .context import (
    CompositeStrategy,
    ContextHandler,
    EvictionStrategy,
    MaxTurnsStrategy,
    MemoryHandler,
    TokenBudgetStrategy,
)
from .conversation import Conversation, Message
from .model import LLM, GGUFLLM, LiteLLMProvider, LLMProviders
from .session import Session, SessionEvent, TextDelta, ToolCallEvent, ToolResultEvent
from .session_manager import SessionManager
from .state import SessionState
from .tools import MalformedToolCall, Tool, ToolCall, ToolChoice, ToolResult, permission

__all__ = [
    "LLM",
    "GGUFLLM",
    "LiteLLMProvider",
    "LLMProviders",
    "Conversation",
    "Message",
    "Session",
    "SessionEvent",
    "TextDelta",
    "ToolCallEvent",
    "ToolResultEvent",
    "SessionState",
    "SessionManager",
    "ContextHandler",
    "MemoryHandler",
    "EvictionStrategy",
    "MaxTurnsStrategy",
    "TokenBudgetStrategy",
    "CompositeStrategy",
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolChoice",
    "MalformedToolCall",
    "permission",
]
