"""voxpipe: modular voice pipeline (ASR, LLM tool calling, streaming TTS, conversation memory).

The package root exposes the LLM and storage building blocks, which need no
audio hardware.  The audio pipeline lives in :mod:`voxpipe.pipeline` and is
imported on demand.
"""

__version__ = "0.2.0"

from voxpipe.core import VoxpipeError
from voxpipe.llm import (
    ContextHandler,
    Conversation,
    EvictionStrategy,
    MaxTurnsStrategy,
    MemoryHandler,
    Session,
    SessionManager,
    SessionState,
    Tool,
)
from voxpipe.storage import MemoryStore, Query, QueryInput, RAMStorage, Record, Retriever, SQLiteStorage, Storage, Storer

__all__ = [
    "__version__",
    "VoxpipeError",
    "Session",
    "SessionManager",
    "SessionState",
    "Conversation",
    "ContextHandler",
    "MemoryHandler",
    "EvictionStrategy",
    "MaxTurnsStrategy",
    "Tool",
    "Record",
    "Query",
    "QueryInput",
    "Retriever",
    "Storer",
    "Storage",
    "RAMStorage",
    "SQLiteStorage",
    "MemoryStore",
]
