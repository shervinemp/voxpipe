"""Context management ABC, eviction indicators, and MemoryHandler implementation."""

from abc import ABC
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, runtime_checkable
from ..core.utils import get_logger

if TYPE_CHECKING:
    from .conversation import Conversation


# 1. PURE INDICATOR STRATEGY PROTOCOL
@runtime_checkable
class EvictionStrategy(Protocol):
    """Protocol for boolean indicator strategies."""

    def __call__(self, conversation: "Conversation") -> bool:
        """Returns True ('blips') when eviction threshold is crossed."""
        ...


class MaxTurnsStrategy:
    """Strategy that blips True when visible turns exceed max_turns."""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns

    def __call__(self, conversation: "Conversation") -> bool:
        return conversation.visible_count() > self.max_turns


# 2. CONTEXT HANDLER ABC
class ContextHandler(ABC):
    """Abstract Base Class for conversation context handlers."""

    def pre_process(self, conversation: "Conversation") -> bool:
        """Runs BEFORE LLM generation. Returns True if context was modified."""
        return False

    def post_process(self, conversation: "Conversation") -> bool:
        """Runs AFTER LLM generation. Returns True if context was modified."""
        return False


# 3. MEMORY HANDLER
class MemoryHandler(ContextHandler):
    """Handler managing soft cutoff shifts, k-batch hard trims/compactions, and memory archival."""

    def __init__(
        self,
        strategy: Optional[EvictionStrategy] = None,
        memory_store: Optional[Any] = None,
        amortize_k: int = 4,
        auto_archive: bool = True,
    ):
        self.logger = get_logger(__name__)
        self.strategy = strategy or MaxTurnsStrategy()
        self.memory_store = memory_store
        max_t = getattr(self.strategy, "max_turns", 20)
        self.amortize_k = min(amortize_k, max_t) if max_t < 4 else amortize_k
        self.auto_archive = auto_archive
        self._blip_counter = 0

    def pre_process(self, conversation: "Conversation") -> bool:
        """PRE-LLM: Track strategy blips, perform soft cutoff shifts, and do hard trim + batch archival every amortize_k times."""
        modified = False

        # 1. Check if strategy "blips" True
        if self.strategy(conversation):
            modified = True
            max_turns = getattr(self.strategy, "max_turns", 20)
            excess = max(1, conversation.visible_count() - max_turns)
            conversation._cutoff_idx += excess
            self._blip_counter += 1

            # 2. Hard Trim, Compact, & Batch Archival every amortize_k blips
            if self._blip_counter >= self.amortize_k:
                # Direct slice of all evicted messages sitting behind cutoff_idx!
                evicted = conversation._messages[:conversation._cutoff_idx]

                # Archive evicted slice to MemoryStore
                if self.auto_archive and self.memory_store and evicted:
                    for msg in evicted:
                        content = getattr(msg, "content", str(msg))
                        role = getattr(msg, "role", "user")
                        self.memory_store.store(content, role=str(role))
                    self.logger.info("Batch archived %d evicted turns to MemoryStore", len(evicted))

                # Hard Trim & Compact: Recreate internal message list and reset cutoff_idx to 0
                conversation.compact()
                self._blip_counter = 0

        # 3. Memory Injection (RAG Recall)
        if self.memory_store and len(conversation._messages) > 0:
            last_msg = conversation._messages[-1]
            if getattr(last_msg, "role", None) in ("user", "User"):
                query_str = getattr(last_msg, "content", "")
                if query_str and len(query_str) > 3:
                    records = self.memory_store.retrieve(query_str, top_k=2, fallback="global")
                    if records:
                        snippets = [r.text for r in records if r.text != query_str]
                        if snippets:
                            self.logger.info("Retrieved %d memory records for context", len(snippets))
                            modified = True

        return modified
