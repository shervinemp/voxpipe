"""Context management ABC, eviction indicators, and MemoryHandler implementation."""

from abc import ABC
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from voxpipe.storage.base import MemoryStore
from ..core.utils import get_logger

if TYPE_CHECKING:
    from .conversation import Conversation


@runtime_checkable
class EvictionStrategy(Protocol):
    """Protocol for eviction strategies. Implementations should return the number of items to evict from the conversation context."""

    def __call__(self, conversation: "Conversation") -> int:
        """Returns the number of items to evict from the conversation context. Returns 0 if no eviction is needed."""
        ...


class MaxTurnsStrategy:
    """Strategy that returns the number of items to evict when visible turns exceed max_turns."""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns

    def __call__(self, conversation: "Conversation") -> int:
        if conversation.visible_count() > self.max_turns:
            return conversation.visible_count() - self.max_turns
        return 0


class ContextHandler(ABC):
    """Abstract Base Class for conversation context handlers."""

    def pre_process(self, conversation: "Conversation") -> bool:
        """Runs BEFORE LLM generation. Returns True if context was modified."""
        return False

    def post_process(self, conversation: "Conversation") -> bool:
        """Runs AFTER LLM generation. Returns True if context was modified."""
        return False


class MemoryHandler(ContextHandler):
    """Handler managing soft cutoff shifts, k-batch hard trims/compactions, and memory archival."""

    def __init__(
        self,
        strategy: EvictionStrategy,
        memory_store: MemoryStore | None = None,
        amortize_k: int = 4,
    ):
        self.logger = get_logger(__name__)
        self.strategy = strategy
        self.memory_store = memory_store
        self.amortize_k = amortize_k
        self._sig_counter = 0

    def pre_process(self, conversation: "Conversation") -> bool:
        """PRE-LLM: Track strategy blips, perform soft cutoff shifts, and do hard trim + batch archival every amortize_k times."""
        modified = False

        if excess := self.strategy(conversation):
            modified = True
            conversation._cutoff_idx += excess
            self._sig_counter += 1

            if self._sig_counter >= self.amortize_k:
                evicted, conversation._messages = (
                    conversation._messages[: (_idx := conversation._cutoff_idx)],
                    conversation._messages[_idx:],
                )

                if self.memory_store and len(evicted):
                    for msg in evicted:
                        content = getattr(msg, "content", str(msg))
                        role = getattr(msg, "role", "user")
                        self.memory_store.store(content, role=str(role))
                    self.logger.info(
                        "Batch archived %d evicted turns to MemoryStore", len(evicted)
                    )

                conversation.compact()
                self._sig_counter = 0

        if self.memory_store and len(conversation._messages) > 0:
            last_msg = conversation._messages[-1]
            if getattr(last_msg, "role", "").lower() in ("user"):
                query_str = getattr(last_msg, "content", "")
                if len(query_str) > 3:
                    records = self.memory_store.retrieve(query_str)
                    if records:
                        snippets = [r.text for r in records if r.text != query_str]
                        if snippets:
                            self.logger.info(
                                "Retrieved %d memory records for context", len(snippets)
                            )
                            modified = True

        return modified
