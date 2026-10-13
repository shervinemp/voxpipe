"""Context window management: eviction strategies and the memory-backed context handler."""

import json
from abc import ABC
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Protocol, runtime_checkable

from ..core.utils import get_logger
from ..storage.memory import MemoryStore
from .conversation import Message

if TYPE_CHECKING:
    from .conversation import Conversation

_ARCHIVED_ROLES = (Message.Role.user, Message.Role.assistant)
_MIN_RECALL_QUERY_CHARS = 4
_MAX_SNIPPET_CHARS = 200


@runtime_checkable
class EvictionStrategy(Protocol):
    """Decides how many of the oldest visible messages no longer fit the context."""

    def __call__(self, conversation: "Conversation") -> int:
        """Return the number of messages to evict; 0 when none need to go."""
        ...


class MaxTurnsStrategy:
    """Evicts whatever exceeds ``max_turns`` visible messages."""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns

    def __call__(self, conversation: "Conversation") -> int:
        return max(0, conversation.visible_count() - self.max_turns)


class TokenBudgetStrategy:
    """Evicts oldest messages until the prompt fits a token budget.

    The budget covers everything sent to the model: system prompt, tool
    schemas and visible messages.  The newest message is never evicted.
    """

    def __init__(self, count_tokens: Callable[[str], int], budget: int, per_message_overhead: int = 4):
        self.budget = budget
        self.per_message_overhead = per_message_overhead
        self._count = lru_cache(maxsize=4096)(count_tokens)

    def __call__(self, conversation: "Conversation") -> int:
        costs = [self._count(m.rendered) + self.per_message_overhead for m in conversation.visible]
        total = sum(costs) + self._count(conversation.system_message)
        if conversation.tools:
            total += self._count(json.dumps([t.to_dict() for t in conversation.tools.values()]))

        evict = 0
        for cost in costs[:-1]:
            if total <= self.budget:
                break
            total -= cost
            evict += 1
        return evict


class CompositeStrategy:
    """Evicts as many messages as the most demanding of several strategies asks for."""

    def __init__(self, *strategies: EvictionStrategy):
        self.strategies = strategies

    def __call__(self, conversation: "Conversation") -> int:
        return max((strategy(conversation) for strategy in self.strategies), default=0)


class ContextHandler(ABC):
    """Hooks a Session calls around each turn to manage the conversation's context."""

    def pre_process(self, conversation: "Conversation") -> bool:
        """Runs before generation. Returns True if the context was modified."""
        return False

    def post_process(self, conversation: "Conversation") -> bool:
        """Runs after generation. Returns True if the context was modified."""
        return False

    def finalize(self, conversation: "Conversation") -> None:
        """Runs when the conversation is about to be discarded (session reset or close)."""


class MemoryHandler(ContextHandler):
    """Keeps the context window within bounds and bridges it to long-term memory.

    Each turn it hides the messages the strategy no longer wants and, every
    ``amortize_k`` such trims, drops the hidden ones for good.  With a memory
    store it archives what it drops (``auto_archive``) and prefixes the newest
    user message with up to ``recall_top_k`` related memories.
    """

    def __init__(
        self,
        strategy: EvictionStrategy,
        memory_store: Optional[MemoryStore] = None,
        amortize_k: int = 4,
        auto_archive: bool = True,
        recall_top_k: int = 2,
    ):
        self.logger = get_logger(__name__)
        self.strategy = strategy
        self.memory_store = memory_store
        self.amortize_k = amortize_k
        self.auto_archive = auto_archive
        self.recall_top_k = recall_top_k
        self._trims_since_drop = 0
        self._recalled_for: Optional[Message] = None

    def pre_process(self, conversation: "Conversation") -> bool:
        trimmed = self._trim(conversation)
        recalled = self._recall(conversation)
        return trimmed or recalled

    def finalize(self, conversation: "Conversation") -> None:
        """Archive everything still held, so a reset or shutdown loses no history."""
        self._archive(conversation.all_messages)
        self._trims_since_drop = 0
        self._recalled_for = None

    def _trim(self, conversation: "Conversation") -> bool:
        if not conversation.hide_oldest(self.strategy(conversation)):
            return False
        self._trims_since_drop += 1
        if self._trims_since_drop >= self.amortize_k:
            self._archive(conversation.drop_hidden())
            self._trims_since_drop = 0
        return True

    def _archive(self, messages: Iterable[Message]) -> None:
        if not (self.memory_store and self.auto_archive):
            return
        pending = [m for m in messages if m.role in _ARCHIVED_ROLES and m.content.strip() and not m.archived]
        try:
            for message in pending:
                self.memory_store.store(message.content, role=message.role.value)
                message.archived = True
        except Exception:  # noqa: BLE001 - a failing memory backend must not break the conversation
            self.logger.warning("Could not archive conversation turns to memory.", exc_info=True)
            return
        if pending:
            self.logger.info("Archived %d conversation turns to memory.", len(pending))

    def _recall(self, conversation: "Conversation") -> bool:
        """Attach related memories to the newest user message, once per message."""
        message = conversation.last_message
        if (
            not self.memory_store
            or self.recall_top_k <= 0
            or message is None
            or message.role is not Message.Role.user
            or message is self._recalled_for
            or len(message.content.strip()) < _MIN_RECALL_QUERY_CHARS
        ):
            return False
        self._recalled_for = message

        try:
            records = self.memory_store.retrieve(message.content, top_k=self.recall_top_k)
        except Exception:  # noqa: BLE001 - see _archive
            self.logger.warning("Could not recall memories.", exc_info=True)
            return False

        in_context = {m.content for m in conversation.visible}
        snippets = [r.text for r in records if r.text not in in_context]
        if not snippets:
            return False

        recalled = "\n".join(f"(Earlier: {snippet[:_MAX_SNIPPET_CHARS]})" for snippet in snippets)
        message.context = f"{recalled}\n{message.context}" if message.context else recalled
        self.logger.info("Recalled %d memories for context.", len(snippets))
        return True
