"""Context window management: eviction strategies and the memory-backed context handler."""

import json
import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Protocol, runtime_checkable

from ..core.utils import get_logger
from ..storage.protocols import Storage
from .conversation import Message

if TYPE_CHECKING:
    from .conversation import Conversation

_ARCHIVED_ROLES = (Message.Role.user, Message.Role.assistant)
_MIN_RECALL_QUERY_CHARS = 4
_MAX_SNIPPET_CHARS = 200
_MAX_ARCHIVE_BACKLOG = 200  # messages kept for another try while the memory backend is failing
# How far below its limit a strategy cuts by default.  Cutting to the limit itself would move the
# start of the prompt on every turn, and a model that caches its prompt prefix would re-read it all.
_TRIM_FRACTION = 0.6


@runtime_checkable
class EvictionStrategy(Protocol):
    """Decides how many of the oldest messages no longer fit the context."""

    def __call__(self, conversation: "Conversation") -> int:
        """Return the number of messages to evict; 0 when none need to go."""
        ...


class MaxTurnsStrategy:
    """Keeps at most ``max_turns`` turns, cutting back to ``trim_to`` once that is exceeded.

    A turn is a user message and everything said in answer to it, tool calls
    included.  ``trim_to`` defaults to 60% of ``max_turns``, which leaves the
    start of the prompt alone for several turns between cuts.
    """

    def __init__(self, max_turns: int = 10, *, trim_to: Optional[int] = None):
        if trim_to is None:
            trim_to = max(1, round(max_turns * _TRIM_FRACTION))
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (max_turns, trim_to)):
            raise TypeError("max_turns and trim_to must be whole numbers.")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        if not 1 <= trim_to <= max_turns:
            raise ValueError("trim_to must be between 1 and max_turns.")
        self.max_turns = max_turns
        self.trim_to = trim_to

    def __call__(self, conversation: "Conversation") -> int:
        turn_starts = [i for i, m in enumerate(conversation.history) if m.role is Message.Role.user]
        if len(turn_starts) <= self.max_turns:
            return 0
        return turn_starts[-self.trim_to]


class TokenBudgetStrategy:
    """Keeps the prompt within a token budget, cutting back to ``trim_to`` once that is exceeded.

    Both figures cover everything sent to the model: system prompt, tool
    schemas and messages.  ``trim_to`` defaults to the fixed part of the prompt
    plus 60% of the room the budget leaves for messages.  The newest message is
    never evicted.
    """

    def __init__(
        self,
        count_tokens: Callable[[str], int],
        budget: int,
        per_message_overhead: int = 4,
        *,
        trim_to: Optional[int] = None,
    ):
        if budget < 1:
            raise ValueError("budget must be positive.")
        if trim_to is not None and not 0 <= trim_to <= budget:
            raise ValueError("trim_to must be between 0 and the budget.")
        self.budget = budget
        self.trim_to = trim_to
        self.per_message_overhead = per_message_overhead
        self._count = lru_cache(maxsize=4096)(count_tokens)

    def __call__(self, conversation: "Conversation") -> int:
        costs = [self._count(_sent_text(m)) + self.per_message_overhead for m in conversation.history]
        fixed = self._count(conversation.system_prompt)
        if conversation.tools:
            fixed += self._count(json.dumps([t.to_dict() for t in conversation.tools.values()]))
        total = fixed + sum(costs)
        if total <= self.budget:
            return 0

        target = self.trim_to
        if target is None:
            target = fixed + int(max(0, self.budget - fixed) * _TRIM_FRACTION)
        evict = 0
        for cost in costs[:-1]:
            if total <= target:
                break
            total -= cost
            evict += 1
        return evict


def _sent_text(message: Message) -> str:
    """Everything of a message that reaches the model: its text, and the tool calls or tool name it carries."""
    parts = [message.rendered]
    if message.name:
        parts.append(message.name)
    parts.extend(f"{call.name}{json.dumps(call.arguments, default=str)}" for call in message.tool_calls)
    return "".join(parts)


class CompositeStrategy:
    """Evicts as many messages as the most demanding of several strategies asks for."""

    def __init__(self, *strategies: EvictionStrategy):
        self.strategies = strategies

    def __call__(self, conversation: "Conversation") -> int:
        return max((strategy(conversation) for strategy in self.strategies), default=0)


class ContextHandler:
    """Hooks a Session calls around each turn to manage the conversation's context.

    Every hook does nothing by default; override the ones you need.
    """

    def pre_process(self, conversation: "Conversation") -> bool:
        """Runs before generation. Returns True if the context was modified."""
        return False

    def post_process(self, conversation: "Conversation") -> bool:
        """Runs after generation. Returns True if the context was modified."""
        return False

    def finalize(self, conversation: "Conversation") -> None:
        """Runs when the conversation is about to be discarded (session reset or close)."""


class MemoryHandler(ContextHandler):
    """Keeps the context within bounds and bridges it to long-term memory.

    Each turn it evicts the messages the strategy no longer wants.  Given a
    ``memory`` (anything with ``store`` and ``retrieve``) it archives what it
    evicts (``auto_archive``) and prefixes the newest user message with up to
    ``recall_top_k`` related memories.  Messages the memory fails to take are
    kept and offered again the next time anything is archived.
    """

    def __init__(
        self,
        strategy: EvictionStrategy,
        memory: Optional[Storage] = None,
        *,
        auto_archive: bool = True,
        recall_top_k: int = 2,
    ):
        self.logger = get_logger(__name__)
        self.strategy = strategy
        self.memory = memory
        self.auto_archive = auto_archive
        self.recall_top_k = recall_top_k
        self._recalled_for: Optional[Message] = None
        self._backlog: List[Message] = []  # evicted, but the memory would not take them
        self._archive_lock = threading.Lock()  # finalize() can arrive from reset() while a turn winds down

    def pre_process(self, conversation: "Conversation") -> bool:
        trimmed = self._trim(conversation)
        recalled = self._recall(conversation)
        return trimmed or recalled

    def finalize(self, conversation: "Conversation") -> None:
        """Archive everything still held, so a reset or shutdown loses no history."""
        self._archive(conversation.history)
        self._recalled_for = None

    def _trim(self, conversation: "Conversation") -> bool:
        evicted = conversation.evict_oldest(self.strategy(conversation))
        self._archive(evicted)
        return bool(evicted)

    def _archive(self, messages: Iterable[Message]) -> None:
        if self.memory is None or not self.auto_archive:
            return
        with self._archive_lock:
            queued = {id(m) for m in self._backlog}
            pending = self._backlog + [
                m
                for m in messages
                if m.role in _ARCHIVED_ROLES and m.content.strip() and not m.archived and id(m) not in queued
            ]
            stored = 0
            try:
                for message in pending:
                    self.memory.store(message.content, role=message.role.value)
                    message.archived = True
                    stored += 1
            except Exception:  # noqa: BLE001 - a failing memory backend must not break the conversation
                self.logger.warning(
                    "Could not archive %d message(s) to memory; they will be offered again.",
                    len(pending) - stored,
                    exc_info=True,
                )
            self._backlog = pending[stored:][-_MAX_ARCHIVE_BACKLOG:]
            if stored:
                self.logger.info("Archived %d messages to memory.", stored)

    def _recall(self, conversation: "Conversation") -> bool:
        """Attach related memories to the newest user message, once per message."""
        message = conversation.last_message
        if (
            self.memory is None
            or self.recall_top_k <= 0
            or message is None
            or message.role is not Message.Role.user
            or message is self._recalled_for
            or len(message.content.strip()) < _MIN_RECALL_QUERY_CHARS
        ):
            return False
        self._recalled_for = message

        try:
            records = self.memory.retrieve(message.content, top_k=self.recall_top_k)
        except Exception:  # noqa: BLE001 - see _archive
            self.logger.warning("Could not recall memories.", exc_info=True)
            return False

        in_context = {m.content for m in conversation.history}
        snippets = [r.text for r in records if r.text not in in_context]
        if not snippets:
            return False

        recalled = "\n".join(f"(Earlier: {snippet[:_MAX_SNIPPET_CHARS]})" for snippet in snippets)
        message.context = f"{recalled}\n{message.context}" if message.context else recalled
        self.logger.info("Recalled %d memories for context.", len(snippets))
        return True
