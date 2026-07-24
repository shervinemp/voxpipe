"""Context management base class, eviction indicators, and MemoryHandler implementation."""

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

    def trim(self, conversation: "Conversation", *args, **kwargs) -> List[Any]:
        """Backwards-compatible helper delegate."""
        ...


class MaxTurnsStrategy:
    """Strategy that blips True when visible turns exceed max_turns."""

    def __init__(self, max_turns: int = 20, amortize_k: int = 4):
        self.max_turns = max_turns
        self.amortize_k = amortize_k

    def __call__(self, conversation: "Conversation") -> bool:
        return conversation.visible_count() > self.max_turns

    def trim(self, conversation: "Conversation", llm: Optional[Any] = None, *args, **kwargs) -> List[Any]:
        """Legacy helper returning evicted messages if threshold or token limit is crossed."""
        old_cutoff = conversation._cutoff_idx
        over_token = False
        if llm:
            ctx = getattr(llm, "n_ctx", 4096)
            reserve = getattr(llm, "max_tokens", 512)
            token_limit = ctx - reserve
            total_tokens = sum(
                llm.count_tokens(conversation.get_message_content(i)) + 4
                for i in range(conversation._cutoff_idx, len(conversation._messages))
            ) if hasattr(llm, "count_tokens") else 0
            if total_tokens > token_limit:
                over_token = True

        visible = conversation.visible_count()
        if (self(conversation) or over_token) and visible > 1:
            target = max(1, self.max_turns // 2)
            excess = min(max(1, visible - target), visible - 1)
            conversation.trim_oldest(excess=excess, max_turns=self.max_turns, llm=llm)
            new_cutoff = conversation._cutoff_idx
            return [conversation._messages[i] for i in range(old_cutoff, new_cutoff)]
        return []


# Backward compatibility aliases
DropOldestStrategy = MaxTurnsStrategy
EvictionPolicy = EvictionStrategy


# 2. CONTEXT HANDLER ABC
class ContextHandler(ABC):
    """Abstract Base Class for conversation context handlers."""

    def pre_process(self, conversation: "Conversation") -> bool:
        """Runs BEFORE LLM generation. Returns True if context was modified."""
        return False

    def post_process(self, conversation: "Conversation") -> bool:
        """Runs AFTER LLM generation. Returns True if context was modified."""
        return False

    def handle(self, conversation: "Conversation", *args, **kwargs) -> bool:
        """Backwards compatible delegate forwarding to pre_process(conversation)."""
        return self.pre_process(conversation)

    def trim(self, conversation: "Conversation", *args, **kwargs) -> bool:
        """Backwards compatible delegate forwarding to pre_process(conversation)."""
        return self.pre_process(conversation)


# 3. MEMORY HANDLER (Owns soft shifts, k-batch hard trims/compactions, and archival)
class MemoryHandler(ContextHandler):
    """Handler managing soft cutoff shifts, k-batch hard trims/compactions, and memory archival."""

    def __init__(
        self,
        strategy: Optional[EvictionStrategy] = None,
        memory_store: Optional[Any] = None,
        amortize_k: int = 4,
        amortize_step: Optional[int] = None,
        max_turns: int = 20,
        eviction_policy: Optional[Any] = None,
        memory: Optional[Any] = None,
        auto_archive: bool = True,
    ):
        self.logger = get_logger(__name__)
        k_val = amortize_step if amortize_step is not None else amortize_k
        self.strategy = strategy or eviction_policy or MaxTurnsStrategy(max_turns=max_turns, amortize_k=k_val)
        self.memory_store = memory_store or memory
        self.amortize_k = getattr(self.strategy, "amortize_k", k_val)
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


# Backward compatibility aliases
ContextStrategy = ContextHandler
EvictionPolicy = EvictionStrategy
