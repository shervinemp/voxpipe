"""ContextHandler and eviction policies for managing conversation context windows and memory retrieval."""

from typing import TYPE_CHECKING, Any, List, Optional
from ..core.utils import get_logger

if TYPE_CHECKING:
    from .conversation import Conversation
    from .model import LLM
    from ..storage.base import MemoryStore


class DropOldestStrategy:
    """Trim oldest conversation turns by turn count AND token budget.

    Dual gate: drops oldest messages when either the visible turn count
    exceeds max_turns OR the total token count exceeds the LLM's context
    window minus a response reserve. Keeps at least 1 message.
    """

    def __init__(self, max_turns: int = 20):
        self.logger = get_logger(__name__)
        self.max_turns = max_turns

    def trim(self, conversation: "Conversation", llm: "LLM") -> List[Any]:
        cutoff = conversation._cutoff_idx
        ctx = getattr(llm, "n_ctx", 4096)
        reserve = getattr(llm, "max_tokens", 512)
        token_limit = ctx - reserve

        # Compute token counts for all visible messages
        total = 0
        msg_tokens = []
        for i in range(cutoff, len(conversation._messages)):
            if conversation.get_token_count(i) == 0:
                raw = conversation.get_message_content(i)
                conversation.set_token_count(i, llm.count_tokens(raw) + 4)
            t = conversation.get_token_count(i)
            msg_tokens.append(t)
            total += t

        visible = len(msg_tokens)
        target = self.max_turns // 2
        turn_excess = max(0, visible - target)

        # Drop oldest messages until under token budget AND turn limit
        new_cutoff = cutoff
        evicted_msgs = []
        for i, t in enumerate(msg_tokens):
            idx = cutoff + i
            if idx >= len(conversation._messages) - 1:
                break  # keep at least 1 message
            dropped = new_cutoff - cutoff
            if total <= token_limit and dropped >= turn_excess:
                break
            total -= t
            evicted_msgs.append(conversation._messages[idx])
            new_cutoff = idx + 1

        excess = new_cutoff - cutoff
        if excess <= 0:
            return []

        total_cut = conversation.trim_oldest(excess, self.max_turns, llm)
        self.logger.info("Trimmed %d messages (~%d tokens)", excess, total_cut)
        return evicted_msgs


class ContextHandler:
    """Handles conversation context window trimming, automatic memory archival, and RAG memory injection."""

    def __init__(
        self,
        max_turns: int = 20,
        eviction_policy: Optional[Any] = None,
        memory: Optional["MemoryStore"] = None,
        auto_archive: bool = True,
    ):
        self.logger = get_logger(__name__)
        self.max_turns = max_turns
        self.eviction_policy = eviction_policy or DropOldestStrategy(max_turns=max_turns)
        self.memory = memory
        self.auto_archive = auto_archive

    def handle(self, conversation: "Conversation", llm: "LLM", session_id: str = "default") -> None:
        """Process active conversation: trim window, archive evicted turns, and inject memory."""
        # 1. Execute eviction trimming
        evicted = self.eviction_policy.trim(conversation, llm)

        # 2. Archive evicted messages if memory is present
        if self.auto_archive and self.memory and evicted:
            bank_name = f"session:{session_id}"
            for msg in evicted:
                content = getattr(msg, "content", str(msg))
                role = getattr(msg, "role", "user")
                self.memory.store(content, bank=bank_name, role=str(role))

        # 3. Retrieve relevant records for the last user message and inject
        if self.memory and len(conversation._messages) > 0:
            last_msg = conversation._messages[-1]
            if getattr(last_msg, "role", None) in ("user", "User"):
                query_str = getattr(last_msg, "content", "")
                if query_str and len(query_str) > 3:
                    bank_name = f"session:{session_id}"
                    records = self.memory.retrieve(query_str, top_k=2, fallback="global")
                    if records:
                        snippets = [r.text for r in records if r.text != query_str]
                        if snippets:
                            self.logger.info("Injected %d memory records into context", len(snippets))

    def trim(self, conversation: "Conversation", llm: "LLM") -> None:
        """Backwards compatible delegate method for legacy callers."""
        self.handle(conversation, llm)
