from typing import TYPE_CHECKING

from ..core.utils import get_logger

if TYPE_CHECKING:
    from .conversation import Conversation
    from .model import LLM


class DropOldestStrategy:
    """Trim oldest conversation turns by turn count AND token budget.

    Dual gate: drops oldest messages when either the visible turn count
    exceeds max_turns OR the total token count exceeds the LLM's context
    window minus a response reserve. Keeps at least 1 message.
    """

    def __init__(self, max_turns: int = 20):
        self.logger = get_logger(__name__)
        self.max_turns = max_turns

    def trim(self, conversation: "Conversation", llm: "LLM") -> None:
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
        for i, t in enumerate(msg_tokens):
            idx = cutoff + i
            if idx >= len(conversation._messages) - 1:
                break  # keep at least 1 message
            dropped = new_cutoff - cutoff
            if total <= token_limit and dropped >= turn_excess:
                break
            total -= t
            new_cutoff = idx + 1

        excess = new_cutoff - cutoff
        if excess <= 0:
            return
        total_cut = conversation.trim_oldest(excess, self.max_turns, llm)
        self.logger.info("Trimmed %d messages (~%d tokens)", excess, total_cut)
