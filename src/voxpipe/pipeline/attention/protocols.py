"""Protocols that keep the attention layer pluggable."""

from typing import Protocol, Sequence

from .models import GateDecision, TopicBuffer, Utterance


class RelatednessScorer(Protocol):
    """Scores how related ``text`` is to each candidate buffer, from 0 to 1.

    ``default_threshold`` is the threshold that makes sense on this scorer's
    own scale.  The factory uses it instead of maintaining a separate table.
    """

    default_threshold: float

    def scores(self, text: str, candidates: Sequence[Sequence[Utterance]]) -> Sequence[float]: ...


class FollowUpPolicy(Protocol):
    """Decides whether an utterance is a short reply to the assistant's question."""

    def is_followup(self, text: str, main: TopicBuffer) -> bool: ...


class ConversationGate(Protocol):
    """Decides whether an utterance should reach the assistant.

    This is intentionally small and domain-neutral.  :class:`TopicTracker` is
    one implementation; a wake-word gate, a workflow-mode gate, or a composite
    gate can implement the same contract without the pipeline knowing which one
    it has.
    """

    def evaluate(self, text: str, *, explicit: bool = False) -> GateDecision:
        """Return whether ``text`` should be answered, and any context to inject.

        ``explicit`` is True for input that is already known to be addressed to
        the assistant (push-to-talk, typed text).  A gate may still record it
        for its own state, but it should not reject it.
        """
        ...

    def record_user(self, text: str) -> None:
        """Record an explicitly addressed utterance without gating it."""
        ...

    def record_reply(self, text: str) -> None:
        """Record the assistant's spoken reply."""
        ...

    def reset(self) -> None:
        """Forget the conversation state."""
        ...

    def close(self) -> None:
        """Release any model or resources."""
        ...
