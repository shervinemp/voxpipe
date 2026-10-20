"""Data types for the attention layer.

These types are deliberately small and domain-neutral.  A buffer is a named
run of utterances; what the name means ("main", "table_4", "shift_notes") is
the caller's business.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional
from collections import deque

_MAX_BUFFER_UTTERANCES = 50


@dataclass(frozen=True)
class Utterance:
    """One thing a participant said; ``speaker`` is free-form ("user", "assistant", ...)."""

    speaker: str
    text: str


@dataclass
class FloorState:
    """Cheap conversational state used by the addressee model."""

    last_speaker: str = "assistant"
    awaiting_reply: bool = False
    other_present: bool = False
    order_phase: Optional[str] = None


@dataclass
class TopicBuffer:
    """A run of utterances, with its lifecycle state.

    ``ttl_sentences`` is optional.  Inferred side buffers use the store's
    default TTL; named buffers default to no expiry and are owned by the
    caller.
    """

    id: int
    utterances: Deque[Utterance] = field(default_factory=lambda: deque(maxlen=_MAX_BUFFER_UTTERANCES))
    last_seen: int = 0
    streak: int = 0
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl_sentences: Optional[int] = None

    def add(self, speaker: str, text: str) -> None:
        self.utterances.append(Utterance(speaker, text))

    def add_user(self, text: str) -> None:
        self.add("user", text)

    def add_assistant(self, text: str) -> None:
        self.add("assistant", text)

    def observe(self, text: str, index: int) -> None:
        """Add a user utterance and update the consecutive-hit streak."""
        consecutive = self.last_seen == index - 1
        self.add_user(text)
        self.last_seen = index
        self.streak = self.streak + 1 if consecutive else 1

    def mark_seen(self, index: int) -> None:
        self.last_seen = index
        self.streak = 0

    def is_expired(self, index: int, default_ttl: Optional[int]) -> bool:
        ttl = self.ttl_sentences if self.ttl_sentences is not None else default_ttl
        return ttl is not None and index - self.last_seen > ttl

    def recent(self, count: int) -> List[Utterance]:
        """The last ``count`` utterances, oldest first."""
        if count <= 0:
            return []
        utterances = list(self.utterances)
        return utterances[-count:]

    @property
    def last(self) -> Optional[Utterance]:
        return self.utterances[-1] if self.utterances else None

    def transcript(self, count: Optional[int] = None, *, labeled: bool = False) -> str:
        """The buffer as text, optionally with speaker labels."""
        utterances = list(self.utterances)
        if count is not None:
            utterances = utterances[-count:]
        if labeled:
            return "\n".join(f"{u.speaker.capitalize()}: {u.text}" for u in utterances)
        return "\n".join(u.text for u in utterances)


class RouteKind(Enum):
    """Why an utterance was placed where it was."""

    START = "start"
    MAIN = "main"
    SIDE = "side"
    NEW_SIDE = "new_side"
    PROMOTED = "promoted"
    FOLLOW_UP = "follow_up"
    EXPLICIT_BUFFER = "explicit_buffer"
    RESET = "reset"
    EMPTY = "empty"


_REASONS = {
    RouteKind.START: "start",
    RouteKind.MAIN: "main",
    RouteKind.SIDE: "side",
    RouteKind.NEW_SIDE: "new-side-topic",
    RouteKind.PROMOTED: "promoted",
    RouteKind.FOLLOW_UP: "follow-up",
    RouteKind.EXPLICIT_BUFFER: "explicit-buffer",
    RouteKind.RESET: "reset",
    RouteKind.EMPTY: "empty",
}


@dataclass(frozen=True)
class TopicDecision:
    """Where an utterance was routed, and whether the pipeline should answer it."""

    text: str
    answer: bool
    buffer_id: int
    kind: RouteKind
    score: float
    context: Optional[str] = None

    @property
    def reason(self) -> str:
        """Human-readable kind, kept for logging and backwards compatibility."""
        return _REASONS[self.kind]

    @property
    def related(self) -> bool:
        if self.kind in (RouteKind.NEW_SIDE, RouteKind.RESET):
            return False
        if self.kind is RouteKind.EXPLICIT_BUFFER:
            return self.answer
        return True

    @property
    def new_topic(self) -> bool:
        return self.kind is RouteKind.NEW_SIDE

    @property
    def promoted(self) -> bool:
        return self.kind is RouteKind.PROMOTED


@dataclass(frozen=True)
class GateDecision:
    """The generic result of a conversation gate.

    ``detail`` lets a specific gate attach its own typed decision.  A
    ``TopicTracker`` attaches a :class:`TopicDecision`; other gates may leave
    it None or attach something else.
    """

    answer: bool
    context: Optional[str] = None
    confidence: float = 1.0
    reason: str = ""
    detail: Any = None
    defer_ms: int = 0

    @property
    def route(self) -> Optional[TopicDecision]:
        """The attached topic decision, when the gate is a TopicTracker."""
        return self.detail if isinstance(self.detail, TopicDecision) else None
