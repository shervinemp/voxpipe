"""The topic tracker: route utterances between an active conversation and side context.

Scoring happens outside the data lock so a slow backend (Laya) cannot block a
reset or shutdown.  A generation counter drops a score result that belongs to a
conversation that was reset while the model was thinking.
"""

import math
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.utils import get_logger
from .models import GateDecision, RouteKind, TopicBuffer, TopicDecision, Utterance
from .policies import DefaultFollowUpPolicy
from .protocols import FollowUpPolicy, RelatednessScorer
from .scorers import LexicalScorer
from .store import BufferStore


class TopicTracker:
    """Routes utterances between an active topic and a few side topics.

    A sentence joins the buffer it is most related to, provided the best score
    reaches the tracker's threshold; otherwise it starts a new inferred side
    buffer.  Inferred side buffers expire after ``ttl_sentences`` utterances
    without a hit and are capped at ``max_unrelated``.

    Named buffers (see :meth:`route_to`) are explicit workflow contexts.  They
    are never selected by similarity, are not evicted by inferred side chatter,
    and do not expire unless given their own TTL.

    ``promote_after`` controls how an inferred side topic becomes the active
    one.  Zero (the default) never promotes: only the original main topic is
    answered.  A value of N promotes a side buffer once N consecutive
    utterances have landed in it, and the prior side utterances are returned as
    ``context`` so the model can pick the thread up.
    """

    def __init__(
        self,
        scorer: RelatednessScorer,
        *,
        threshold: Optional[float] = None,
        ttl_sentences: int = 10,
        max_unrelated: int = 4,
        history_sentences: int = 8,
        promote_after: int = 0,
        fallback: Optional[RelatednessScorer] = None,
        fallback_threshold: Optional[float] = None,
        follow_up_policy: Optional[FollowUpPolicy] = None,
    ):
        self.logger = get_logger(__name__)
        self.scorer = scorer
        self.threshold = threshold if threshold is not None else getattr(scorer, "default_threshold", 0.4)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        if ttl_sentences < 1:
            raise ValueError("ttl_sentences must be at least 1.")
        if max_unrelated < 1:
            raise ValueError("max_unrelated must be at least 1.")
        if history_sentences < 1:
            raise ValueError("history_sentences must be at least 1.")
        if promote_after < 0:
            raise ValueError("promote_after must not be negative.")

        self._fallback = fallback or LexicalScorer()
        self.fallback_threshold = (
            fallback_threshold
            if fallback_threshold is not None
            else getattr(self._fallback, "default_threshold", self.threshold)
        )
        if not 0.0 <= self.fallback_threshold <= 1.0:
            raise ValueError("fallback_threshold must be between 0 and 1.")

        self.ttl_sentences = ttl_sentences
        self.max_unrelated = max_unrelated
        self.history_sentences = history_sentences
        self.promote_after = promote_after
        self._follow_up: FollowUpPolicy = follow_up_policy or DefaultFollowUpPolicy()
        self._store = BufferStore(max_side=max_unrelated, ttl_sentences=ttl_sentences)
        self._lock = threading.RLock()  # protects buffer data
        self._route_lock = threading.Lock()  # serialises routing without blocking reset/close
        self._generation = 0
        self._warned = False
        self._closed = False

    @property
    def main(self) -> TopicBuffer:
        """The buffer the assistant is currently answering."""
        return self._store.main

    @property
    def unrelated(self) -> List[TopicBuffer]:
        """Inferred side buffers, oldest first.  A copy."""
        return self._store.side

    @property
    def side(self) -> List[TopicBuffer]:
        """Alias for :attr:`unrelated`."""
        return self._store.side

    @property
    def named(self) -> List[TopicBuffer]:
        """Explicit workflow buffers."""
        return self._store.named

    def buffer(self, name: str) -> Optional[TopicBuffer]:
        """Return the main buffer or a named buffer, if present."""
        with self._lock:
            return self._store.find(name)

    def evaluate(self, text: str, *, explicit: bool = False) -> GateDecision:
        """The :class:`ConversationGate` entry point.

        Explicit input is recorded in the main buffer and always answered;
        everything else goes through :meth:`route`.
        """
        with self._lock:
            closed = self._closed
        if closed:
            decision = TopicDecision(
                text=(text or "").strip(),
                answer=False,
                buffer_id=-1,
                kind=RouteKind.RESET,
                score=0.0,
            )
            return GateDecision(answer=False, confidence=0.0, reason="closed", detail=decision)
        if explicit:
            self.record_user(text)
            return GateDecision(answer=True, confidence=1.0, reason="explicit")
        decision = self.route(text)
        return GateDecision(
            answer=decision.answer,
            context=decision.context,
            confidence=decision.score,
            reason=decision.reason,
            detail=decision,
        )

    def route(self, text: str) -> TopicDecision:
        """Place ``text`` in a buffer and say whether it should be answered."""
        cleaned = (text or "").strip()
        with self._route_lock:
            with self._lock:
                if self._closed:
                    return TopicDecision(
                        text=cleaned, answer=False, buffer_id=-1, kind=RouteKind.RESET, score=0.0
                    )
                if not cleaned:
                    return TopicDecision(
                        text="", answer=True, buffer_id=self._store.main.id, kind=RouteKind.EMPTY, score=1.0
                    )
                index = self._store.tick()
                generation = self._generation
                main = self._store.main
                if not main.utterances:
                    # The first utterance has nothing to be unrelated to; it opens the conversation.
                    main.add_user(cleaned)
                    return TopicDecision(
                        text=cleaned,
                        answer=True,
                        buffer_id=main.id,
                        kind=RouteKind.START,
                        score=1.0,
                    )
                if self._follow_up.is_followup(cleaned, main):
                    main.add_user(cleaned)
                    return TopicDecision(
                        text=cleaned,
                        answer=True,
                        buffer_id=main.id,
                        kind=RouteKind.FOLLOW_UP,
                        score=1.0,
                    )
                buffers = [main, *self._store.side]
                candidates = [buffer.recent(self.history_sentences) for buffer in buffers]

            scores, used_fallback = self._scores(cleaned, candidates)
            threshold = self.fallback_threshold if used_fallback else self.threshold
            best = max(range(len(scores)), key=scores.__getitem__)
            score = scores[best]

            with self._lock:
                if generation != self._generation:
                    # reset() or close() happened while the scorer was thinking.
                    return TopicDecision(
                        text=cleaned,
                        answer=False,
                        buffer_id=-1,
                        kind=RouteKind.RESET,
                        score=0.0,
                    )
                if score >= threshold:
                    buffer = buffers[best]
                    if buffer is self._store.main:
                        buffer.add_user(cleaned)
                        return TopicDecision(
                            text=cleaned,
                            answer=True,
                            buffer_id=buffer.id,
                            kind=RouteKind.MAIN,
                            score=score,
                        )
                    buffer.observe(cleaned, index)
                    if self.promote_after and buffer.streak >= self.promote_after:
                        return self._promote(buffer, cleaned, score)
                    return TopicDecision(
                        text=cleaned,
                        answer=False,
                        buffer_id=buffer.id,
                        kind=RouteKind.SIDE,
                        score=score,
                    )

                # Nothing matched: this is a new inferred thread of its own.
                buffer = self._store.add_side()
                buffer.observe(cleaned, index)
                self._store.trim()
                if self.promote_after and buffer.streak >= self.promote_after:
                    return self._promote(buffer, cleaned, score)
                return TopicDecision(
                    text=cleaned,
                    answer=False,
                    buffer_id=buffer.id,
                    kind=RouteKind.NEW_SIDE,
                    score=score,
                )

    def route_to(
        self,
        name: str,
        text: str,
        *,
        ttl_sentences: Optional[int] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> TopicDecision:
        """Put ``text`` in an explicitly named buffer, creating it if needed.

        This is the general escape hatch for workflow contexts: a restaurant
        can use ``"table_4"``, a support tool ``"ticket_123"``, a meeting
        ``"action_items"``.  Named buffers are not candidates for inferred
        routing, are not trimmed with the inferred side buffers, and do not
        expire unless ``ttl_sentences`` is given.
        """
        if not name:
            raise ValueError("Buffer name must not be empty; use 'main' for the main buffer.")
        cleaned = (text or "").strip()
        with self._route_lock:
            with self._lock:
                if self._closed:
                    return TopicDecision(
                        text=cleaned, answer=False, buffer_id=-1, kind=RouteKind.RESET, score=0.0
                    )
                if not cleaned:
                    return TopicDecision(
                        text="", answer=True, buffer_id=self._store.main.id, kind=RouteKind.EMPTY, score=1.0
                    )
                index = self._store.tick()
                buffer = self._store.add_named(name, ttl_sentences=ttl_sentences, metadata=metadata)
                buffer.add_user(cleaned)
                buffer.mark_seen(index)
                return TopicDecision(
                    text=cleaned,
                    answer=buffer is self._store.main,
                    buffer_id=buffer.id,
                    kind=RouteKind.EXPLICIT_BUFFER,
                    score=1.0,
                )

    def record_user(self, text: str) -> None:
        """Add an explicitly addressed utterance to the main topic, without gating it."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        with self._route_lock:
            with self._lock:
                if self._closed:
                    return
                self._store.tick()
                self._store.main.add_user(cleaned)

    def record_reply(self, text: str) -> None:
        """Add the assistant's spoken reply to the active buffer."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        with self._route_lock:
            with self._lock:
                if not self._closed:
                    self._store.main.add_assistant(cleaned)

    def close_buffer(self, name: str) -> None:
        """Close a named workflow buffer; the main buffer cannot be closed."""
        with self._lock:
            self._store.remove_named(name)

    def reset(self) -> None:
        """Forget every buffer; used when the conversation is reset."""
        with self._lock:
            self._generation += 1
            self._store.reset()

    def close(self) -> None:
        """Release any model the scorer holds, then forget the buffers.  Safe to call twice."""
        with self._route_lock:  # wait for an in-flight score before unloading the model
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                self._generation += 1
                self._store.reset()
                close = getattr(self.scorer, "close", None)
                if callable(close):
                    close()
                if self._fallback is not self.scorer:
                    close = getattr(self._fallback, "close", None)
                    if callable(close):
                        close()

    # --- internals -------------------------------------------------------

    def _scores(self, text: str, candidates: Sequence[Sequence[Utterance]]) -> Tuple[List[float], bool]:
        """Return one score per buffer and whether the fallback scorer was used."""
        used_fallback = False
        try:
            values = [float(value) for value in self.scorer.scores(text, candidates)]
            if len(values) != len(candidates):
                raise ValueError(f"scorer returned {len(values)} scores for {len(candidates)} buffers")
            if any(not math.isfinite(value) for value in values):
                raise ValueError("scorer returned a non-finite score")
        except Exception:  # noqa: BLE001 - a broken backend must not silence the assistant
            used_fallback = True
            if not self._warned:
                self._warned = True
                self.logger.warning("Topic relevance scorer failed; using lexical fallback.", exc_info=True)
            try:
                values = [float(value) for value in self._fallback.scores(text, candidates)]
            except Exception:  # noqa: BLE001 - the fallback is best effort as well
                self.logger.warning("Topic fallback scorer failed too; treating the sentence as new.", exc_info=True)
                values = []
            if len(values) != len(candidates) or any(not math.isfinite(value) for value in values):
                values = [0.0] * len(candidates)
        return [max(0.0, min(1.0, value)) for value in values], used_fallback

    def _promote(self, buffer: TopicBuffer, text: str, score: float) -> TopicDecision:
        prior = list(buffer.utterances)[:-1]
        self._store.promote(buffer)
        context = None
        if prior:
            spoken = " / ".join(
                f"{utterance.speaker}: {utterance.text}" for utterance in prior[-self.history_sentences :]
            )
            context = f'(Earlier in this thread: "{spoken}". Continue from there.)'
        self.logger.info("Topic %d promoted to the active conversation.", buffer.id)
        return TopicDecision(
            text=text,
            answer=True,
            buffer_id=buffer.id,
            kind=RouteKind.PROMOTED,
            score=score,
            context=context,
        )
