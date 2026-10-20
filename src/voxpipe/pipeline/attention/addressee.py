"""Addressee-aware floor control for busy, multi-person environments.

This is deliberately a *gate*, not a topic model.  It answers one question:
is this utterance addressed to the assistant, or to someone else in the room?
A tiny personal k-NN model over labeled examples is combined with cheap
dialogue-state and phrase heuristics.  The gate never needs a GPU, learns from
"not you" corrections, and biases to silence.
"""

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ...core.utils import get_logger, write_atomically
from .embeddings import TextEmbedder
from .models import GateDecision, TopicBuffer
from .scorers import LexicalScorer
from .text import normalize_phrase, words
from .tracker import TopicTracker

_NEGATIVE_CORRECTIONS = frozenset(
    {
        "not you",
        "no not you",
        "i was talking to the customer",
        "that was for the customer",
        "not for you",
        "i was not talking to you",
    }
)
_POSITIVE_CORRECTIONS = frozenset(
    {
        "that was me",
        "yes that was me",
        "i was talking to you",
        "that was for you",
        "i meant you",
    }
)

_AI_RE = re.compile(
    r"^(add|note|check|remove|undo|cancel|print|send|tell|remind|set|show|find|look up|calculate)\b"
    r"|^(what'?s|what is)\b.*\b(total|status|price|bill|table|order|next)\b"
    r"|^table\s+\w+|^order\s+\w+",
    re.IGNORECASE,
)
_CUSTOMER_RE = re.compile(
    r"^(hi|hello|hey|good (morning|afternoon|evening)|welcome)\b"
    r"|^(what can i get|what would you like|would you like|do you want|are you ready|for here|to go"
    r"|anything else|that'?ll be|your total|cash or card|here you go|thank you|have a (nice|good)|next)\b",
    re.IGNORECASE,
)


@dataclass
class FloorState:
    """The cheap conversational state used by the addressee model."""

    last_speaker: str = "assistant"
    awaiting_reply: bool = False
    customer_present: bool = False
    order_phase: Optional[str] = None


@dataclass
class Example:
    text: str
    addressed: bool


class AddresseeModel:
    """A personal k-NN classifier with a rules-based cold start.

    Examples are stored as JSON lines and updated by corrections.  When no
    examples exist, the heuristic carries the gate; after a few corrections,
    the k-NN vote takes over.  If the embedder is unavailable, the model keeps
    working from the heuristic.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        *,
        examples_path: Optional[str] = None,
        k_neighbors: int = 5,
        max_examples: int = 1000,
    ):
        self.logger = get_logger(__name__)
        self.embedder = embedder
        self.examples_path = examples_path
        self.k_neighbors = max(1, k_neighbors)
        self.max_examples = max(1, max_examples)
        self._examples: List[Example] = []
        self._vectors: Optional[np.ndarray] = None
        self._embed_failed = False
        self._lock = threading.RLock()
        self._load()

    @property
    def examples(self) -> List[Example]:
        return list(self._examples)

    def learn(self, text: str, addressed: bool) -> None:
        """Add or update one labeled example and persist it."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        normalized = normalize_phrase(cleaned)
        with self._lock:
            for example in self._examples:
                if normalize_phrase(example.text) == normalized:
                    example.text = cleaned
                    example.addressed = addressed
                    break
            else:
                self._examples.append(Example(cleaned, addressed))
            self._examples = self._examples[-self.max_examples :]
            self._vectors = None
            self._embed_failed = False
            self._save()

    def score(self, text: str, state: FloorState) -> float:
        """P(addressed to the assistant), with a rules-based fallback."""
        heuristic = self._heuristic(text, state)
        with self._lock:
            if not self._examples or self._embed_failed:
                return heuristic
            target = self._embed([text])
            vectors = self._example_vectors()
            if target is None or vectors is None:
                return heuristic
            similarities = vectors @ target[0]
            count = min(self.k_neighbors, len(similarities))
            best = np.argsort(similarities)[-count:]
            weights = np.maximum(0.0, similarities[best])
            total = float(weights.sum())
            if total <= 0.0:
                return heuristic
            positive = float(sum(weight for weight, index in zip(weights, best) if self._examples[index].addressed))
            knn = (positive + 1.0) / (total + 2.0)  # Laplace smoothing
            return max(0.0, min(1.0, 0.7 * knn + 0.3 * heuristic))

    def close(self) -> None:
        self.embedder.close()

    def _heuristic(self, text: str, state: FloorState) -> float:
        normalized = normalize_phrase(text)
        if not normalized:
            return 0.5
        ai = bool(_AI_RE.search(normalized))
        customer = bool(_CUSTOMER_RE.search(normalized))
        score = 0.85 if ai and not customer else 0.15 if customer and not ai else 0.5
        if state.awaiting_reply and len(normalized.split()) <= 4 and len(words(normalized)) <= 2:
            score = max(score, 0.80)
        if state.customer_present and not ai:
            score = min(score, 0.40)
        return max(0.0, min(1.0, score))

    def _embed(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        try:
            vectors = [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in self.embedder.embed(texts)]
        except Exception:  # noqa: BLE001 - a model failure must not silence the gate
            if not self._embed_failed:
                self._embed_failed = True
                self.logger.warning("Addressee embedding failed; using heuristics.", exc_info=True)
            return None
        if len(vectors) != len(texts):
            return None
        norms = np.linalg.norm(np.vstack(vectors), axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.vstack(vectors) / norms

    def _example_vectors(self) -> Optional[np.ndarray]:
        if self._vectors is not None:
            return self._vectors
        if not self._examples:
            return None
        vectors = self._embed([example.text for example in self._examples])
        if vectors is None:
            return None
        self._vectors = vectors
        return vectors

    def _load(self) -> None:
        if not self.examples_path:
            return
        path = Path(self.examples_path)
        if not path.exists():
            return
        loaded: List[Example] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                text = str(data["text"])
                addressed = data["addressed"]
                if not isinstance(addressed, bool):
                    raise ValueError("addressed must be true or false")
                if text.strip():
                    loaded.append(Example(text, addressed))
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.logger.warning("Could not read addressee examples from %s: %s", path, e)
            return
        self._examples = loaded[-self.max_examples :]

    def _save(self) -> None:
        if not self.examples_path:
            return
        path = self.examples_path
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            data = "\n".join(
                json.dumps({"text": example.text, "addressed": example.addressed}, ensure_ascii=False)
                for example in self._examples
            )
            write_atomically(path, (data + ("\n" if data else "")).encode("utf-8"))
        except OSError as e:
            self.logger.warning("Could not save addressee examples to %s: %s", path, e)


class AddresseeGate:
    """A silence-biased gate for environments with more than one speaker.

    Explicit input (push-to-talk/typed) always answers.  Everything else is
    scored by :class:`AddresseeModel`; high scores go to the main conversation,
    low scores go to a customer buffer, and the ambiguous band is kept as
    background.  ``defer_ms`` asks the pipeline to wait for a possible reply
    before speaking.
    """

    def __init__(
        self,
        model: AddresseeModel,
        *,
        tracker: Optional[TopicTracker] = None,
        defer_ms: int = 0,
        speak_threshold: float = 0.70,
        silence_threshold: float = 0.35,
        shadow: bool = False,
        customer_ttl_sentences: int = 50,
        background_ttl_sentences: int = 10,
    ):
        if not 0.0 <= silence_threshold < speak_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= silence_threshold < speak_threshold <= 1.")
        if defer_ms < 0:
            raise ValueError("defer_ms must not be negative.")
        self.model = model
        self.tracker = tracker or TopicTracker(LexicalScorer())
        self.defer_ms = defer_ms
        self.speak_threshold = speak_threshold
        self.silence_threshold = silence_threshold
        self.shadow = shadow
        self.customer_ttl_sentences = customer_ttl_sentences
        self.background_ttl_sentences = background_ttl_sentences
        self.state = FloorState()
        self._last_text: Optional[str] = None
        self._lock = threading.RLock()
        self._closed = False

    @property
    def examples(self) -> List[Example]:
        return self.model.examples

    def evaluate(self, text: str, *, explicit: bool = False) -> GateDecision:
        with self._lock:
            if self._closed:
                return GateDecision(answer=False, reason="closed")
        cleaned = (text or "").strip()
        if not cleaned:
            return GateDecision(answer=True, reason="empty")
        normalized = normalize_phrase(cleaned)
        if normalized in _NEGATIVE_CORRECTIONS:
            self.correct_last(False)
            return GateDecision(answer=False, confidence=1.0, reason="correction")
        if normalized in _POSITIVE_CORRECTIONS:
            self.correct_last(True)
            return GateDecision(answer=False, confidence=1.0, reason="correction")

        with self._lock:
            self._last_text = cleaned
            if explicit:
                self.tracker.record_user(cleaned)
                self.state.last_speaker = "user"
                self.state.awaiting_reply = False
                return GateDecision(answer=True, confidence=1.0, reason="explicit")

            score = self.model.score(cleaned, self.state)
            self.state.last_speaker = "user"
            self.state.awaiting_reply = False

            if score >= self.speak_threshold:
                self.tracker.record_user(cleaned)
                return GateDecision(
                    answer=not self.shadow,
                    confidence=score,
                    reason="shadow" if self.shadow else "addressed",
                    defer_ms=self.defer_ms,
                )

            target = "customer" if score <= self.silence_threshold else "background"
            ttl = self.customer_ttl_sentences if target == "customer" else self.background_ttl_sentences
            self.tracker.route_to(target, cleaned, ttl_sentences=ttl)
            return GateDecision(
                answer=False,
                confidence=score,
                reason="shadow" if self.shadow else target,
            )

    def record_user(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        with self._lock:
            if self._closed:
                return
            self._last_text = cleaned
            self.tracker.record_user(cleaned)
            self.state.last_speaker = "user"
            self.state.awaiting_reply = False

    def record_reply(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        with self._lock:
            if self._closed:
                return
            self.tracker.record_reply(cleaned)
            self.state.last_speaker = "assistant"
            self.state.awaiting_reply = cleaned.rstrip().endswith("?")

    def correct_last(self, addressed: bool) -> None:
        """Label the most recent utterance; used by “not you” / “that was me”."""
        with self._lock:
            if not self._closed and self._last_text:
                self.model.learn(self._last_text, addressed)

    def set_customer_present(self, present: bool) -> None:
        with self._lock:
            self.state.customer_present = present

    def set_order_phase(self, phase: Optional[str]) -> None:
        with self._lock:
            self.state.order_phase = phase

    def buffer(self, name: str) -> Optional[TopicBuffer]:
        return self.tracker.buffer(name)

    def reset(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.tracker.reset()
            self.state = FloorState()
            self._last_text = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.tracker.close()
            self.model.close()
