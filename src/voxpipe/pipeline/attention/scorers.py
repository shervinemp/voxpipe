"""Relatedness scorers: lexical, embedding and Laya.

Each scorer declares ``default_threshold`` on its own scale, so the factory and
the tracker do not need a separate backend-to-threshold table.
"""

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...core.exceptions import ConfigError
from ...core.utils import get_logger
from .models import Utterance
from .text import clauses, jaccard, trigrams, words

_EMBEDDING_CACHE_SIZE = 4096
_LAYA_FAILURE_LIMIT = 3
_LAYA_COOLDOWN_SECONDS = 30.0


class LexicalScorer:
    """Word and character-trigram overlap; no model and no extra dependency.

    This is a useful baseline for names, places and repeated vocabulary.  It
    will not recognise a paraphrase that shares no words, which is exactly why
    the embedding and Laya backends exist.  Its stopword list and stemmer are
    English-leaning by design.
    """

    default_threshold = 0.12

    def scores(self, text: str, candidates: Sequence[Sequence[Utterance]]) -> List[float]:
        target_words = set(words(text))
        target_chars = trigrams(text)
        results: List[float] = []
        for candidate in candidates:
            best = 0.0
            for utterance in candidate:
                for clause in clauses(utterance.text):
                    best = max(best, self._pair(target_words, target_chars, clause))
            results.append(best)
        return results

    @staticmethod
    def _pair(target_words: set[str], target_chars: frozenset[str], clause: str) -> float:
        candidate_words = set(words(clause))
        if not target_words or not candidate_words:
            token = 0.0
        elif len(target_words) <= 2:
            # A short answer such as "Paris" should score highly when the name appears,
            # even though a long candidate buffer makes ordinary Jaccard tiny.
            token = len(target_words & candidate_words) / len(target_words)
        else:
            token = len(target_words & candidate_words) / len(target_words | candidate_words)
        char = jaccard(target_chars, trigrams(clause))
        return 0.6 * token + 0.4 * char


class EmbeddingScorer:
    """Cosine similarity over a small fastembed bi-encoder.

    One forward pass embeds the new sentence and the sentences of every
    candidate buffer; comparing the vectors is then just a dot product.  This
    is the recommended backend for always-on tracking: it understands
    paraphrases and is far cheaper than running a large decision model once per
    buffer.
    """

    default_threshold = 0.55

    def __init__(
        self,
        *,
        model: str,
        cache_dir: Optional[str] = None,
        embedder: Any = None,
        max_cache: int = _EMBEDDING_CACHE_SIZE,
    ):
        self.model = model
        self._embedder = embedder if embedder is not None else self._load(model, cache_dir)
        self._max_cache = max_cache
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    @staticmethod
    def _load(model: str, cache_dir: Optional[str]) -> Any:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ConfigError(
                "The 'embedding' topic backend needs the fastembed package: pip install 'voxpipe[embed]'."
            ) from e
        try:
            return TextEmbedding(model_name=model, cache_dir=cache_dir)
        except Exception as e:  # noqa: BLE001 - a missing or corrupt model is a configuration problem
            raise ConfigError(f"Could not load the topic embedding model {model!r}: {e}") from e

    def scores(self, text: str, candidates: Sequence[Sequence[Utterance]]) -> List[float]:
        texts: List[str] = []
        spans: List[tuple[int, int]] = []
        for candidate in candidates:
            start = len(texts)
            for utterance in candidate:
                texts.extend(clauses(utterance.text))
            spans.append((start, len(texts)))
        texts.append(text)
        vectors = self._vectors(texts)
        target = vectors[-1]
        results: List[float] = []
        for start, end in spans:
            best = max(
                (float(np.dot(target, vectors[index])) for index in range(start, end)),
                default=0.0,
            )
            results.append(max(0.0, min(1.0, best)))
        return results

    def _vectors(self, texts: Sequence[str]) -> List[np.ndarray]:
        if self._embedder is None:
            raise RuntimeError("Topic embedding model has been closed.")
        # Snapshot everything available for this batch.  A tiny cache can evict a
        # sentence while the rest of the batch is being written; the local map keeps
        # the result complete regardless.
        batch: Dict[str, np.ndarray] = {text: self._cache[text] for text in texts if text in self._cache}
        missing = [text for text in texts if text not in batch]
        if missing:
            try:
                produced = list(self._embedder.embed(missing))
            except Exception as e:  # noqa: BLE001 - surfaced to the tracker, which can fall back
                raise RuntimeError(f"Topic embedding failed: {e}") from e
            if len(produced) != len(missing):
                raise RuntimeError(
                    f"Topic embedding returned {len(produced)} vectors for {len(missing)} sentences."
                )
            for sentence, vector in zip(missing, produced):
                unit = _unit(vector)
                batch[sentence] = unit
                self._put(sentence, unit)
        return [batch[text] for text in texts]

    def _put(self, text: str, vector: np.ndarray) -> None:
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)

    def close(self) -> None:
        close = getattr(self._embedder, "close", None)
        if callable(close):
            close()
        self._embedder = None
        self._cache.clear()


def _unit(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else array


_LAYA_QUESTION = {
    "related": {
        "type": "noul",
        "instructions": (
            "The new utterance is a continuation of the conversation excerpt: it replies to it, "
            "follows it, or stays on the same topic."
        ),
    }
}


class LayaScorer:
    """Use Laya's yes/no decision head to judge each candidate buffer.

    Laya is a 322-421M parameter encoder, so this backend is heavier than the
    embedding one.  Requests for all buffers are sent through
    :meth:`laya.Router.predict_batch` where available, so an utterance costs one
    batched forward pass rather than one call per buffer.

    A short circuit breaker stops repeated failed calls from dominating
    latency: after a few consecutive failures the scorer stays on the fallback
    for a cooldown period.
    """

    default_threshold = 0.50

    def __init__(
        self,
        *,
        model: str = "auto",
        device: Optional[str] = None,
        preload: bool = False,
        max_len: int = 1024,
        router: Any = None,
        failure_limit: int = _LAYA_FAILURE_LIMIT,
        cooldown_seconds: float = _LAYA_COOLDOWN_SECONDS,
    ):
        self.logger = get_logger(__name__)
        self.model = model
        self.max_len = max_len
        self._max_state_chars = max(500, max_len * 4)
        self._router = router if router is not None else self._load(model, device, preload)
        self._failure_limit = max(1, failure_limit)
        self._cooldown_seconds = max(0.0, cooldown_seconds)
        self._failures = 0
        self._open_until = 0.0

    @staticmethod
    def _load(model: str, device: Optional[str], preload: bool) -> Any:
        try:
            from laya import Router
        except ImportError as e:
            raise ConfigError(
                "The 'laya' topic backend needs the laya package: pip install 'voxpipe[laya]'."
            ) from e
        options: Dict[str, Any] = {}
        if device:
            options["device"] = device
        if model != "auto":
            options["default"] = model
        try:
            router = Router(**options)
            if preload:
                router.preload([model] if model != "auto" else ["english", "multilingual"])
            return router
        except Exception as e:  # noqa: BLE001 - bad checkpoint, missing weights, or no torch
            raise ConfigError(f"Could not start the Laya topic backend: {e}") from e

    def scores(self, text: str, candidates: Sequence[Sequence[Utterance]]) -> List[float]:
        if time.monotonic() < self._open_until:
            raise RuntimeError("Laya topic scorer is cooling down after repeated failures.")
        try:
            requests = [self._request(text, candidate) for candidate in candidates]
            values = [self._probability(result) for result in self._predict(requests)]
        except Exception:
            self._failures += 1
            if self._failures >= self._failure_limit:
                self._open_until = time.monotonic() + self._cooldown_seconds
                self._failures = 0
            raise
        self._failures = 0
        return values

    def _request(self, text: str, candidate: Sequence[Utterance]) -> Dict[str, Any]:
        conversation = "\n".join(f"{utterance.speaker.capitalize()}: {utterance.text}" for utterance in candidate)
        if len(conversation) > self._max_state_chars:
            conversation = conversation[-self._max_state_chars :]
        request: Dict[str, Any] = {
            "state": {"conversation": conversation, "new_utterance": text},
            "questions": _LAYA_QUESTION,
            "max_len": self.max_len,
        }
        if self.model != "auto":
            request["model"] = self.model
        return request

    def _predict(self, requests: Sequence[Dict[str, Any]]) -> List[Any]:
        if not requests:
            return []
        if hasattr(self._router, "predict_batch"):
            try:
                results = list(self._router.predict_batch(requests))
                if len(results) == len(requests):
                    return results
            except Exception:  # noqa: BLE001 - an older or partial Router may not batch this request
                self.logger.warning("Laya batch prediction failed; trying one request at a time.", exc_info=True)
        results = []
        for request in requests:
            options = {key: value for key, value in request.items() if key not in ("state", "questions")}
            results.append(self._router.predict(request["state"], request["questions"], **options))
        return results

    @staticmethod
    def _probability(result: Any) -> float:
        """Extract the answer, raising when the response does not have the expected shape.

        A malformed response is a bug or a version mismatch, not evidence that the
        utterance is unrelated.  Raising lets the tracker use its fallback scorer
        instead of silently treating everything as a new topic.
        """
        answer = result["answers"]["related"]
        value = answer["noul"] if isinstance(answer, dict) else answer
        return max(0.0, min(1.0, float(value)))

    def close(self) -> None:
        """Release the checkpoint when the pipeline shuts down."""
        unload = getattr(self._router, "unload", None)
        if callable(unload):
            unload()
