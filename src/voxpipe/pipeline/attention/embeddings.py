"""Embedding backends for the personal addressee classifier.

The busy preset prefers fastembed, but falls back to a deterministic hashed
bag-of-words embedder so it still works with the base install.  The fallback is
lexical, not semantic; it is there to keep the gate usable and to collect
labels, not to replace a real embedding model.
"""

import hashlib
import re
from typing import List, Optional, Protocol, Sequence

import numpy as np

from ...core.exceptions import ConfigError
from ...core.utils import get_logger

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


class TextEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[np.ndarray]: ...

    def close(self) -> None: ...


class FastEmbedder:
    """A thin wrapper around fastembed's small bi-encoder."""

    def __init__(self, model: str, cache_dir: Optional[str] = None):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ConfigError("The addressee embedding backend needs fastembed: pip install 'voxpipe[embed]'.") from e
        try:
            self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)
        except Exception as e:  # noqa: BLE001 - a missing or corrupt model is a configuration problem
            raise ConfigError(f"Could not load the addressee embedding model {model!r}: {e}") from e

    def embed(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [np.asarray(vector, dtype=np.float32) for vector in self._model.embed(texts)]

    def close(self) -> None:
        close = getattr(self._model, "close", None)
        if callable(close):
            close()


class HashingEmbedder:
    """Deterministic signed token hashing; no model, no download, no network."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = max(16, dimensions)

    def embed(self, texts: Sequence[str]) -> List[np.ndarray]:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                index = value % self.dimensions
                sign = 1.0 if (value >> 8) & 1 else -1.0
                vectors[row, index] += sign
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result: List[np.ndarray] = []
        for index in range(len(texts)):
            result.append(vectors[index] / norms[index, 0])
        return result

    def close(self) -> None:
        pass


def create_embedder(
    model: str,
    cache_dir: Optional[str] = None,
    *,
    allow_fallback: bool = True,
) -> TextEmbedder:
    """Build the preferred embedder, optionally falling back to hashing."""
    try:
        return FastEmbedder(model, cache_dir)
    except ConfigError as e:
        if not allow_fallback:
            raise
        get_logger(__name__).warning(
            "Addressee embedding model unavailable (%s); using the lexical hashing fallback. "
            "Install 'voxpipe[embed]' for semantic matching.",
            e,
        )
        return HashingEmbedder()
