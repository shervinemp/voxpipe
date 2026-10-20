"""Small text helpers shared by the attention scorers and policies.

The lexical helpers are intentionally English-biased: they are a
zero-dependency baseline, not a multilingual semantic model.  Keeping them in
one module makes that limitation visible and makes it easy to swap in a
language-aware implementation later.
"""

import re
from functools import lru_cache
from typing import AbstractSet, FrozenSet, Tuple

_STOPWORDS = frozenset(
    """
    a about above after again against all also already am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just later lot maybe
    me more most much my myself no nor not now of off on once only or other our ours ourselves out over own
    perhaps quite same she should so some soon still such than that the their theirs them themselves then
    there these they thing things this those through to today tomorrow tonight too under until up very was
    way we were what when where which while who whom why will with would yet you your yours yourself
    yourselves
    """.split()
)

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def stem(word: str) -> str:
    """A deliberately small stemmer: plurals and past tenses, no dependency."""
    if len(word) > 4 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


@lru_cache(maxsize=4096)
def words(text: str) -> Tuple[str, ...]:
    """Lower-case content words, without punctuation or common stopwords."""
    return tuple(stem(word) for word in _WORD_RE.findall(text.lower()) if word not in _STOPWORDS)


@lru_cache(maxsize=2048)
def trigrams(text: str) -> FrozenSet[str]:
    """Character trigrams over the compact lower-case text."""
    compact = "".join(_WORD_RE.findall(text.lower()))
    if len(compact) < 3:
        return frozenset({compact}) if compact else frozenset()
    return frozenset(compact[index : index + 3] for index in range(len(compact) - 2))


@lru_cache(maxsize=2048)
def clauses(text: str) -> Tuple[str, ...]:
    """Split a long reply into sentence-sized pieces for fair comparison."""
    pieces = tuple(piece.strip() for piece in re.split(r"(?<=[.!?;])\s+|\n+", text) if piece.strip())
    return pieces or (text,)


def normalize_phrase(text: str) -> str:
    """Lower-case, collapse whitespace, and strip terminal punctuation."""
    return " ".join(text.lower().strip().rstrip(".!?,").split())


def jaccard(left: AbstractSet[str], right: AbstractSet[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
