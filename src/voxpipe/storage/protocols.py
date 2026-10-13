"""Structural protocols for retrieval and storage, plus the query and scoping helpers backends share."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Union, runtime_checkable

from .record import Record

GLOBAL_BANK = "global"
_SESSION_PREFIX = "session:"

# Words too common to say anything about what a query is looking for.
_STOP_WORDS = frozenset(
    """
    about all also and any are because been before being but can could did does each for from had has have
    her here him his how into its just more most not off only other our out over she should some such than
    that the their them then there these they this those through too under until very was were what when
    where which while who whom why will with would you your
    """.split()
)


@dataclass
class Query:
    """Structured query for keyword, vector, and hybrid retrieval."""

    text: Optional[str] = None
    embedding: Optional[List[float]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    top_k: Optional[int] = None

    def __str__(self) -> str:
        return self.text or ""


QueryInput = Union[str, Query]


@runtime_checkable
class Retriever(Protocol):
    """Read-only retrieval interface (web search, static document RAG, ...).

    Implementations may accept further keyword arguments (scoping, filters).
    """

    def retrieve(self, query: QueryInput, *, top_k: int = 3) -> List[Record]: ...


@runtime_checkable
class Storer(Protocol):
    """Write-only storage interface (loggers, event archival, ...).

    Implementations may accept further keyword arguments (scoping, metadata).
    """

    def store(self, content: Any, *, role: str = "user") -> Record: ...


@runtime_checkable
class Storage(Retriever, Storer, Protocol):
    """Combined read and write storage interface."""


def unpack_query(query: QueryInput, top_k: int) -> Tuple[str, int]:
    """Return (query text, result limit); a Query's own ``top_k`` wins over the argument."""
    if isinstance(query, Query):
        return query.text or "", query.top_k if query.top_k is not None else top_k
    return str(query), top_k


def keywords(text: str) -> List[str]:
    """Distinct search terms of ``text``: lowercased words minus stop words and fragments."""
    seen: Dict[str, None] = {}
    for word in re.findall(r"\w+", text.lower()):
        if len(word) > 2 and word not in _STOP_WORDS:
            seen.setdefault(word)
    return list(seen)


def session_bank(session_id: str) -> str:
    """The bank holding one session's private records."""
    session_id = str(session_id)
    return session_id if session_id.startswith(_SESSION_PREFIX) else f"{_SESSION_PREFIX}{session_id}"


def store_bank(bank: Optional[str], session_id: Optional[str]) -> str:
    """Bank a record is written to: an explicit bank, else the session's, else the global one."""
    if bank:
        return bank
    return session_bank(session_id) if session_id else GLOBAL_BANK


def search_banks(
    banks: Union[str, Iterable[str], None], session_id: Optional[str], include_global: bool
) -> List[str]:
    """Banks a retrieval reads: explicit banks, else the session's (plus global), else global."""
    if banks:
        return [banks] if isinstance(banks, str) else list(banks)
    if session_id:
        return [session_bank(session_id), GLOBAL_BANK] if include_global else [session_bank(session_id)]
    return [GLOBAL_BANK]
