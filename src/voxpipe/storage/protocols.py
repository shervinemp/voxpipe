"""Structural protocols for retrieval, storage, and search operations."""

from dataclasses import dataclass, field
from typing import Protocol, Any, Dict, List, Optional, Union, runtime_checkable
from .record import Record


@dataclass
class Query:
    """Structured query object for keyword, vector, and hybrid retrieval."""

    text: Optional[str] = None
    embedding: Optional[List[float]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 3

    def __str__(self) -> str:
        return self.text or ""


QueryInput = Union[str, List[float], Query, Any]


@runtime_checkable
class Retriever(Protocol):
    """Read-only retrieval interface (Web Search, static PDF RAG, etc.)."""

    def retrieve(self, query: QueryInput, **kwargs) -> List[Record]:
        ...


@runtime_checkable
class Storer(Protocol):
    """Write-only storage interface (Loggers, event archival)."""

    def store(self, content: Any, **kwargs) -> Any:
        ...


@runtime_checkable
class Storage(Retriever, Storer, Protocol):
    """Combined Read & Write storage interface (RAMStorage, SQLiteStorage, etc.)."""

    ...
