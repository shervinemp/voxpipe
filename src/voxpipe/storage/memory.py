"""MemoryStore: a Storage backend bound to one session's namespace."""

from typing import Any, List, Optional

from .protocols import QueryInput, Storage
from .ram import RAMStorage
from .record import Record


class MemoryStore:
    """Scopes an interchangeable Storage backend to a session.

    Records are written to the session's bank (the shared global bank when
    ``session_id`` is None) and recalled from it, plus the global bank unless
    ``include_global`` is False.  ``storage`` may be reassigned at any time to
    swap backends.
    """

    def __init__(
        self,
        storage: Optional[Storage] = None,
        *,
        session_id: Optional[str] = None,
        include_global: bool = True,
    ):
        # Not `storage or ...`: an empty backend that defines __len__ is falsy but perfectly usable.
        self.storage: Storage = storage if storage is not None else RAMStorage()
        self.session_id = session_id
        self.include_global = include_global

    def store(self, content: Any, **kwargs) -> Record:
        """Store content in this store's bank."""
        return self.storage.store(content, **self._scoped(kwargs))

    def retrieve(self, query: QueryInput, **kwargs) -> List[Record]:
        """Retrieve matching records visible to this store's session."""
        if self.session_id is not None:
            kwargs.setdefault("include_global", self.include_global)
        return self.storage.retrieve(query, **self._scoped(kwargs))

    def _scoped(self, kwargs: dict) -> dict:
        if self.session_id is not None:
            kwargs.setdefault("session_id", self.session_id)
        return kwargs
