"""High-level MemoryStore facade wrapping an interchangeable Storage backend."""

from typing import Any, List
from .protocols import QueryInput, Storage
from .ram import RAMStorage
from .record import Record


class MemoryStore:
    """High-level MemoryStore facade managing backend delegation and collection namespace scoping."""

    def __init__(
        self,
        storage: Storage,
    ):
        self.storage = storage or RAMStorage()

    def store(self, content: Any, **kwargs) -> Record:
        """Store content under the bound collection namespace."""
        return self.storage.store(content, **kwargs)

    def retrieve(self, query: QueryInput, **kwargs) -> List[Record]:
        """Retrieve matching Records from the bound collection."""
        return self.storage.retrieve(query, **kwargs)
