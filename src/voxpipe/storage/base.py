"""High-level MemoryStore facade wrapping an interchangeable Storage backend."""

from typing import Any, List, Optional
from .protocols import QueryInput, Storage
from .ram import RAMStorage
from .record import Record


class MemoryStore:
    """High-level MemoryStore facade managing backend delegation and bank namespace scoping."""

    def __init__(self, backend: Optional[Storage] = None, bank: str = "global"):
        self.backend = backend or RAMStorage()
        self.bank = bank

    def store(self, content: Any, **kwargs) -> Record:
        """Store content under the bound bank namespace."""
        target_bank = kwargs.pop("bank", self.bank)
        return self.backend.store(content, bank=target_bank, **kwargs)

    def retrieve(self, query: QueryInput, top_k: int = 3, **kwargs) -> List[Record]:
        """Retrieve matching Records from the bound bank (plus optional fallbacks)."""
        banks = kwargs.pop("banks", None)
        if banks is None:
            fallback = kwargs.pop("fallback", "global")
            banks = [self.bank]
            if fallback and fallback != self.bank:
                banks.append(fallback)

        return self.backend.retrieve(query, banks=banks, top_k=top_k, **kwargs)
