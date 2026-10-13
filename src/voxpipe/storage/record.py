"""Universal record container for retrieved and stored memory items."""

import json
from dataclasses import dataclass, field, fields
from typing import Any, Dict


@dataclass
class Record:
    """Payload returned by any storage, search, or retrieval component."""

    content: Any
    source: str = "global"
    score: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, dict):
            return json.dumps(self.content)
        return str(self.content)

    @property
    def text(self) -> str:
        """The content rendered as a string."""
        return str(self)

    def __getitem__(self, key: str) -> Any:
        """Mapping-style access to the record's fields, then to its metadata."""
        if key in _FIELD_NAMES:
            return getattr(self, key)
        try:
            return self.meta[key]
        except (KeyError, TypeError):
            raise KeyError(key) from None

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (key in _FIELD_NAMES or key in self.meta)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


_FIELD_NAMES = frozenset(f.name for f in fields(Record))
