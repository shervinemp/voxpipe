"""Universal record container for retrieved and stored memory items."""

from dataclasses import dataclass, field
import json
from typing import Any, Dict


@dataclass
class Record:
    """Universal record payload returned by any storage, search, or retrieval component."""

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
        """Convenience property returning string representation of content."""
        return str(self)

    def __getitem__(self, key: str) -> Any:
        """Dual-access support: record['content'] or record.content."""
        if isinstance(key, str) and hasattr(self, key):
            return getattr(self, key)
        if isinstance(key, str) and key in self.meta:
            return self.meta[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        """Support 'key in record' syntax."""
        if isinstance(key, str):
            return hasattr(self, key) or key in self.meta
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get access for backward compatibility."""
        try:
            return self[key]
        except (KeyError, TypeError):
            return default
