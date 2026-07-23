"""SQLite-backed conversation memory engine adhering to Storage protocol."""

from .sqlite import SQLiteStorage

# Backward compatibility alias
Memory = SQLiteStorage

__all__ = ["Memory", "SQLiteStorage"]
