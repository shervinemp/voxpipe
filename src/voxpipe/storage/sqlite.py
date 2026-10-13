"""SQLite-backed storage engine with FTS5 keyword search implementing the Storage protocol."""

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from ..core.exceptions import StorageError
from ..core.utils import get_logger
from .protocols import QueryInput, keywords, search_banks, store_bank, unpack_query
from .record import Record

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bank TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_bank ON records(bank);
CREATE INDEX IF NOT EXISTS idx_records_created ON records(created_at);
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts
    USING fts5(content, content='records', content_rowid='id', tokenize='porter');
CREATE TRIGGER IF NOT EXISTS records_fts_insert AFTER INSERT ON records BEGIN
    INSERT INTO records_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS records_fts_delete AFTER DELETE ON records BEGIN
    INSERT INTO records_fts(records_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
"""

# Rows fetched per requested result, so duplicates can be dropped without a second query.
_OVERFETCH = 4


class SQLiteStorage:
    """Persistent storage with multi-bank scoping, BM25-ranked recall, and TTL/count eviction."""

    def __init__(self, db_path: str, max_entries: int = 1000, ttl_days: Optional[float] = 30):
        """``ttl_days`` of None or 0 keeps entries until ``max_entries`` pushes them out."""
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1.")
        self.logger = get_logger(__name__)
        self.db_path = db_path
        self.max_entries = max_entries
        self.ttl_seconds = ttl_days * 86400 if ttl_days else None
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        except sqlite3.Error as e:
            raise StorageError(f"Could not open {db_path}: {e}") from e
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._migrate_legacy_tables()
        except sqlite3.Error as e:
            self._conn.close()  # or the file stays locked on Windows
            hint = " (this SQLite build lacks FTS5, which voxpipe needs)" if "fts5" in str(e).lower() else ""
            raise StorageError(f"Could not initialise {db_path}: {e}{hint}") from e

    def _migrate_legacy_tables(self) -> None:
        """Carry rows over from the pre-0.2 ``conversations`` table, then drop it."""
        tables = {row[0] for row in self._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "conversations" not in tables:
            return
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(conversations)")}
        bank = "COALESCE(session_id, 'global')" if "session_id" in columns else "'global'"
        with self._conn:
            self._conn.execute(
                "INSERT INTO records (bank, role, content, meta, created_at) "
                f"SELECT {bank}, role, content, COALESCE(meta, '{{}}'), created_at "
                "FROM conversations ORDER BY created_at"
            )
            self._conn.execute("DROP TABLE conversations")
            self._conn.execute("DROP TABLE IF EXISTS conv_fts")
        self.logger.info("Migrated legacy conversation history in %s", self.db_path)

    def store(
        self,
        content: Any,
        *,
        bank: Optional[str] = None,
        role: str = "user",
        meta: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Record:
        """Store ``content`` as text; a dict is saved as JSON and is recalled as that JSON string."""
        bank = store_bank(bank, session_id)
        text = json.dumps(content) if isinstance(content, dict) else str(content)
        now = time.time()

        with self._lock, self._translating_errors("store to"), self._conn:
            self._conn.execute(
                "INSERT INTO records (bank, role, content, meta, created_at) VALUES (?, ?, ?, ?, ?)",
                (bank, role, text, json.dumps(meta or {}), now),
            )
            self._evict(now)

        return Record(
            content=content,
            source=f"sqlite:{bank}",
            meta={**(meta or {}), "role": role, "created_at": now, "bank": bank},
        )

    def retrieve(
        self,
        query: QueryInput,
        *,
        banks: Union[str, Iterable[str], None] = None,
        top_k: int = 3,
        role: Optional[str] = None,
        session_id: Optional[str] = None,
        include_global: bool = True,
    ) -> List[Record]:
        """Return up to ``top_k`` records matching any query keyword, best BM25 match first."""
        text, top_k = unpack_query(query, top_k)
        terms = keywords(text)
        if not terms or top_k <= 0:
            return []
        targets = search_banks(banks, session_id, include_global)

        # Each term is quoted so FTS5 reads it as a phrase, never as query syntax.
        match = " OR ".join('"{}"'.format(term.replace('"', '""')) for term in terms)
        sql = (
            "SELECT r.content, r.role, r.created_at, r.bank, r.meta, bm25(records_fts) AS rank "
            "FROM records_fts JOIN records r ON r.id = records_fts.rowid "
            f"WHERE records_fts MATCH ? AND r.bank IN ({', '.join('?' * len(targets))})"
        )
        params: List[Any] = [match, *targets]
        if role:
            sql += " AND r.role = ?"
            params.append(role)
        if self.ttl_seconds is not None:
            # Expired rows are only deleted on the next store(); until then they must not be recalled.
            sql += " AND r.created_at >= ?"
            params.append(time.time() - self.ttl_seconds)
        sql += " ORDER BY rank, r.id DESC LIMIT ?"
        params.append(top_k * _OVERFETCH)

        with self._lock, self._translating_errors("search"):
            rows = self._conn.execute(sql, params).fetchall()

        records: Dict[str, Record] = {}
        for content, row_role, created_at, bank, meta_json, rank in rows:
            if content in records:
                continue
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except json.JSONDecodeError:
                meta = {}
            records[content] = Record(
                content=content,
                source=f"sqlite:{bank}",
                score=-rank,  # bm25() is negative; closer to zero means a weaker match
                meta={**meta, "role": row_role, "created_at": created_at, "bank": bank},
            )
            if len(records) >= top_k:
                break
        return list(records.values())

    def _evict(self, now: float) -> None:
        """Drop expired rows, then the oldest rows beyond ``max_entries``. Caller holds the lock."""
        if self.ttl_seconds is not None:
            self._conn.execute("DELETE FROM records WHERE created_at < ?", (now - self.ttl_seconds,))
        surplus = self._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] - self.max_entries
        if surplus > 0:
            self._conn.execute(
                "DELETE FROM records WHERE id IN (SELECT id FROM records ORDER BY id ASC LIMIT ?)", (surplus,)
            )

    @contextmanager
    def _translating_errors(self, action: str) -> Iterator[None]:
        """Report SQLite failures (a closed or locked database, a full disk) as StorageError."""
        try:
            yield
        except sqlite3.Error as e:
            raise StorageError(f"Could not {action} {self.db_path}: {e}") from e

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteStorage":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
