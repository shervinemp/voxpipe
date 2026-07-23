"""SQLite-backed storage engine with FTS5 keyword search implementing Storage protocol."""

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any, List, Optional

from voxpipe.core.utils import get_logger
from .protocols import QueryInput, Storage
from .record import Record


class SQLiteStorage(Storage):
    """Production SQLite storage engine with WAL mode and multi-bank support."""

    def __init__(self, db_path: str, max_entries: int = 1000, ttl_days: int = 30):
        self.logger = get_logger(__name__)
        self.db_path = db_path
        self.max_entries = max_entries
        self.ttl_seconds = ttl_days * 86400
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at);
                CREATE INDEX IF NOT EXISTS idx_conv_role ON conversations(role);
                CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
            """)
            cur = self._conn.execute("PRAGMA table_info(conversations)")
            cols = [row[1] for row in cur.fetchall()]
            if "session_id" not in cols:
                self._conn.execute("ALTER TABLE conversations ADD COLUMN session_id TEXT")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id)")

            try:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS conv_fts "
                    "USING fts5(content, tokenize='porter')"
                )
                self._conn.execute(
                    "CREATE TRIGGER IF NOT EXISTS conv_fts_del "
                    "AFTER DELETE ON conversations BEGIN "
                    "DELETE FROM conv_fts WHERE rowid = old.rowid; END;"
                )
            except sqlite3.OperationalError:
                pass

    def store(self, content: Any, bank: str = "global", **kwargs) -> Record:
        role = kwargs.get("role", "user")
        meta = kwargs.get("meta") or {}
        if "bank" not in meta:
            meta["bank"] = bank

        content_str = json.dumps(content) if isinstance(content, dict) else str(content)
        entry_id = hashlib.md5(f"{role}:{content_str}:{time.time()}".encode()).hexdigest()[:16]
        meta_str = json.dumps(meta)
        now = time.time()

        record = Record(
            content=content,
            source=f"sqlite:{bank}",
            score=1.0,
            meta={"role": role, "created_at": now, "bank": bank, **meta},
        )

        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO conversations "
                "(id, session_id, role, content, meta, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (entry_id, bank, role, content_str, meta_str, now),
            )
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO conv_fts (rowid, content) "
                    "VALUES (last_insert_rowid(), ?)", (content_str,),
                )
            except sqlite3.OperationalError:
                pass
            self._evict_old()

        return record

    def retrieve(
        self,
        query: QueryInput,
        banks: Optional[List[str] | str] = "global",
        top_k: int = 3,
        **kwargs,
    ) -> List[Record]:
        from .protocols import Query
        if isinstance(query, Query):
            query_str = query.text or ""
            top_k = query.top_k or top_k
        else:
            query_str = str(query)

        role = kwargs.get("role")
        keywords = [w for w in query_str.lower().split() if len(w) > 2]
        if not keywords:
            return []

        if isinstance(banks, str):
            target_banks = [banks]
        elif isinstance(banks, list):
            target_banks = banks
        else:
            target_banks = ["global"]

        placeholders = ",".join("?" for _ in target_banks)
        sess_clause = f" AND (c.session_id IN ({placeholders}) OR c.session_id IS NULL) " if target_banks else ""

        results: List[Record] = []
        seen = set()

        try:
            for kw in keywords[:5]:
                sql_params = [f'"{kw}"']
                if role:
                    sql_params.append(role)
                    sql_params.extend(target_banks)
                    sql_params.append(top_k)
                    cur = self._conn.execute(
                        "SELECT c.content, c.role, c.created_at, c.session_id, c.meta "
                        "FROM conversations c WHERE c.rowid IN ("
                        "SELECT rowid FROM conv_fts WHERE conv_fts MATCH ?)"
                        f"AND c.role = ? {sess_clause} ORDER BY c.created_at DESC LIMIT ?",
                        tuple(sql_params),
                    )
                else:
                    sql_params.extend(target_banks)
                    sql_params.append(top_k)
                    cur = self._conn.execute(
                        "SELECT c.content, c.role, c.created_at, c.session_id, c.meta "
                        "FROM conversations c WHERE c.rowid IN ("
                        "SELECT rowid FROM conv_fts WHERE conv_fts MATCH ?)"
                        f"{sess_clause} ORDER BY c.created_at DESC LIMIT ?",
                        tuple(sql_params),
                    )

                for row in cur.fetchall():
                    c_text, c_role, c_time, c_bank, c_meta_raw = row
                    if c_text not in seen:
                        seen.add(c_text)
                        try:
                            c_meta = json.loads(c_meta_raw) if c_meta_raw else {}
                        except Exception:
                            c_meta = {}

                        rec = Record(
                            content=c_text,
                            source=f"sqlite:{c_bank or 'global'}",
                            score=1.0,
                            meta={"role": c_role, "created_at": c_time, "bank": c_bank, **c_meta},
                        )
                        results.append(rec)
                        if len(results) >= top_k:
                            return results
            return results
        except sqlite3.OperationalError:
            return []

    def _evict_old(self):
        cutoff = time.time() - (self.ttl_seconds if self.ttl_seconds > 0 else 1.0)
        self._conn.execute("DELETE FROM conversations WHERE created_at < ?", (cutoff,))
        count = self._conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        if count > self.max_entries:
            to_remove = count - self.max_entries
            self._conn.execute(
                "DELETE FROM conversations WHERE rowid IN ("
                "SELECT rowid FROM conversations ORDER BY created_at ASC LIMIT ?)",
                (to_remove,),
            )

    def close(self):
        self._conn.close()
