"""SQLite-backed conversation memory with FTS5 keyword search."""

import hashlib
import json
import os
import sqlite3
import threading
import time

from voxpipe.core.utils import get_logger
from voxpipe.storage.retriever import Retriever


class Memory(Retriever):
    def __init__(self, db_path: str, max_entries: int = 1000,
                 ttl_days: int = 30):
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
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conv_created
                    ON conversations(created_at);
                CREATE INDEX IF NOT EXISTS idx_conv_role
                    ON conversations(role);
            """)
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

    def store(self, content: str, **kwargs):
        role = kwargs.get("role", "user")
        meta = kwargs.get("meta")
        entry_id = hashlib.md5(
            f"{role}:{content}:{time.time()}".encode()
        ).hexdigest()[:16]
        meta_str = json.dumps(meta or {})
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO conversations "
                "(id, role, content, meta, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (entry_id, role, content, meta_str, time.time()),
            )
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO conv_fts (rowid, content) "
                    "VALUES (last_insert_rowid(), ?)", (content,),
                )
            except sqlite3.OperationalError:
                pass
            self._evict_old()

    def retrieve(self, query: str, top_k: int = 3,
                 **kwargs) -> list[dict]:
        role = kwargs.get("role")
        keywords = [w for w in query.lower().split() if len(w) > 3]
        if not keywords:
            return []
        results = []
        seen = set()
        try:
            for kw in keywords[:5]:
                if role:
                    cur = self._conn.execute(
                        "SELECT c.content, c.role, c.created_at "
                        "FROM conversations c WHERE c.rowid IN ("
                        "SELECT rowid FROM conv_fts WHERE conv_fts MATCH ?)"
                        "AND c.role = ? ORDER BY c.created_at DESC LIMIT ?",
                        (f'"{kw}"', role, top_k),
                    )
                else:
                    cur = self._conn.execute(
                        "SELECT c.content, c.role, c.created_at "
                        "FROM conversations c WHERE c.rowid IN ("
                        "SELECT rowid FROM conv_fts WHERE conv_fts MATCH ?)"
                        "ORDER BY c.created_at DESC LIMIT ?",
                        (f'"{kw}"', top_k),
                    )
                for row in cur.fetchall():
                    if row[0] not in seen:
                        seen.add(row[0])
                        results.append({
                            "content": row[0], "role": row[1],
                            "created_at": row[2],
                        })
                        if len(results) >= top_k:
                            return results
            return results
        except sqlite3.OperationalError:
            return []

    def _evict_old(self):
        cutoff = time.time() - (self.ttl_seconds if self.ttl_seconds > 0 else 1.0)
        self._conn.execute(
            "DELETE FROM conversations WHERE created_at < ?", (cutoff,))
        count = self._conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]
        if count > self.max_entries:
            to_remove = count - self.max_entries
            self._conn.execute(
                "DELETE FROM conversations WHERE rowid IN ("
                "SELECT rowid FROM conversations "
                "ORDER BY created_at ASC LIMIT ?)", (to_remove,),
            )

    def close(self):
        self._conn.close()
