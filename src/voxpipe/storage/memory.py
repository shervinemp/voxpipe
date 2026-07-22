"""SQLite-backed conversation memory with FTS5 keyword search and session/global pool support."""

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
                    session_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conv_created
                    ON conversations(created_at);
                CREATE INDEX IF NOT EXISTS idx_conv_role
                    ON conversations(role);
                CREATE INDEX IF NOT EXISTS idx_conv_session
                    ON conversations(session_id);
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

    def store(self, content: str, session_id: str | None = None, **kwargs):
        role = kwargs.get("role", "user")
        meta = kwargs.get("meta") or {}
        if session_id and "session_id" not in meta:
            meta["session_id"] = session_id

        entry_id = hashlib.md5(
            f"{role}:{content}:{time.time()}".encode()
        ).hexdigest()[:16]
        meta_str = json.dumps(meta)
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO conversations "
                "(id, session_id, role, content, meta, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (entry_id, session_id, role, content, meta_str, time.time()),
            )
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO conv_fts (rowid, content) "
                    "VALUES (last_insert_rowid(), ?)", (content,),
                )
            except sqlite3.OperationalError:
                pass
            self._evict_old()

    def retrieve(self, query: str, top_k: int = 3, session_id: str | None = None,
                 include_global: bool = True, **kwargs) -> list[dict]:
        role = kwargs.get("role")
        keywords = [w for w in query.lower().split() if len(w) > 3]
        if not keywords:
            return []
        results = []
        seen = set()

        sess_clause = ""
        params_extra = []
        if session_id is not None:
            if include_global:
                sess_clause = " AND (c.session_id = ? OR c.session_id IS NULL) "
                params_extra.append(session_id)
            else:
                sess_clause = " AND c.session_id = ? "
                params_extra.append(session_id)
        else:
            if not include_global:
                sess_clause = " AND c.session_id IS NULL "

        try:
            for kw in keywords[:5]:
                sql_params = [f'"{kw}"']
                if role:
                    sql_params.append(role)
                    sql_params.extend(params_extra)
                    sql_params.append(top_k)
                    cur = self._conn.execute(
                        "SELECT c.content, c.role, c.created_at, c.session_id "
                        "FROM conversations c WHERE c.rowid IN ("
                        "SELECT rowid FROM conv_fts WHERE conv_fts MATCH ?)"
                        f"AND c.role = ? {sess_clause} ORDER BY c.created_at DESC LIMIT ?",
                        tuple(sql_params),
                    )
                else:
                    sql_params.extend(params_extra)
                    sql_params.append(top_k)
                    cur = self._conn.execute(
                        "SELECT c.content, c.role, c.created_at, c.session_id "
                        "FROM conversations c WHERE c.rowid IN ("
                        "SELECT rowid FROM conv_fts WHERE conv_fts MATCH ?)"
                        f"{sess_clause} ORDER BY c.created_at DESC LIMIT ?",
                        tuple(sql_params),
                    )
                for row in cur.fetchall():
                    if row[0] not in seen:
                        seen.add(row[0])
                        results.append({
                            "content": row[0],
                            "role": row[1],
                            "created_at": row[2],
                            "session_id": row[3],
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
