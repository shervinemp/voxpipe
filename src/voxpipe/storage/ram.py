"""Ephemeral in-memory storage engine implementing the Storage protocol."""

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union

from .protocols import QueryInput, keywords, search_banks, store_bank, unpack_query
from .record import Record


class RAMStorage:
    """RAM-backed storage for transient sessions and tests.

    Matches a record when any query keyword occurs in its text, ranking by how
    many keywords matched and then by recency.
    """

    def __init__(self, max_entries: int = 1000):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1.")
        self.max_entries = max_entries
        self._entries: Deque[Tuple[str, Record]] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def store(
        self,
        content: Any,
        *,
        bank: Optional[str] = None,
        role: str = "user",
        meta: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Record:
        bank = store_bank(bank, session_id)
        record = Record(
            content=content,
            source=f"ram:{bank}",
            meta={**(meta or {}), "role": role, "created_at": time.time(), "bank": bank},
        )
        with self._lock:
            self._entries.append((bank, record))
        return record

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
        text, top_k = unpack_query(query, top_k)
        terms = keywords(text)
        if not terms or top_k <= 0:
            return []
        targets = set(search_banks(banks, session_id, include_global))

        with self._lock:
            candidates = [record for bank, record in self._entries if bank in targets]

        scored: Dict[str, Tuple[float, int, Record]] = {}
        for age, record in enumerate(reversed(candidates)):
            if role and record.meta.get("role") != role:
                continue
            haystack = record.text.lower()
            hits = sum(term in haystack for term in terms)
            # Newest copy of a repeated text wins: it is visited first.
            if hits and record.text not in scored:
                scored[record.text] = (hits / len(terms), -age, record)

        ranked = sorted(scored.values(), key=lambda item: item[:2], reverse=True)[:top_k]
        return [
            Record(content=record.content, source=record.source, score=score, meta=dict(record.meta))
            for score, _, record in ranked
        ]
