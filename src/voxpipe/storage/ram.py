"""Ephemeral in-memory storage engine implementing Storage protocol."""

from collections import defaultdict
import threading
import time
from typing import Any, Dict, List, Optional

from .protocols import QueryInput, Storage
from .record import Record


class RAMStorage(Storage):
    """In-memory (RAM-backed) storage engine for transient sessions and unit tests."""

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._banks: Dict[str, List[Record]] = defaultdict(list)
        self._lock = threading.Lock()

    def store(self, content: Any, bank: str = "global", **kwargs) -> Any:
        role = kwargs.get("role", "user")
        meta = kwargs.get("meta") or {}
        record = Record(
            content=content,
            source=f"ram:{bank}",
            score=1.0,
            meta={"role": role, "created_at": time.time(), **meta},
        )
        with self._lock:
            entries = self._banks[bank]
            entries.append(record)
            if len(entries) > self.max_entries:
                self._banks[bank] = entries[-self.max_entries:]
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
            query_str = (query.text or "").lower()
            top_k = query.top_k or top_k
        else:
            query_str = str(query).lower()

        keywords = [w for w in query_str.split() if len(w) > 2]
        if not keywords and query_str:
            keywords = [query_str]

        if isinstance(banks, str):
            target_banks = [banks]
        elif isinstance(banks, list):
            target_banks = banks
        else:
            target_banks = ["global"]

        role_filter = kwargs.get("role")
        results: List[Record] = []
        seen_content = set()

        with self._lock:
            for b in target_banks:
                for record in reversed(self._banks.get(b, [])):
                    rec_content_str = str(record).lower()
                    rec_role = record.meta.get("role")
                    if role_filter and rec_role != role_filter:
                        continue
                    if any(kw in rec_content_str for kw in keywords) or not keywords:
                        if record.content not in seen_content:
                            seen_content.add(record.content)
                            results.append(record)
                            if len(results) >= top_k:
                                return results
        return results
