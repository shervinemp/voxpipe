"""Buffer storage and lifecycle.

Inferred side buffers are disposable: they expire after the store's TTL and are
capped by ``max_side``.  Named buffers are explicit workflow state: they are
never selected by similarity and, unless given their own TTL, never expire.
That separation is what lets a caller keep a table/order buffer without
background chatter evicting it.
"""

from typing import Dict, List, Optional

from .models import TopicBuffer


class BufferStore:
    def __init__(self, *, max_side: int = 4, ttl_sentences: int = 10):
        self.max_side = max_side
        self.ttl_sentences = ttl_sentences
        self.index = 0
        self._next_id = 1
        self.main = TopicBuffer(id=0, name="main")
        self._side: List[TopicBuffer] = []
        self._named: Dict[str, TopicBuffer] = {}

    @property
    def side(self) -> List[TopicBuffer]:
        """Inferred side buffers, oldest first (a copy)."""
        return list(self._side)

    @property
    def named(self) -> List[TopicBuffer]:
        """Explicit workflow buffers (a copy)."""
        return list(self._named.values())

    def tick(self) -> int:
        """Advance the clock and expire anything due."""
        self.index += 1
        self.expire()
        return self.index

    def expire(self) -> None:
        self._side = [buffer for buffer in self._side if not buffer.is_expired(self.index, self.ttl_sentences)]
        for name, buffer in list(self._named.items()):
            if buffer.is_expired(self.index, None):
                del self._named[name]

    def add_side(self) -> TopicBuffer:
        buffer = TopicBuffer(id=self._next_id)
        self._next_id += 1
        self._side.append(buffer)
        return buffer

    def add_named(
        self,
        name: str,
        *,
        ttl_sentences: Optional[int] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> TopicBuffer:
        if not name or name == "main":
            return self.main
        buffer = self._named.get(name)
        if buffer is None:
            buffer = TopicBuffer(id=self._next_id, name=name, ttl_sentences=ttl_sentences)
            self._next_id += 1
            self._named[name] = buffer
        elif ttl_sentences is not None:
            buffer.ttl_sentences = ttl_sentences
        if metadata:
            buffer.metadata.update(metadata)
        return buffer

    def find(self, name: str) -> Optional[TopicBuffer]:
        if not name or name == "main":
            return self.main
        return self._named.get(name)

    def trim(self) -> None:
        while len(self._side) > self.max_side:
            victim = min(self._side, key=lambda buffer: (buffer.last_seen, buffer.id))
            self._side.remove(victim)

    def promote(self, buffer: TopicBuffer) -> TopicBuffer:
        """Make an inferred side buffer the main one; return the old main."""
        self._side.remove(buffer)
        old_main = self.main
        old_main.name = None
        old_main.mark_seen(self.index)
        self._side.append(old_main)
        self.trim()
        self.main = buffer
        buffer.name = "main"
        buffer.streak = 0
        return old_main

    def remove_side(self, buffer: TopicBuffer) -> None:
        if buffer in self._side:
            self._side.remove(buffer)

    def remove_named(self, name: str) -> None:
        self._named.pop(name, None)

    def reset(self) -> None:
        self.index = 0
        self._next_id = 1
        self.main = TopicBuffer(id=0, name="main")
        self._side = []
        self._named = {}
