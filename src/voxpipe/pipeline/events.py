from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, Callable, Dict, List, Tuple

from ..core.utils import get_logger


class EventEmitter:

    def __init__(self, max_workers: int = 4):
        self.logger = get_logger(__name__)
        self._handlers: Dict[str, List[Tuple[Callable, bool]]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="event"
        )

    def on(self, event: str, handler: Callable, *, async_: bool = False):
        if not callable(handler):
            raise TypeError("Event handler must be callable.")
        with self._lock:
            self._handlers.setdefault(event, []).append((handler, async_))

    def off(self, event: str, handler: Callable | None = None):
        with self._lock:
            if handler is None:
                self._handlers.pop(event, None)
            else:
                handlers = self._handlers.get(event, [])
                self._handlers[event] = [
                    (h, a) for h, a in handlers if h is not handler
                ]

    def emit(self, event: str, *args: Any, **kwargs: Any):
        with self._lock:
            handlers = list(self._handlers.get(event, []))

        for handler, async_ in handlers:
            try:
                if async_:
                    self._executor.submit(handler, *args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                self.logger.error(
                    "Event handler %s for %r failed: %s",
                    getattr(handler, "__name__", handler),
                    event,
                    e,
                )

    def close(self):
        self._executor.shutdown(wait=False)
