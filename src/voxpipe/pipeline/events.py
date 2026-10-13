"""A small synchronous/asynchronous event emitter for pipeline notifications."""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Tuple

from ..core.utils import get_logger


class EventEmitter:
    """Dispatches named events to handlers; a failing handler is logged, never propagated.

    Handlers registered with ``async_=True`` run on a small thread pool, so a
    slow listener cannot hold up the pipeline that emitted the event.
    """

    def __init__(self, max_workers: int = 4):
        self.logger = get_logger(__name__)
        self._handlers: Dict[str, List[Tuple[Callable, bool]]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="event")
        self._closed = False

    def on(self, event: str, handler: Callable, *, async_: bool = False) -> None:
        if not callable(handler):
            raise TypeError("Event handler must be callable.")
        with self._lock:
            self._handlers.setdefault(event, []).append((handler, async_))

    def off(self, event: str, handler: Callable | None = None) -> None:
        """Remove ``handler`` from an event, or every handler of the event when none is given."""
        with self._lock:
            if handler is None:
                self._handlers.pop(event, None)
            else:
                self._handlers[event] = [(h, a) for h, a in self._handlers.get(event, []) if h is not handler]

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            if self._closed:
                return
            handlers = list(self._handlers.get(event, []))

        for handler, async_ in handlers:
            if async_:
                self._executor.submit(self._call, event, handler, args, kwargs)
            else:
                self._call(event, handler, args, kwargs)

    def _call(self, event: str, handler: Callable, args: tuple, kwargs: dict) -> None:
        try:
            handler(*args, **kwargs)
        except Exception:  # noqa: BLE001 - listeners are third-party code; isolate their failures
            self.logger.error(
                "Event handler %s for %r failed.", getattr(handler, "__name__", handler), event, exc_info=True
            )

    def close(self) -> None:
        """Stop dispatching; handlers already running are left to finish."""
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False)
