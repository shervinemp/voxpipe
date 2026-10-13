"""Per-session state: tool permissions and pending choices, user variables, provider caches."""

import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..core.utils import write_atomically
from .tools import ToolCall, ToolChoice

STATE_FILE = "state.json"

# Turn number of a pending choice that did not arise during a live turn (restored from disk, added by hand).
NO_TURN = -1


@dataclass
class PendingChoice:
    """A tool call waiting on the user's answer to the ToolChoice it produced."""

    call: ToolCall
    choice: ToolChoice
    turn: int = NO_TURN


class ToolStateView:
    """Remembered tool permissions and the choices awaiting an answer."""

    def __init__(self, lock: threading.RLock):
        self._lock = lock
        self._permissions: Dict[str, bool] = {}
        self._pending: Dict[str, PendingChoice] = {}

    def get_permission(self, tool_name: str) -> Optional[bool]:
        """True or False once the user has decided for good, None while they have not."""
        with self._lock:
            return self._permissions.get(tool_name)

    def set_permission(self, tool_name: str, allow: Optional[bool]) -> None:
        with self._lock:
            if allow is None:
                self._permissions.pop(tool_name, None)
            else:
                self._permissions[tool_name] = allow

    def revoke_permission(self, tool_name: str) -> None:
        self.set_permission(tool_name, None)

    def add_pending(self, call: ToolCall, choice: ToolChoice, *, turn: int = NO_TURN) -> PendingChoice:
        pending = PendingChoice(call=call, choice=choice, turn=turn)
        with self._lock:
            self._pending[choice.uid] = pending
        return pending

    def get_pending(self, uid: str) -> Optional[PendingChoice]:
        with self._lock:
            return self._pending.get(uid)

    def remove_pending(self, uid: str) -> Optional[PendingChoice]:
        with self._lock:
            return self._pending.pop(uid, None)

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def _clear(self) -> None:
        self._permissions.clear()
        self._pending.clear()

    def _dump(self) -> Dict[str, Any]:
        return {
            "permissions": dict(self._permissions),
            "pending": {
                uid: {
                    "call": {"id": p.call.id, "name": p.call.name, "arguments": p.call.arguments},
                    # Saved as the model saw it: a payload may hold values (dates, ...) JSON has no type for.
                    "choice": {"result": json.loads(p.choice.result_json), "speech": p.choice.speech},
                }
                for uid, p in self._pending.items()
            },
        }

    def _restore(self, data: Dict[str, Any]) -> None:
        self._permissions = {name: bool(allow) for name, allow in data.get("permissions", {}).items()}
        self._pending = {}
        for uid, entry in data.get("pending", {}).items():
            call, choice = entry["call"], entry["choice"]
            self._pending[uid] = PendingChoice(
                call=ToolCall(name=call["name"], arguments=call.get("arguments", {}), id=call["id"]),
                choice=ToolChoice(result=choice.get("result", {}), speech=choice.get("speech"), uid=uid),
            )


class ModelStateView:
    """Provider objects tied to this session, such as a local model's KV cache.

    They live only in this process and are never saved: a restored session
    simply has the model read its history again on the first turn.
    """

    def __init__(self):
        self.runtime: Dict[str, Any] = {}

    def _clear(self) -> None:
        self.runtime.clear()


class UserStateView:
    """Application variables attached to the session; values must be JSON-serialisable to be saved."""

    def __init__(self, lock: threading.RLock):
        self._lock = lock
        self._data: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def pop(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.pop(key, default)


class SessionState:
    """Thread-safe session state, namespaced as ``tools``, ``model`` and ``user``."""

    def __init__(self):
        self._lock = threading.RLock()
        self.tools = ToolStateView(self._lock)
        self.model = ModelStateView()
        self.user = UserStateView(self._lock)

    def atomic(self) -> threading.RLock:
        """Lock to hold (as a context manager) across a multi-step read-modify-write."""
        return self._lock

    def clear(self) -> None:
        with self._lock:
            self.tools._clear()
            self.model._clear()
            self.user._data.clear()

    def save(self, path: str) -> None:
        """Write ``state.json`` into the directory."""
        os.makedirs(path, exist_ok=True)
        with self._lock:
            payload = json.dumps({"tools": self.tools._dump(), "user": dict(self.user._data)}, indent=2)
        write_atomically(os.path.join(path, STATE_FILE), payload.encode("utf-8"))

    def load(self, path: str) -> None:
        """Replace this state with what :meth:`save` wrote into the directory."""
        state_file = os.path.join(path, STATE_FILE)
        data: Dict[str, Any] = {}
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

        with self._lock:
            self.tools._restore(data.get("tools", {}))
            self.user._data = dict(data.get("user", {}))
            self.model._clear()

