"""SessionState manager providing namespaced, thread-safe, auto-saving session state."""

from dataclasses import dataclass, field
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class ToolStateView:
    """Namespaced view for managing tool permissions, pending choices, and tool metadata."""

    def __init__(self, parent: "SessionState"):
        self._parent = parent

    def get_permission(self, tool_name: str) -> bool | None:
        with self._parent.atomic():
            return self._parent._tools_data["permissions"].get(tool_name)

    def set_permission(self, tool_name: str, allow: bool | None) -> None:
        with self._parent.atomic():
            perms = self._parent._tools_data["permissions"]
            if allow is None:
                perms.pop(tool_name, None)
            else:
                perms[tool_name] = allow

    def revoke_permission(self, tool_name: str) -> None:
        self.set_permission(tool_name, None)

    def add_pending(self, uid: str, call_obj: Any, choice_obj: Any) -> None:
        with self._parent.atomic():
            self._parent._tools_data["pending"][uid] = (call_obj, choice_obj)

    def remove_pending(self, uid: str) -> Optional[Tuple[Any, Any]]:
        with self._parent.atomic():
            return self._parent._tools_data["pending"].pop(uid, None)

    def get_pending(self, uid: str) -> Optional[Tuple[Any, Any]]:
        with self._parent.atomic():
            return self._parent._tools_data["pending"].get(uid)

    def get_all_pending(self) -> Dict[str, Tuple[Any, Any]]:
        with self._parent.atomic():
            return dict(self._parent._tools_data["pending"])


class ModelStateView:
    """Namespaced view for managing LLM provider model state and KV cache buffers."""

    def __init__(self, parent: "SessionState"):
        self._parent = parent

    def set_kv_cache(self, raw_bytes: bytes) -> None:
        with self._parent.atomic():
            self._parent._binary_artifacts["kv_cache.bin"] = raw_bytes

    def get_kv_cache(self) -> Optional[bytes]:
        with self._parent.atomic():
            return self._parent._binary_artifacts.get("kv_cache.bin")


class UserStateView:
    """Namespaced view for managing user session properties and custom variables."""

    def __init__(self, parent: "SessionState"):
        self._parent = parent

    def get(self, key: str, default: Any = None) -> Any:
        with self._parent.atomic():
            return self._parent._user_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._parent.atomic():
            self._parent._user_data[key] = value

    def pop(self, key: str, default: Any = None) -> Any:
        with self._parent.atomic():
            return self._parent._user_data.pop(key, default)


class SessionState:
    """Thread-safe, namespaced session state manager with automatic storage routing."""

    def __init__(self):
        self._lock = threading.RLock()
        self._tools_data: Dict[str, Any] = {"permissions": {}, "pending": {}}
        self._user_data: Dict[str, Any] = {}
        self._model_data: Dict[str, Any] = {}
        self._custom_data: Dict[str, Any] = {}
        self._binary_artifacts: Dict[str, bytes] = {}

        self.tools = ToolStateView(self)
        self.model = ModelStateView(self)
        self.user = UserStateView(self)

    def atomic(self) -> threading.RLock:
        """Context manager lock for multi-field atomic mutations."""
        return self._lock

    @property
    def _data(self) -> Dict[str, Any]:
        """Backward compatibility dictionary accessor."""
        with self._lock:
            res = dict(self._custom_data)
            res["tools"] = self._tools_data
            res["user"] = self._user_data
            res["model"] = self._model_data
            return res

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            if key in self._custom_data:
                return self._custom_data[key]
            if key == "tools":
                return self._tools_data
            if key == "user":
                return self._user_data
            if key == "model":
                return self._model_data
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            if key == "model_state" or isinstance(value, (bytes, bytearray)):
                b_val = bytes(value) if isinstance(value, (bytes, bytearray)) else value
                self._binary_artifacts["kv_cache.bin"] = b_val
                self._binary_artifacts["model_state"] = b_val
            else:
                self._custom_data[key] = value

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._custom_data or key in ("tools", "user", "model")

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            try:
                return self[key]
            except KeyError:
                return default

    def clear(self) -> None:
        with self._lock:
            self._tools_data["permissions"].clear()
            self._tools_data["pending"].clear()
            self._user_data.clear()
            self._model_data.clear()
            self._custom_data.clear()
            self._binary_artifacts.clear()

    def save(self, path: str, save_binary: bool = False) -> None:
        """Persist state cleanly: serializes JSON data to state.json and binary to artifacts."""
        os.makedirs(path, exist_ok=True)
        with self._lock:
            serializable_state = dict(self._custom_data)
            serializable_state["tools"] = {
                "permissions": dict(self._tools_data["permissions"]),
                "pending": {
                    uid: (
                        {"name": call.name, "arguments": call.arguments},
                        {"result": choice.raw_result, "speech": choice.speech, "uid": choice.uid},
                    )
                    for uid, (call, choice) in self._tools_data["pending"].items()
                },
            }
            serializable_state["user"] = dict(self._user_data)
            serializable_state["model"] = dict(self._model_data)
            serializable_state["custom"] = dict(self._custom_data)

            with open(os.path.join(path, "state.json"), "w") as f:
                json.dump(serializable_state, f, indent=2)

            if save_binary:
                for fname, data in self._binary_artifacts.items():
                    if isinstance(data, (bytes, bytearray)):
                        with open(os.path.join(path, fname), "wb") as f:
                            f.write(data)

    def load(self, path: str) -> None:
        """Load state from directory."""
        from .tools import ToolCall, ToolChoice

        state_file = os.path.join(path, "state.json")
        kv_file = os.path.join(path, "kv_cache.bin")

        with self._lock:
            if os.path.exists(state_file):
                with open(state_file) as f:
                    data = json.load(f)

                if "tools" in data:
                    self._tools_data["permissions"] = data["tools"].get("permissions", {})
                    raw_pending = data["tools"].get("pending", {})
                    reconstructed_pending = {}
                    for uid, (c_dict, ch_dict) in raw_pending.items():
                        c_obj = ToolCall(name=c_dict["name"], arguments=c_dict.get("arguments", {}))
                        ch_obj = ToolChoice(
                            result=ch_dict.get("result", {}),
                            speech=ch_dict.get("speech"),
                            uid=ch_dict.get("uid"),
                        )
                        reconstructed_pending[uid] = (c_obj, ch_obj)
                    self._tools_data["pending"] = reconstructed_pending

                if "user" in data:
                    self._user_data = data["user"]
                if "model" in data:
                    self._model_data = data["model"]
                if "custom" in data:
                    self._custom_data.update(data["custom"])

                for k, v in data.items():
                    if k not in ("tools", "user", "model", "custom"):
                        self._custom_data[k] = v

            if os.path.exists(kv_file):
                with open(kv_file, "rb") as f:
                    b_val = f.read()
                    self._binary_artifacts["kv_cache.bin"] = b_val
                    self._binary_artifacts["model_state"] = b_val


class SessionStateDictProxy(dict):
    """Transparent dict proxy for backward-compatible _session_state dict access."""

    def __init__(self, state: SessionState):
        super().__init__()
        self._state = state

    def __getitem__(self, key: str) -> Any:
        return self._state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._state[key] = value

    def __delitem__(self, key: str) -> None:
        self._state._custom_data.pop(key, None)

    def __contains__(self, key: object) -> bool:
        return key in self._state

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def clear(self) -> None:
        self._state.clear()

    def update(self, *args, **kwargs) -> None:
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def items(self):
        return self._state._data.items()

    def keys(self):
        return self._state._data.keys()

    def values(self):
        return self._state._data.values()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            if not other:
                return (
                    not self._state._custom_data
                    and not self._state._tools_data["permissions"]
                    and not self._state._tools_data["pending"]
                    and not self._state._user_data
                    and not self._state._model_data
                )
            return dict(self.items()) == other
        return super().__eq__(other)

    def __repr__(self) -> str:
        return repr(dict(self.items()))
