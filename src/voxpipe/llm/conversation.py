import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Dict, Optional, Tuple
from .tools import Tool, ToolCall, ToolChoice

if TYPE_CHECKING:
    from .model import LLM


@dataclass
class Message:

    class Role(Enum):
        system = "system"
        user = "user"
        assistant = "assistant"
        tool = "tool"

    role: Role
    content: str

    def asdict(self):
        return {"role": self.role.value, "content": self.content}

    @staticmethod
    def from_dict(data: Dict[str, str]):
        return Message(Message.Role(data["role"]), data["content"])


class MessageList(list):
    def __init__(self, other: Iterable | None = None):
        super().__init__()
        if other:
            for item in other:
                self.append(item)

    def append(self, x: Message | Dict[str, str]):
        if isinstance(x, Message):
            super().append(x)
        elif isinstance(x, dict):
            super().append(Message.from_dict(x))
        else:
            raise NotImplementedError

    def __getitem__(
        self, key: int | slice
    ) -> Dict[str, str] | List[Dict[str, str]]:
        if isinstance(key, slice):
            return MessageList(super().__getitem__(key))
        else:
            return Message.asdict(super().__getitem__(key))

    def __setitem__(self, key: int | slice, value: Any):
        if isinstance(key, slice):
            converted = [
                v if isinstance(v, Message) else Message.from_dict(v)
                for v in value
            ]
            super().__setitem__(key, converted)
        else:
            v = (
                value
                if isinstance(value, Message)
                else Message.from_dict(value)
            )
            super().__setitem__(key, v)

    def __iter__(self) -> Iterable[Dict[str, str]]:
        return map(Message.asdict, super().__iter__())

    def __add__(self, other):
        if isinstance(other, list):
            return MessageList(super().__add__(other))
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, list):
            return MessageList(other.__add__(self))
        return NotImplemented


class Conversation:

    def __init__(self):
        self._messages: MessageList = MessageList()
        self._token_counts: List[int] = []

        self._system: str = ""
        self._tools: Dict[str, Tool] = {}
        self._cutoff_idx: int = 0
        self._meta: Dict[str, Dict[str, Any]] = {}

    def get_meta(self, tool_name: str) -> Dict[str, Any]:
        return self._meta.setdefault(tool_name, {})

    def set_meta(self, tool_name: str, key: str, value: Any):
        self.get_meta(tool_name)[key] = value

    def clear_meta(self):
        self._meta.clear()

    def get_permission(self, tool_name: str) -> bool | None:
        return self.get_meta(tool_name).get("_permission")

    def set_permission(self, tool_name: str, allow: bool | None):
        if allow is None:
            self.get_meta(tool_name).pop("_permission", None)
        else:
            self.set_meta(tool_name, "_permission", allow)

    def revoke_permission(self, tool_name: str):
        self.set_permission(tool_name, None)

    def set_system_message(self, content: str):
        self._system = content

    def add_user_message(self, content: str):
        msg = Message(role=Message.Role.user, content=content)
        self._messages.append(msg)
        self._token_counts.append(0)

    def add_assistant_message(self, content: str):
        msg = Message(role=Message.Role.assistant, content=content)
        self._messages.append(msg)
        self._token_counts.append(0)

    def add_tool_message(self, content: str):
        msg = Message(role=Message.Role.tool, content=content)
        self._messages.append(msg)
        self._token_counts.append(0)

    def clear(self):
        self._messages.clear()
        self._token_counts.clear()
        self._meta.clear()

    @property
    def messages(self) -> List[Dict[str, str]]:
        msgs = []
        if self._system:
            msgs.append({"role": "system", "content": self._system})
        for m in self._messages[self._cutoff_idx :]:
            msgs.append(m.asdict() if isinstance(m, Message) else m)
        return msgs

    @property
    def system(self) -> Message:
        return Message(role=Message.Role.system, content=self._system)

    @property
    def tools(self) -> Dict[str, Tool]:
        return self._tools

    @tools.setter
    def tools(self, tools: Dict[str, Tool] | Iterable[Tool]):
        if not isinstance(tools, dict):
            tools = {t.name: t for t in tools}
        self._tools = tools

    def to_dict(self) -> Dict:
        serialized_meta = {}
        for t_name, meta_dict in self._meta.items():
            s_meta = {}
            for k, v in meta_dict.items():
                if k == "pending" and isinstance(v, dict):
                    s_pending = {}
                    for uid, (call, choice) in v.items():
                        s_pending[uid] = (
                            {"name": call.name, "arguments": call.arguments},
                            {"result": choice.raw_result, "speech": choice.speech, "uid": choice.uid},
                        )
                    s_meta["pending"] = s_pending
                else:
                    s_meta[k] = v
            serialized_meta[t_name] = s_meta

        return {
            "system": self._system,
            "messages": [m.asdict() for m in list.__iter__(self._messages)],
            "cutoff_idx": self._cutoff_idx,
            "_meta": serialized_meta,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Conversation":
        conv = cls()
        conv._system = data.get("system", "")
        for msg_data in data.get("messages", []):
            conv._messages.append(Message.from_dict(msg_data))
            conv._token_counts.append(0)
        conv._cutoff_idx = data.get("cutoff_idx", 0)

        raw_meta = data.get("_meta", {})
        reconstructed_meta = {}
        for t_name, meta_dict in raw_meta.items():
            r_meta = {}
            for k, v in meta_dict.items():
                if k == "pending" and isinstance(v, dict):
                    r_pending = {}
                    for uid, (call_dict, choice_dict) in v.items():
                        call_obj = ToolCall(name=call_dict["name"], arguments=call_dict.get("arguments", {}))
                        choice_obj = ToolChoice(result=choice_dict["result"], speech=choice_dict.get("speech"), uid=choice_dict.get("uid"))
                        r_pending[uid] = (call_obj, choice_obj)
                    r_meta["pending"] = r_pending
                else:
                    r_meta[k] = v
            reconstructed_meta[t_name] = r_meta

        conv._meta = reconstructed_meta
        return conv

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Conversation":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @property
    def cutoff_idx(self):
        if not self._messages or self._cutoff_idx >= len(self._messages):
            return len(self._messages)
        return range(len(self._messages))[self._cutoff_idx]

    @cutoff_idx.setter
    def cutoff_idx(self, value):
        self._cutoff_idx = value

    def visible_count(self) -> int:
        return len(self._messages) - self._cutoff_idx

    def _get_raw_message(self, idx: int) -> Message:
        """Access internal Message object bypassing MessageList's dict-returning __getitem__."""
        return list.__getitem__(self._messages, idx)

    def get_message_content(self, idx: int) -> str:
        """Get the content string of a raw message by internal index."""
        return self._get_raw_message(idx).content

    def get_token_count(self, idx: int) -> int:
        return self._token_counts[idx]

    def set_token_count(self, idx: int, count: int):
        self._token_counts[idx] = count

    def trim_oldest(self, excess: int, max_turns: int, llm: "LLM") -> int:
        """Advance cutoff forward by *excess* messages. O(1) -- never deletes.

        A rare :meth:`compact` pass recreates the list when hidden messages
        exceed ``max_turns * 4``, allowing the GC to free the old storage.
        """
        visible = len(self._messages) - self._cutoff_idx
        excess = min(excess, visible)
        if excess <= 0:
            return 0
        total_cut = 0
        for i in range(self._cutoff_idx, self._cutoff_idx + excess):
            if self._token_counts[i] == 0:
                self._token_counts[i] = llm.count_tokens(
                    self.get_message_content(i)
                ) + 4
            total_cut += self._token_counts[i]

        self._cutoff_idx += excess

        if self._cutoff_idx > max_turns * 4:
            self.compact()

        return total_cut

    def compact(self):
        """Create new message list from visible portion. Old list to GC."""
        if self._cutoff_idx <= 0:
            return
        self._messages = MessageList(self._messages[self._cutoff_idx:])
        self._token_counts = self._token_counts[self._cutoff_idx:]
        self._cutoff_idx = 0
