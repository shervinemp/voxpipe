"""Conversation history: messages, the visible context window, and provider wire formats."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from .tools import Tool, ToolCall


@dataclass
class Message:
    """One turn of a conversation.

    ``content`` is what was said.  ``context`` is a preamble shown to the model
    ahead of it (recalled memories, annotations) that is not part of what was
    said, so it never reaches long-term memory.  Assistant turns that request
    tools carry ``tool_calls``; the tool turns answering them carry the matching
    ``tool_call_id`` and tool ``name``.  A tool turn with neither is a note to
    the model rather than a result.
    """

    class Role(Enum):
        system = "system"
        user = "user"
        assistant = "assistant"
        tool = "tool"

    role: Role
    content: str
    context: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    archived: bool = False

    @property
    def rendered(self) -> str:
        """The text the model sees for this turn."""
        return f"{self.context}\n{self.content}" if self.context else self.content

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for persistence, omitting fields left at their defaults."""
        data: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.context:
            data["context"] = self.context
        if self.tool_calls:
            data["tool_calls"] = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        if self.archived:
            data["archived"] = True
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=cls.Role(data["role"]),
            content=data.get("content") or "",
            context=data.get("context"),
            tool_calls=[
                ToolCall(name=c["name"], arguments=c.get("arguments", {}), id=c["id"])
                for c in data.get("tool_calls", [])
            ],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            archived=bool(data.get("archived", False)),
        )


class Conversation:
    """An ordered message history with a system prompt, a tool set and a visible window.

    Messages before the cutoff stay in the history but are hidden from the
    model; :meth:`hide_oldest` advances the cutoff and :meth:`drop_hidden`
    removes what lies behind it.
    """

    def __init__(self):
        self._messages: List[Message] = []
        self._system: str = ""
        self._tools: Dict[str, Tool] = {}
        self._cutoff_idx: int = 0

    # --- building --------------------------------------------------------

    def set_system_message(self, content: str) -> None:
        self._system = content

    def add_user_message(self, content: str, *, context: Optional[str] = None) -> Message:
        return self._append(Message(role=Message.Role.user, content=content, context=context))

    def add_assistant_message(self, content: str, *, tool_calls: Iterable[ToolCall] = ()) -> Message:
        return self._append(Message(role=Message.Role.assistant, content=content, tool_calls=list(tool_calls)))

    def add_tool_message(self, content: str, *, tool_call: Optional[ToolCall] = None) -> Message:
        """Add a tool's answer to ``tool_call``, or, without one, a note to the model."""
        return self._append(
            Message(
                role=Message.Role.tool,
                content=content,
                tool_call_id=tool_call.id if tool_call else None,
                name=tool_call.name if tool_call else None,
            )
        )

    def _append(self, message: Message) -> Message:
        self._messages.append(message)
        return message

    def discard(self, messages: Iterable[Message]) -> None:
        """Remove the given messages (matched by identity) from the history."""
        doomed = {id(m) for m in messages}
        hidden = sum(1 for m in self._messages[: self._cutoff_idx] if id(m) in doomed)
        self._messages = [m for m in self._messages if id(m) not in doomed]
        self._cutoff_idx -= hidden

    def clear(self) -> None:
        """Forget every message; the system prompt and tools stay."""
        self._messages.clear()
        self._cutoff_idx = 0

    # --- accessors -------------------------------------------------------

    @property
    def system_message(self) -> str:
        return self._system

    @property
    def tools(self) -> Dict[str, Tool]:
        return self._tools

    @tools.setter
    def tools(self, tools: Dict[str, Tool] | Iterable[Tool]) -> None:
        self._tools = dict(tools) if isinstance(tools, dict) else {t.name: t for t in tools}

    @property
    def all_messages(self) -> List[Message]:
        """Every message still held, hidden ones included."""
        return list(self._messages)

    @property
    def visible(self) -> List[Message]:
        """The messages the model currently sees, oldest first."""
        return self._messages[self._cutoff_idx :]

    @property
    def last_message(self) -> Optional[Message]:
        return self._messages[-1] if self._messages else None

    # --- wire formats ----------------------------------------------------

    @property
    def messages(self) -> List[Dict[str, str]]:
        """The visible window as plain role/content turns, for local chat templates.

        Tool results read ``"<tool>: <payload>"``; an assistant turn that only
        requested tools has nothing to show in this format and is left out.
        """
        rendered = [{"role": "system", "content": self._system}] if self._system else []
        for m in self.visible:
            if m.role is Message.Role.assistant and m.tool_calls and not m.content:
                continue
            content = f"{m.name}: {m.content}" if m.role is Message.Role.tool and m.name else m.rendered
            rendered.append({"role": m.role.value, "content": content})
        return rendered

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """The visible window in OpenAI chat-completions format, with tool calls paired to results.

        Notes to the model (tool turns answering no call) are folded into the
        tool result they follow, or sent as a user turn when there is none, since
        the format admits no free-standing tool message.
        """
        rendered: List[Dict[str, Any]] = [{"role": "system", "content": self._system}] if self._system else []
        for m in self.visible:
            if m.role is Message.Role.tool and not m.tool_call_id:
                if rendered and rendered[-1]["role"] == "tool":
                    rendered[-1]["content"] += f"\n\n{m.content}"
                else:
                    rendered.append({"role": "user", "content": m.content})
            elif m.role is Message.Role.tool:
                rendered.append({"role": "tool", "tool_call_id": m.tool_call_id, "name": m.name, "content": m.content})
            elif m.tool_calls:
                rendered.append(
                    {
                        "role": "assistant",
                        "content": m.content,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                            }
                            for c in m.tool_calls
                        ],
                    }
                )
            else:
                rendered.append({"role": m.role.value, "content": m.rendered})
        return rendered

    # --- context window --------------------------------------------------

    @property
    def cutoff_idx(self) -> int:
        """Index of the first visible message."""
        return self._cutoff_idx

    @cutoff_idx.setter
    def cutoff_idx(self, value: int) -> None:
        self._cutoff_idx = max(0, min(int(value), len(self._messages)))

    def visible_count(self) -> int:
        return len(self._messages) - self._cutoff_idx

    def hide_oldest(self, count: int) -> int:
        """Hide the ``count`` oldest visible messages, rounded to a turn boundary.

        The window must open on a user turn: starting mid-exchange would strand
        tool results from the calls they answer.  So the cut moves forward to
        the next user message, or, when none follows, back to the one opening
        the current exchange.  Returns how many messages were hidden.
        """
        if count <= 0:
            return 0
        start = self._cutoff_idx
        target = min(start + count, len(self._messages))

        def is_user(index: int) -> bool:
            return self._messages[index].role is Message.Role.user

        forward = next((i for i in range(target, len(self._messages)) if is_user(i)), None)
        if forward is not None:
            target = forward
        else:
            target = next((i for i in range(target - 1, start - 1, -1) if is_user(i)), start)

        self._cutoff_idx = target
        return target - start

    def drop_hidden(self) -> List[Message]:
        """Remove and return the messages behind the cutoff."""
        dropped, self._messages = self._messages[: self._cutoff_idx], self._messages[self._cutoff_idx :]
        self._cutoff_idx = 0
        return dropped

    # --- persistence -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": self._system,
            "messages": [m.to_dict() for m in self._messages],
            "cutoff_idx": self._cutoff_idx,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        conv = cls()
        conv._system = data.get("system", "")
        conv._messages = [Message.from_dict(m) for m in data.get("messages", [])]
        conv.cutoff_idx = data.get("cutoff_idx", 0)
        return conv

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Conversation":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
