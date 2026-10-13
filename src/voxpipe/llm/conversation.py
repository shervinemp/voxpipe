"""The model's context: a system prompt, a tool set and the message history, in provider wire formats."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..core.utils import write_atomically
from .tools import Tool, ToolCall, ToolSet


@dataclass
class Message:
    """One message of a conversation.

    ``content`` is what was said.  ``context`` is a preamble shown to the model
    ahead of it (recalled memories, annotations) that is not part of what was
    said, so it never reaches long-term memory.  Assistant messages that request
    tools carry ``tool_calls``; the tool messages answering them carry the
    matching ``tool_call_id`` and tool ``name``.  A tool message with neither is
    a note to the model rather than a result.
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
        """The text the model sees for this message."""
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
    """Everything the model is shown: a system prompt, the tools it may call, and the messages so far.

    The system prompt (:attr:`system_prompt`) is the system message followed
    by a list of rules: each tool's ``instruction``, then the standing
    :attr:`rules`.  A *turn* is a user message together with everything said
    in answer to it; history is only ever cut between turns.
    """

    def __init__(self):
        self._messages: List[Message] = []
        self._system: str = ""
        self._tools: ToolSet = ToolSet()
        self.rules: List[str] = []

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
        self._messages = [m for m in self._messages if id(m) not in doomed]

    def evict_oldest(self, count: int) -> List[Message]:
        """Remove and return the ``count`` oldest messages, rounded to a turn boundary.

        What remains must open on a user message: starting mid-turn would
        strand tool results from the calls they answer.  So the cut moves
        forward to the next user message, or, when none follows, back to the
        one opening the current turn.
        """
        if count <= 0:
            return []
        target = min(count, len(self._messages))

        def is_user(index: int) -> bool:
            return self._messages[index].role is Message.Role.user

        forward = next((i for i in range(target, len(self._messages)) if is_user(i)), None)
        if forward is not None:
            target = forward
        else:
            target = next((i for i in range(target - 1, -1, -1) if is_user(i)), 0)

        evicted, self._messages = self._messages[:target], self._messages[target:]
        return evicted

    def clear(self) -> None:
        """Forget every message; the system prompt, rules and tools stay."""
        self._messages.clear()

    def blank_copy(self) -> "Conversation":
        """A conversation with the same system prompt, rules and tools, and no messages."""
        copy = Conversation()
        copy._system = self._system
        copy.rules = list(self.rules)
        copy.tools = self._tools
        return copy

    # --- accessors -------------------------------------------------------

    @property
    def system_message(self) -> str:
        """The opening text of the system prompt, as set by :meth:`set_system_message`."""
        return self._system

    @property
    def system_prompt(self) -> str:
        """The system prompt as the model receives it: the system message, then the rules."""
        rules = [" ".join(tool.instruction.split()) for tool in self._tools.values() if tool.instruction]
        rules += self.rules
        sections = [self._system] if self._system else []
        if rules:
            sections.append("\n".join(["Rules:", *(f"- {rule}" for rule in rules)]))
        return "\n\n".join(sections)

    @property
    def tools(self) -> ToolSet:
        return self._tools

    @tools.setter
    def tools(self, tools: Mapping[str, Tool] | Iterable[Tool]) -> None:
        self._tools = ToolSet(tools)

    @property
    def history(self) -> List[Message]:
        """The messages so far, oldest first."""
        return list(self._messages)

    @property
    def last_message(self) -> Optional[Message]:
        return self._messages[-1] if self._messages else None

    # --- wire formats ----------------------------------------------------

    @property
    def messages(self) -> List[Dict[str, str]]:
        """The context as plain role/content messages, for local chat templates.

        Tool results read ``"<tool>: <payload>"``; an assistant message that
        only requested tools has nothing to show in this format and is left out.
        """
        rendered = self._system_messages()
        for m in self._messages:
            if m.role is Message.Role.assistant and m.tool_calls and not m.content:
                continue
            content = f"{m.name}: {m.content}" if m.role is Message.Role.tool and m.name else m.rendered
            rendered.append({"role": m.role.value, "content": content})
        return rendered

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """The context in OpenAI chat-completions format, with tool calls paired to results.

        Notes to the model (tool messages answering no call) are folded into
        the tool result they follow, or sent as a user message when there is
        none, since the format admits no free-standing tool message.
        """
        rendered: List[Dict[str, Any]] = self._system_messages()
        for m in self._messages:
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

    def _system_messages(self) -> List[Dict[str, Any]]:
        prompt = self.system_prompt
        return [{"role": "system", "content": prompt}] if prompt else []

    # --- persistence -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"system": self._system, "messages": [m.to_dict() for m in self._messages]}
        if self.rules:
            data["rules"] = list(self.rules)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        conv = cls()
        conv._system = data.get("system", "")
        conv.rules = list(data.get("rules", []))
        conv._messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return conv

    def save(self, path: str) -> None:
        write_atomically(path, json.dumps(self.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"))

    @classmethod
    def load(cls, path: str) -> "Conversation":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
