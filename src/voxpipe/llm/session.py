"""Session: runs conversation turns against an LLM, executing the tools it calls."""

import asyncio
import concurrent.futures
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Mapping, Optional, Tuple, Union

from ..core.exceptions import LLMError, ToolError
from ..core.utils import get_logger
from .context import ContextHandler
from .conversation import Conversation, Message
from .model import LLM
from .state import SessionState
from .tools import PERMISSION_KEYS, MalformedToolCall, Tool, ToolCall, ToolChoice, ToolResult, ToolSet

CONFIRM_TOOL = "_confirm"

_ANSWER_NOTE = "Now, generate an answer based only on the returned responses."
_FAILURE_NOTE = (
    "One or more tools failed to execute. Please inform the user of the error and suggest an alternative action."
)
_ASK_USER_NOTE = "Ask the user, then call _confirm."
_PARSE_ERROR = "Could not parse tool call. Please try again with valid JSON."
_FALLBACK_REPLY = "I encountered a persistent error processing tool calls."


@dataclass(frozen=True)
class TextDelta:
    """A piece of text for the user, in the order it should be spoken."""

    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    """The model asked for a tool; it is about to run (or be refused)."""

    call: ToolCall


@dataclass(frozen=True)
class ToolResultEvent:
    """A requested tool finished: exactly one of ``result`` and ``error`` is set."""

    call: ToolCall
    result: Optional[ToolResult] = None
    error: Optional[str] = None


SessionEvent = Union[TextDelta, ToolCallEvent, ToolResultEvent]
EventStream = Generator[SessionEvent, None, None]


@dataclass
class ToolExecutionResult:
    """Outcome of one requested tool call; ``call`` is None when the request itself was unparseable."""

    call: Optional[ToolCall]
    result: Optional[ToolResult] = None
    error: Optional[str] = None


@dataclass
class _Turn:
    """Bookkeeping for one turn in flight."""

    conversation: Conversation
    cancelled: threading.Event = field(default_factory=threading.Event)
    added: List[Message] = field(default_factory=list)
    unrecorded_speech: List[str] = field(default_factory=list)


class Session:
    """A conversation bound to an LLM.

    Calling the session with the user's text runs one turn and streams the
    reply.  Turns run one at a time; a second caller waits for the first to
    finish.  :meth:`cancel` and :meth:`reset` never wait: they stop the turn in
    flight at its next chunk.

    What the model is told and may do is set here: :attr:`instructions` open
    the system prompt, the tools' own instructions and the :attr:`rules` are
    listed after them, and :attr:`tools` holds the tools.  All three survive
    :meth:`reset`.
    """

    def __init__(
        self,
        llm: LLM,
        conversation: Optional[Conversation] = None,
        *,
        instructions: Optional[str] = None,
        rules: Optional[Iterable[str]] = None,
        tools: Optional[Iterable[Tool]] = None,
        max_tool_iterations: int = 1,
        context_handler: Optional[ContextHandler] = None,
        tool_timeout: float = 10.0,
    ):
        """``instructions`` replace, and ``rules`` and ``tools`` add to, what ``conversation`` already carries."""
        self.logger = get_logger(__name__)
        self.llm = llm
        self.conversation = conversation if conversation is not None else Conversation()
        if instructions is not None:
            self.instructions = instructions
        if rules is not None:
            self.rules.extend(rules)
        if tools is not None:
            self.tools.add(*tools)
        self.max_tool_iterations = max_tool_iterations
        self.context_handler = context_handler
        self.tool_timeout = tool_timeout
        self.state = SessionState()

        self._lock = threading.Lock()  # held for the length of a turn
        self._active: Optional[_Turn] = None
        self._turn_number = 0

        self.tool_caller = ToolCaller()
        self.tool_caller.start()

    # --- what the model is told and may do -------------------------------

    @property
    def instructions(self) -> str:
        """The opening text of the system prompt."""
        return self.conversation.system_message

    @instructions.setter
    def instructions(self, text: str) -> None:
        self.conversation.set_system_message(text)

    @property
    def rules(self) -> List[str]:
        """Standing rules, listed in the system prompt after the tools' own instructions."""
        return self.conversation.rules

    @rules.setter
    def rules(self, rules: Iterable[str]) -> None:
        self.conversation.rules = list(rules)

    @property
    def tools(self) -> ToolSet:
        """The tools the model may call; add more with ``session.tools.add(tool)``."""
        return self.conversation.tools

    @tools.setter
    def tools(self, tools: Mapping[str, Tool] | Iterable[Tool]) -> None:
        self.conversation.tools = ToolSet(tools)

    @property
    def system_prompt(self) -> str:
        """The system prompt as the model receives it: instructions, then tool instructions and rules."""
        return self.conversation.system_prompt

    # --- lifecycle -------------------------------------------------------

    def cancel(self) -> None:
        """Stop the turn in flight, if any, at its next chunk. What it already said stays in history."""
        turn = self._active
        if turn is not None:
            turn.cancelled.set()

    def reset(self, conversation: Optional[Conversation] = None) -> None:
        """Start over, cancelling any turn in flight and clearing session state.

        Without an argument the instructions, rules and tools carry over to a
        fresh conversation; pass a Conversation to replace them too.
        """
        self.cancel()
        previous = self.conversation
        if self.context_handler:
            self.context_handler.finalize(previous)
        # A new object rather than a cleared one: a cancelled turn may still be writing to the old.
        self.conversation = conversation if conversation is not None else previous.blank_copy()
        self.state.clear()

    def close(self) -> None:
        """Stop any turn in flight, flush the context handler and release the tool loop."""
        self.cancel()
        if self.context_handler:
            self.context_handler.finalize(self.conversation)
        self.tool_caller.stop()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def save(self, path: str) -> None:
        """Save the conversation and session state into a directory."""
        os.makedirs(path, exist_ok=True)
        self.conversation.save(os.path.join(path, "conversation.json"))
        self.state.save(path)

    @classmethod
    def load(cls, path: str, llm: LLM, **kwargs) -> "Session":
        """Restore a saved session around ``llm``; keyword arguments go to the constructor.

        Tools are not persisted: pass ``tools=`` again.
        """
        conv_path = os.path.join(path, "conversation.json")
        conversation = Conversation.load(conv_path) if os.path.exists(conv_path) else Conversation()
        session = cls(llm=llm, conversation=conversation, **kwargs)
        session.state.load(path)
        return session

    # --- running turns ---------------------------------------------------

    def __call__(
        self, query: str | None = None, *, context: str | None = None, **kwargs
    ) -> Generator[str, None, None]:
        """Run a turn and stream the text of the reply. See :meth:`events`."""
        events = self.events(query, context=context, **kwargs)
        try:
            for event in events:
                if isinstance(event, TextDelta):
                    yield event.text
        finally:
            # Closing this stream must end the turn now, not whenever the inner generator is collected.
            events.close()

    def events(self, query: str | None = None, *, context: str | None = None, **kwargs) -> EventStream:
        """Run a turn, streaming text and tool activity as it happens.

        ``query`` is the user's message; omit it to continue from the
        conversation as it stands.  ``context`` is shown to the model ahead of
        the query without becoming part of what the user said.  Remaining
        keyword arguments are generation parameters for the LLM.

        If the LLM fails, the turn's messages are rolled back and LLMError is
        raised.  Closing the generator early cancels the turn.
        """
        with self._lock:
            turn = _Turn(self.conversation)
            self._active = turn
            try:
                if query:
                    self._turn_number += 1
                    self._add_user_message(turn, query, context)
                    self.logger.info("query=%r", query)
                self._register_confirm(turn.conversation)
                if self.context_handler:
                    self.context_handler.pre_process(turn.conversation)

                yield from self._run(turn, kwargs)

                if self.context_handler and not turn.cancelled.is_set():
                    self.context_handler.post_process(turn.conversation)
            except Exception:
                turn.unrecorded_speech.clear()
                turn.conversation.discard(turn.added)
                raise
            finally:
                self._flush_speech(turn)
                if self._active is turn:
                    self._active = None

    def _add_user_message(self, turn: _Turn, query: str, context: Optional[str]) -> None:
        """Append the user's message, or extend their previous one if it never got a reply."""
        last = turn.conversation.last_message
        if last is not None and last.role is Message.Role.user:
            last.content = f"{last.content}\n{query}"
            if context:
                last.context = f"{last.context}\n{context}" if last.context else context
        else:
            turn.added.append(turn.conversation.add_user_message(query, context=context))

    def _run(self, turn: _Turn, llm_kwargs: Dict[str, Any]) -> EventStream:
        """Alternate model passes and tool execution until the model answers in text."""
        for index in range(self.max_tool_iterations + 1):
            final = index == self.max_tool_iterations
            text: List[str] = []
            requests: List[Union[ToolCall, MalformedToolCall]] = []
            interrupted = True
            try:
                yield from self._stream_pass(turn, "none" if final else "auto", llm_kwargs, text, requests)
                interrupted = turn.cancelled.is_set()
            finally:
                # Whatever was streamed has been heard; keep it even if the turn stops here.
                # Only a pass that goes on to run tools records its text with the tool calls instead.
                reply = "".join(text)
                if interrupted or final or not requests:
                    self._record_reply(turn, reply)

            if turn.cancelled.is_set():
                return
            if final or not requests:
                if requests and not reply:
                    self.logger.warning("Model kept calling tools when asked to answer; giving up on the turn.")
                    self._record_reply(turn, _FALLBACK_REPLY)
                    yield TextDelta(_FALLBACK_REPLY)
                elif requests:
                    self.logger.warning("Ignored %d tool call(s) made during the answer pass.", len(requests))
                return

            results = yield from self._execute(turn, requests)
            if turn.cancelled.is_set():
                return
            yield from self._record_tool_round(turn, reply, results)
            if all(r.result is not None and r.result.speech for r in results):
                return  # every tool answered the user itself; nothing left for the model to say

    def _stream_pass(
        self,
        turn: _Turn,
        tool_choice: str,
        llm_kwargs: Dict[str, Any],
        text: List[str],
        requests: List[Union[ToolCall, MalformedToolCall]],
    ) -> EventStream:
        """Stream one model pass, collecting its text and tool requests into the given lists."""
        self.logger.debug("Starting LLM pass (tool_choice=%s)", tool_choice)
        stream = self.llm(turn.conversation, session_state=self.state, tool_choice=tool_choice, **llm_kwargs)
        try:
            for item in stream:
                if turn.cancelled.is_set():
                    return
                if isinstance(item, str):
                    text.append(item)
                    yield TextDelta(item)
                else:
                    requests.append(item)
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()

    def _execute(
        self, turn: _Turn, requests: List[Union[ToolCall, MalformedToolCall]]
    ) -> Generator[SessionEvent, None, List[ToolExecutionResult]]:
        """Run the requested tools concurrently; returns their results in request order."""
        results: List[ToolExecutionResult] = []
        running: List[Tuple[ToolExecutionResult, ToolCall, "concurrent.futures.Future[ToolResult]"]] = []
        tools = turn.conversation.tools

        for request in requests:
            if isinstance(request, MalformedToolCall):
                results.append(ToolExecutionResult(call=None, error=_PARSE_ERROR))
                continue
            self.logger.info("ToolCall: %s args=%s", request.name, request.arguments)
            yield ToolCallEvent(request)
            outcome = ToolExecutionResult(call=request)
            tool = tools.get(request.name)
            if tool is None:
                outcome.error = f"'{request.name}' is not a recognized tool. Available: {list(tools)}."
            else:
                outcome.result = tool.permission_gate(self.state.tools.get_permission(tool.name))
                if outcome.result is None:
                    running.append((outcome, request, self.tool_caller.submit(tool, request.arguments)))
            results.append(outcome)

        deadline = time.monotonic() + self.tool_timeout
        for outcome, call, future in running:
            try:
                outcome.result = future.result(timeout=max(0.0, deadline - time.monotonic()))
            except concurrent.futures.TimeoutError:
                future.cancel()
                outcome.error = f"{call.name} timed out"
                self.logger.error("Tool %s timed out", call.name)
            except Exception as e:  # noqa: BLE001 - a tool may raise anything; the model is told
                outcome.error = str(e) or type(e).__name__
                self.logger.error("Error calling tool %s: %s", call.name, e, exc_info=True)

        for outcome in results:
            if outcome.call is None:
                continue
            if isinstance(outcome.result, ToolChoice):
                self.state.tools.add_pending(outcome.call, outcome.result, turn=self._turn_number)
            yield ToolResultEvent(outcome.call, outcome.result, outcome.error)
        return results

    def _record_tool_round(self, turn: _Turn, reply: str, results: List[ToolExecutionResult]) -> EventStream:
        """Write a round of tool calls and results to history and speak what the tools said."""
        conversation = turn.conversation
        calls = [r.call for r in results if r.call is not None]
        if calls or reply:
            turn.added.append(conversation.add_assistant_message(reply, tool_calls=calls))

        notes: Dict[str, None] = {}  # insertion-ordered set
        for outcome in results:
            result = outcome.result
            if result is None:
                content = f"Tool Error: {outcome.error}"
                notes.setdefault(_FAILURE_NOTE)
            else:
                content = result.result
                if isinstance(result, ToolChoice):
                    notes.setdefault(_ASK_USER_NOTE)
                elif not result.speech:
                    notes.setdefault(_ANSWER_NOTE)
            turn.added.append(conversation.add_tool_message(content, tool_call=outcome.call))
        for note in notes:
            turn.added.append(conversation.add_tool_message(note))

        if any(isinstance(r.result, ToolChoice) for r in results):
            self._register_confirm(conversation)
        for outcome in results:
            if outcome.result is not None and outcome.result.speech:
                turn.unrecorded_speech.append(outcome.result.speech)
                yield TextDelta(outcome.result.speech)

    def _record_reply(self, turn: _Turn, reply: str) -> None:
        """Record the assistant's spoken reply, led by any tool speech not yet in history."""
        spoken = " ".join(part for part in (*turn.unrecorded_speech, reply) if part)
        turn.unrecorded_speech.clear()
        if spoken:
            turn.added.append(turn.conversation.add_assistant_message(spoken))
            self.logger.info("response=%s", spoken)

    def _flush_speech(self, turn: _Turn) -> None:
        if turn.unrecorded_speech:
            self._record_reply(turn, "")

    # --- confirmations ---------------------------------------------------

    def _register_confirm(self, conversation: Conversation) -> None:
        """Offer the model the ``_confirm`` tool once something may need the user's answer."""
        tools = conversation.tools
        if CONFIRM_TOOL in tools:
            return
        needs_confirm = self.state.tools.has_pending() or any(
            t.may_return_choice or t.requires_permission for t in tools.values()
        )
        if not needs_confirm:
            return

        tools[CONFIRM_TOOL] = Tool(
            name=CONFIRM_TOOL,
            description=(
                "Confirm or resolve a pending ToolChoice or permission request using its unique UID.\n"
                "Match keys from the ToolChoice payload: pick one item for list options (e.g. choice={'slot': '2'} "
                "from {'slot': ['1','2']}), or provide text for open keys (e.g. choice={'answer': 'user response'}).\n"
                "Examples:\n"
                "- Permission: choice={'allow': true, 'remember': false} "
                "(set remember=true if user says 'always' or 'remember')\n"
                "- Option selection: choice={'slot': '2'}\n"
                "- Open text choice: choice={'answer': 'user input'}"
            ),
            parameters=Tool.Parameter(
                type="object",
                properties={
                    "uid": Tool.Parameter(
                        type="string",
                        description="The exact UID string from the pending ToolChoice payload (e.g. 'tc_1234')",
                    ),
                    "choice": Tool.Parameter(
                        type="object",
                        description="Dictionary payload mapping the selection keys to their values "
                        "matching the ToolChoice options",
                    ),
                },
                required=["uid", "choice"],
            ),
            callback=self._on_confirm,
        )

    async def _on_confirm(self, uid: str, choice: Union[dict, str]) -> ToolResult:
        """Resolve a pending ToolChoice with the user's answer and run the tool it was holding back."""
        pending = self.state.tools.get_pending(uid)
        if pending is None:
            raise ToolError(f"No pending ToolChoice found for UID '{uid}'")
        if pending.turn == self._turn_number:
            # The answer must be the user's: a model cannot grant its own request in the same turn.
            raise ToolError("The user has not answered yet. Ask them, then call _confirm after they reply.")

        if isinstance(choice, str):
            try:
                choice = json.loads(choice)
            except json.JSONDecodeError:
                raise ToolError("Parameter 'choice' could not be parsed as valid JSON.") from None
        if not isinstance(choice, dict):
            raise ToolError("Parameter 'choice' must be a dictionary.")

        tool = self.conversation.tools.get(pending.call.name)
        if tool is None:
            raise ToolError(f"Unknown tool '{pending.call.name}' for _confirm")
        expected = set(pending.choice.choices_dict) - {"uid"}
        given = set(choice) - {"uid"}

        if expected == PERMISSION_KEYS:
            if "allow" not in given or not given <= PERMISSION_KEYS:
                raise ToolError(f"_confirm choice keys {sorted(given)} must be 'allow' and optionally 'remember'")
            allow = _as_bool(choice["allow"])
            if _as_bool(choice.get("remember", False)):
                self.state.tools.set_permission(tool.name, allow)
            self.state.tools.remove_pending(uid)
            if not allow:
                return ToolResult(
                    {"status": "cancelled", "action": tool.name},
                    speech=f"Action {tool.name} was cancelled.",
                )
            return await tool.aexecute(**pending.call.arguments)

        if expected != given:
            raise ToolError(f"_confirm choice keys {sorted(given)} do not match expected keys {sorted(expected)}")
        self.state.tools.remove_pending(uid)
        return await tool.aexecute(**{**pending.call.arguments, **{k: choice[k] for k in given}})


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() in {"true", "1"}


class ToolCaller:
    """Runs tool callbacks on a background asyncio loop, so async and blocking tools share one path."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

    def submit(self, tool: Tool, arguments: Dict[str, Any]) -> "concurrent.futures.Future[ToolResult]":
        """Start ``tool`` with the given arguments; the future resolves to its ToolResult."""
        if self._loop is None:
            raise LLMError("ToolCaller is not running.")
        return asyncio.run_coroutine_threadsafe(tool.aexecute(**arguments), self._loop)

    def start(self) -> None:
        """Start the background event loop; a no-op if it is already running."""
        if self._loop_thread and self._loop_thread.is_alive():
            return

        ready = threading.Event()

        def run_loop():
            loop = asyncio.new_event_loop()
            self._loop = loop
            loop.call_soon(ready.set)
            try:
                loop.run_forever()
            finally:
                self._loop = None
                loop.close()

        self._loop_thread = threading.Thread(target=run_loop, name="voxpipe-tools", daemon=True)
        self._loop_thread.start()
        if not ready.wait(timeout=5.0):
            raise LLMError("Failed to start background asyncio loop within timeout.")

    def stop(self) -> None:
        """Stop the background event loop; tools still running are abandoned."""
        loop, thread = self._loop, self._loop_thread
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass  # loop already closed
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                self.logger.warning("Background loop thread did not stop gracefully.")
        self._loop_thread = None

    def __enter__(self) -> "ToolCaller":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
