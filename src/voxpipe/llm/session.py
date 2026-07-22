from dataclasses import dataclass
from functools import partial
import asyncio
import inspect
import json
import os
from queue import Queue
import threading
from typing import Any, Dict, Generator, Optional, Tuple

from ..core.utils import get_logger
from ..core.exceptions import LLMError, ToolError

from .model import LLM
from .conversation import Conversation
from .tools import Tool, ToolCall, ToolResult, ToolChoice, ToolRegistry
from .context import ContextHandler, DropOldestStrategy
from ..storage import MemoryStore, RAMStorage, SQLiteStorage, Record


class Session:

    def __init__(
        self,
        llm: LLM,
        conversation: Optional[Conversation] = None,
        max_turns: int = 20,
        max_tool_iterations: int = 1,
        context_handler: Optional[ContextHandler] = None,
        memory: Optional[Any] = None,
    ):
        self.logger = get_logger(__name__)
        self.llm = llm
        self.conversation = conversation or Conversation()
        self.tool_caller = ToolCaller()
        self.max_tool_iterations = max_tool_iterations

        if memory is not None and not isinstance(memory, MemoryStore):
            memory = MemoryStore(backend=memory)

        self.context_handler = context_handler or ContextHandler(max_turns=max_turns, memory=memory)
        self._context_strategy = self.context_handler

        self._session_state = dict()
        self._lock = threading.Lock()

        self.tool_caller.start()

    def _register_confirm(self):
        if "_confirm" in self.conversation.tools:
            return
        if not any(
            getattr(t, "may_return_choice", False) or getattr(t, "requires_permission", False)
            for t in self.conversation.tools.values()
        ) and not any("pending" in self.conversation.get_meta(tn) for tn in self.conversation.tools):
            return

        self.conversation.tools["_confirm"] = Tool(
            name="_confirm",
            description=(
                "Confirm or resolve a pending ToolChoice or permission request using its unique UID.\n"
                "Match keys from the ToolChoice payload: pick one item for list options (e.g. choice={'slot': '2'} from {'slot': ['1','2']}), or provide text for open keys (e.g. choice={'answer': 'user response'}).\n"
                "Examples:\n"
                "- Permission: choice={'allow': true, 'remember': false} (set remember=true if user says 'always' or 'remember')\n"
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
                        description="Dictionary payload mapping the selection keys to their values matching the ToolChoice options",
                    ),
                },
                required=["uid", "choice"],
            ),
            callback=self._on_confirm,
            instruction="When the user answers a pending ToolChoice prompt, call _confirm with its uid and choice payload.",
        )

    def _on_confirm(self, uid: str = None, choice: dict = None, **kwargs) -> ToolResult:
        if kwargs:
            if not uid and "uid" in kwargs:
                uid = kwargs["uid"]
            if not choice:
                if "choice" in kwargs and isinstance(kwargs["choice"], dict):
                    choice = kwargs["choice"]
                else:
                    choice = {k: v for k, v in kwargs.items() if k != "uid"}

        if not uid:
            raise ToolError("Missing required parameter 'uid' for _confirm.")

        found_call, found_choice, target_tool_name = None, None, None
        for t_name in self.conversation.tools:
            meta = self.conversation.get_meta(t_name)
            pending = meta.get("pending", {})
            if uid in pending:
                target_tool_name = t_name
                found_call, found_choice = pending[uid]
                break

        if not found_call:
            raise ToolError(f"No pending ToolChoice found for UID '{uid}'")

        if isinstance(choice, str):
            try:
                choice = json.loads(choice)
            except Exception:
                raise ToolError("Parameter 'choice' could not be parsed as valid JSON.")

        if choice is None or not isinstance(choice, dict):
            raise ToolError("Parameter 'choice' must be a dictionary.")

        expected_keys = set(found_choice.choices_dict.keys()) - {"uid"}
        given_keys = set(choice.keys()) - {"uid"}

        if expected_keys == {"allow", "remember"} and "allow" in given_keys:
            raw_allow = choice["allow"]
            raw_remember = choice.get("remember", False)
            allow = raw_allow if isinstance(raw_allow, bool) else str(raw_allow).lower() in {"true", "1"}
            remember = raw_remember if isinstance(raw_remember, bool) else str(raw_remember).lower() in {"true", "1"}
            if remember:
                self.conversation.set_permission(target_tool_name, allow)

            self.conversation.get_meta(target_tool_name).get("pending", {}).pop(uid, None)

            if allow:
                tool = self.conversation.tools.get(target_tool_name)
                if tool is None:
                    raise ToolError(f"Unknown tool '{target_tool_name}' for _confirm")
                return tool._execute_direct(**found_call.arguments)
            else:
                return ToolResult(
                    {"status": "cancelled", "action": target_tool_name},
                    speech=f"Action {target_tool_name} was cancelled."
                )

        if expected_keys != given_keys:
            raise ToolError(
                f"_confirm choice keys {list(given_keys)} do not match expected keys {list(expected_keys)}"
            )

        self.conversation.get_meta(target_tool_name).get("pending", {}).pop(uid, None)
        tool = self.conversation.tools.get(target_tool_name)
        if tool is None:
            raise ToolError(f"Unknown tool '{target_tool_name}' for _confirm")
        final_args = dict(found_call.arguments)
        final_args.update(choice)
        return tool._execute_direct(**final_args)

    def reset(self, conversation: Optional[Conversation] = None):
        """Reset conversation and provider state without replacing the session."""
        with self._lock:
            self.conversation = conversation or Conversation()
            self._session_state.clear()

    def close(self):
        """Release the background tool execution loop."""
        self.tool_caller.stop()

    def save(self, path: str, *, save_kv_cache: bool = False):
        """Save session to a directory."""
        os.makedirs(path, exist_ok=True)
        self.conversation.save(os.path.join(path, "conversation.json"))
        state = {}
        with self._lock:
            for k, v in self._session_state.items():
                if k == "model_state" and not save_kv_cache:
                    continue
                if isinstance(v, (bytes, bytearray)):
                    continue
                state[k] = v
        with open(os.path.join(path, "state.json"), "w") as f:
            json.dump(state, f, indent=2)

        if save_kv_cache:
            state_data = self._session_state.get("model_state")
            if isinstance(state_data, (bytes, bytearray)):
                with open(os.path.join(path, "kv_cache.bin"), "wb") as f:
                    f.write(state_data)

    @classmethod
    def load(cls, path: str, llm: LLM) -> "Session":
        """Load session from a directory with a fresh LLM instance."""
        conv_path = os.path.join(path, "conversation.json")
        state_path = os.path.join(path, "state.json")
        kv_path = os.path.join(path, "kv_cache.bin")

        conversation = Conversation.load(conv_path) if os.path.exists(conv_path) else Conversation()
        session = cls(llm=llm, conversation=conversation)

        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
                session._session_state.update(state)

        if os.path.exists(kv_path):
            with open(kv_path, "rb") as f:
                session._session_state["model_state"] = f.read()

        return session

    def complete_once(
        self,
        query: str,
        *,
        system: str | None = None,
        **kwargs,
    ) -> str:
        """Run an isolated completion without mutating conversation history."""
        if not isinstance(query, str) or not query.strip():
            raise LLMError("One-shot queries must be non-empty strings.")

        conversation = Conversation()
        if system:
            conversation.set_system_message(system)
        conversation.add_user_message(query)
        state = {}
        response_chunks = []

        # ASVS 15.4.1 / 15.4.3: the shared provider and its model state are
        # accessed under the Session-owned lock, just like normal completions.
        with self._lock:
            for chunk in self.llm(
                conversation,
                session_state=state,
                tool_choice="none",
                **kwargs,
            ):
                if isinstance(chunk, ToolCall):
                    self.logger.warning(
                        "Ignored an unexpected tool call during one-shot inference."
                    )
                    continue
                response_chunks.append(chunk)
        return "".join(response_chunks).strip()

    def __call__(
        self, query: str | None = None, **kwargs
    ) -> Generator[str, None, None]:
        with self._lock:
            self._register_confirm()
            if query:
                self.conversation.add_user_message(query)
                self.logger.info(f"{query=}")

            self.context_handler.handle(self.conversation, self.llm)
            self.logger.info("Starting LLM call...")

            if self.max_tool_iterations == 0:
                yield from self._generate_response(tool_choice="none")
            else:
                for iteration in range(self.max_tool_iterations + 1):
                    is_final = (iteration == self.max_tool_iterations)
                    tc = "none" if is_final else "auto"
                    yield from self._generate_response(tool_choice=tc)
                    results = self.tool_caller.gather(conversation=self.conversation)
                    if not results:
                        break
                    any_error = False
                    for name, exec_res in results.items():
                        result = exec_res.result
                        if isinstance(result, ToolChoice):
                            self._register_confirm()
                            self.conversation.add_tool_message(f"{name}: {result.result}")
                            self.conversation.add_assistant_message("Ask the user, then call _confirm.")
                            if result.speech:
                                yield result.speech
                        elif isinstance(result, ToolResult):
                            self.conversation.add_tool_message(f"{name}: {result.result}")
                            if result.speech:
                                self.conversation.add_assistant_message(result.speech)
                                yield result.speech
                            else:
                                self.conversation.add_tool_message(
                                    "Now, generate an answer based only on the returned responses."
                                )
                        elif isinstance(result, str) and result.startswith("Tool Error:"):
                            any_error = True
                            self.conversation.add_tool_message(f"{name}: {result}")
                        else:
                            raise TypeError(
                                f"Tool '{name}' returned {type(result).__name__}, expected ToolResult"
                            )
                    if any_error:
                        self.conversation.add_tool_message(
                            "One or more tools failed to execute. Please inform the user of the error and suggest an alternative action."
                        )

            self.tool_caller.drain()

    def _generate_response(self, _retry_count: int = 0, **kwargs) -> Generator[str, None, None]:
        if _retry_count > 2:
            self.logger.error("Too many tool parse retries. Aborting.")
            yield "I encountered a persistent error processing tool calls."
            return

        response_chunks = []
        tool_dispatched = False
        llm_stream = self.llm(
            self.conversation, session_state=self._session_state, **kwargs
        )
        self.logger.debug("_generate_response: starting model stream")
        for chunk in llm_stream:
            if isinstance(chunk, ToolCall):
                self.logger.info("ToolCall: %s args=%s", chunk.name, chunk.arguments)
                try:
                    tool_name = chunk.name
                    tool_args = chunk.arguments
                except Exception as e:
                    self.logger.warning(f"Error parsing tool response: {repr(e)}")
                    self.conversation.add_tool_message(
                        "Tool Error: Could not parse tool call. Please try again with valid JSON."
                    )
                    yield from self._generate_response(_retry_count=_retry_count + 1, tool_choice="none")
                    return
                tool = self.conversation.tools.get(tool_name)
                if tool is None:
                    self.logger.warning(f"Tool '{tool_name}' not found in conversation tools.")
                    self.conversation.add_tool_message(
                        f"Tool Error: '{tool_name}' is not a recognized tool. Available: {list(self.conversation.tools.keys())}."
                    )
                    yield from self._generate_response(_retry_count=_retry_count + 1, tool_choice="none")
                    return
                tool_dispatched = True
                response_chunks.clear()
                if kwargs.get("tool_choice") != "none":
                    self.tool_caller(tool, tool_call=chunk, **tool_args)
                else:
                    self.logger.warning("Spurious tool call in final pass, ignoring %s", tool_name)
            else:
                response_chunks.append(chunk)

        if not tool_dispatched:
            final_response = "".join(response_chunks)
            if final_response:
                self.conversation.add_assistant_message(final_response)
                self.logger.info(f"response={final_response}")
                yield final_response


@dataclass
class ToolExecutionResult:
    tool_call: ToolCall
    result: ToolResult | ToolChoice | str


class ToolCaller:

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop = None
        self._loop_thread: threading.Thread = None
        self._loop_ready_event = threading.Event()
        self._futures: Queue[Tuple[ToolCall, asyncio.Future]] = None
        self.logger = get_logger(__name__)

    def __call__(self, tool, tool_call: ToolCall | None = None, **tool_args):
        if tool_call is None:
            tool_call = ToolCall(name=tool.name, arguments=tool_args)
        tool_callable = partial(tool.__call__, **tool_args)

        if asyncio.iscoroutinefunction(tool.callback) or inspect.iscoroutinefunction(getattr(tool_callable, "func", None)):
            future = asyncio.run_coroutine_threadsafe(
                tool_callable(), self._loop
            )
        else:
            future = asyncio.run_coroutine_threadsafe(
                asyncio.to_thread(tool_callable), self._loop
            )
        self._futures.put((tool_call, future))

    def gather(self, conversation: Conversation | None = None) -> Dict[str, ToolExecutionResult]:
        responses = {}
        while not self._futures.empty():
            tool_call, future = self._futures.get()
            tool_name = tool_call.name
            try:
                self.logger.debug("Gathering tool %s ...", tool_name)
                res = future.result(timeout=10.0)
                if isinstance(res, ToolChoice) and conversation is not None:
                    meta = conversation.get_meta(tool_name)
                    pending_map = meta.setdefault("pending", {})
                    pending_map[res.uid] = (tool_call, res)
                responses[tool_name] = ToolExecutionResult(tool_call=tool_call, result=res)
                self.logger.debug("Gathered tool %s: %s", tool_name, str(res)[:80])
            except TimeoutError:
                err_msg = f"Tool Error: {tool_name} timed out"
                responses[tool_name] = ToolExecutionResult(tool_call=tool_call, result=err_msg)
                self.logger.error(f"Tool {tool_name} timed out")
            except Exception as e:
                err_msg = f"Tool Error: {e}"
                responses[tool_name] = ToolExecutionResult(tool_call=tool_call, result=err_msg)
                self.logger.error(
                    f"Error calling tool {tool_name}: {e}", exc_info=True
                )

        return responses

    def drain(self):
        """Cancel any leftover tool futures (spurious second-pass calls)."""
        drained = 0
        while not self._futures.empty():
            _, future = self._futures.get()
            future.cancel()
            drained += 1
        if drained:
            self.logger.debug("Drained %d stale tool future(s)", drained)

    def start(self):
        """Starts the background asyncio event loop for tool execution."""
        if self._loop_thread and self._loop_thread.is_alive():
            return

        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.call_soon_threadsafe(self._loop_ready_event.set)
            loop.run_forever()
            loop.close()
            self._loop = None
            self._loop_ready_event.clear()

        self._futures = Queue()
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

        self._loop_ready_event.wait(timeout=5.0)
        if not self._loop_ready_event.is_set():
            raise LLMError(
                "Failed to start background asyncio loop within timeout."
            )

    def stop(self):
        """Stops the background asyncio event loop gracefully."""
        if self._loop:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass  # loop already closed
            self._loop_thread.join(timeout=5)
            if self._loop_thread.is_alive():
                self.logger.warning(
                    "Background loop thread did not stop gracefully."
                )
            self._loop_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def __del__(self):
        try:
            self.stop()
        except Exception:
            # Destructors may run during interpreter teardown.
            pass
