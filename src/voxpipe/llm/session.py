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
from .tools import Tool, ToolCall, ToolResult, ToolChoice


class Session:

    def __init__(
        self,
        llm: LLM,
        conversation: Optional[Conversation] = None,
        max_turns: int = 20,
        max_tool_iterations: int = 1,
    ):
        self.logger = get_logger(__name__)
        self.llm = llm
        self.conversation = conversation or Conversation()
        self.tool_caller = ToolCaller()
        self._context_strategy = llm.create_context_strategy(max_turns)
        self.max_tool_iterations = max_tool_iterations

        self._session_state = dict()
        self._lock = threading.Lock()

        self.tool_caller.start()

    def _register_confirm(self):
        if "_confirm" in self.conversation.tools:
            return
        if not any(
            getattr(t, "may_return_choice", False)
            for t in self.conversation.tools.values()
        ):
            return
        self.conversation.tools["_confirm"] = Tool(
            name="_confirm",
            description="Call this with the user's resolved parameters to confirm a pending action.",
            parameters=Tool.Parameter(
                type="object",
                properties={
                    "action": Tool.Parameter(type="string", description="The tool to confirm"),
                    "params": Tool.Parameter(type="object", description="The resolved parameters"),
                },
                required=["action", "params"],
            ),
            callback=self._on_confirm,
            instruction="Call _confirm when the user has made their choice.",
        )

    def _on_confirm(self, action: str, params: dict) -> ToolResult:
        tool = self.conversation.tools.get(action)
        if tool is None:
            raise ToolError(f"Unknown tool '{action}' for _confirm")
        return tool(**params)

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

            self._context_strategy.trim(self.conversation, self.llm)
            self.logger.info("Starting LLM call...")

            if self.max_tool_iterations == 0:
                yield from self._generate_response(tool_choice="none")
            else:
                for iteration in range(self.max_tool_iterations + 1):
                    is_final = (iteration == self.max_tool_iterations)
                    tc = "none" if is_final else "auto"
                    yield from self._generate_response(tool_choice=tc)
                    results = self.tool_caller.gather()
                    if not results:
                        break
                    any_error = False
                    for name, result in results.items():
                        if isinstance(result, ToolChoice):
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
                    self.tool_caller(tool, **tool_args)
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


class ToolCaller:

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop = None
        self._loop_thread: threading.Thread = None
        self._loop_ready_event = threading.Event()
        self._futures: Queue[Tuple[str, asyncio.Future]] = None
        self.logger = get_logger(__name__)

    def __call__(self, tool, **tool_args):
        tool_callable = partial(tool.__call__, **tool_args)

        if asyncio.iscoroutinefunction(tool.callback) or inspect.iscoroutinefunction(getattr(tool_callable, "func", None)):
            future = asyncio.run_coroutine_threadsafe(
                tool_callable(), self._loop
            )
        else:
            future = asyncio.run_coroutine_threadsafe(
                asyncio.to_thread(tool_callable), self._loop
            )
        self._futures.put((tool.name, future))

    def gather(self) -> Dict[str, Any]:
        responses = {}
        while not self._futures.empty():
            tool_name, future = self._futures.get()
            try:
                self.logger.debug("Gathering tool %s ...", tool_name)
                responses[tool_name] = future.result(timeout=10.0)
                self.logger.debug("Gathered tool %s: %s", tool_name, str(responses[tool_name])[:80])
            except TimeoutError:
                responses[tool_name] = f"Tool Error: {tool_name} timed out"
                self.logger.error(f"Tool {tool_name} timed out")
            except Exception as e:
                responses[tool_name] = f"Tool Error: {e}"
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
