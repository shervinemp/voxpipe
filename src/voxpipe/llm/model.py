"""LLM providers: the common interface, local GGUF models, and LiteLLM-backed remote APIs."""

import ipaddress
import json
import os
import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional
from urllib.parse import urlparse

from ..core.config import get_config
from ..core.downloader import ensure_downloaded
from ..core.exceptions import LLMError, ProviderError
from ..core.models import AppConfig, LocalModelEntry
from ..core.utils import get_logger
from .conversation import Conversation
from .decoders import NativeDecoder, StreamDecoder, StreamItem, get_decoder
from .tools import MalformedToolCall, ToolCall

if TYPE_CHECKING:
    from .state import SessionState

ToolChoiceMode = str | Dict[str, Any]


class LLM(ABC):
    """A chat model that streams text and tool calls for a conversation."""

    decoder: StreamDecoder = NativeDecoder()
    n_ctx: int = 4096
    max_tokens: int = 512
    logger = get_logger(__name__)

    def __call__(
        self,
        conversation: Conversation,
        session_state: Optional["SessionState"] = None,
        **kwargs,
    ) -> Iterator[StreamItem]:
        """Stream the model's reply. Raises LLMError if inference fails."""
        raw_stream = self._infer(conversation=conversation, session_state=session_state, **kwargs)
        try:
            yield from self.decoder(raw_stream)
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001 - provider SDKs raise arbitrary types; callers get one
            raise LLMError(f"LLM inference failed: {e}") from e
        finally:
            # A decoder may stop early (at a tool call); release the provider's stream and locks now.
            close = getattr(raw_stream, "close", None)
            if close:
                close()

    @abstractmethod
    def _infer(
        self, conversation: Conversation, *, session_state: Optional["SessionState"], **kwargs
    ) -> Iterator[StreamItem]: ...

    def complete(self, query: str, *, system: str | None = None, **kwargs) -> str:
        """Answer one query on its own, with no tools and no conversation behind it."""
        if not isinstance(query, str) or not query.strip():
            raise LLMError("Queries must be non-empty strings.")

        conversation = Conversation()
        if system:
            conversation.set_system_message(system)
        conversation.add_user_message(query)

        chunks = []
        for chunk in self(conversation, tool_choice="none", **kwargs):
            if isinstance(chunk, str):
                chunks.append(chunk)
            else:
                self.logger.warning("Ignored an unexpected tool call during a one-off completion.")
        return "".join(chunks).strip()

    def count_tokens(self, text: str) -> int:
        """Estimate the token count of ``text`` without network access.

        Good enough for context budgeting with remote providers; local models
        override this with their exact tokenizer.
        """
        return max(1, (len(text) + 3) // 4)


class GGUFLLM(LLM):
    """A local GGUF model run through llama-cpp-python."""

    _GGML_CACHE_TYPES = {"f16": 1, "q4_0": 2, "q8_0": 8}
    _KV_STATE_KEY = "llama_state"

    def __init__(
        self,
        model: Any,
        *,
        n_ctx: int,
        max_tokens: int,
        decoder: Optional[StreamDecoder] = None,
        name: str = "model",
    ):
        """Wrap an already loaded ``llama_cpp.Llama``; see :meth:`from_config` to load one."""
        self.logger = get_logger(f"GGUF.{name}")
        self.model = model
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        if decoder is not None:
            self.decoder = decoder
        self._lock = Lock()
        self._active_state: Optional["SessionState"] = None

    @classmethod
    def from_config(cls, model_key: str, entry: Optional[LocalModelEntry] = None) -> "GGUFLLM":
        """Load the model configured under ``llm.local.<model_key>``, downloading it if needed."""
        if entry is None:
            entry = get_config().llm.local.get(model_key)
        if entry is None:
            raise ProviderError(f"Missing local model config for '{model_key}'.")

        decoder = get_decoder(entry.decoder)
        model_path = entry.model_path or ensure_downloaded(model_key)["model"]
        model = cls._load(model_path, entry, get_logger(f"GGUF.{model_key}"))
        return cls(model, n_ctx=entry.n_ctx, max_tokens=entry.max_tokens, decoder=decoder, name=model_key)

    @classmethod
    def _load(cls, model_path: str, entry: LocalModelEntry, logger) -> Any:
        """Load with GPU offload, falling back to CPU and then to CPU without flash attention."""
        from llama_cpp import Llama

        options: Dict[str, Any] = dict(model_path=model_path, n_ctx=entry.n_ctx, verbose=False)
        if entry.type_k:
            options["type_k"] = cls._GGML_CACHE_TYPES.get(entry.type_k, entry.type_k)
        if entry.type_v:
            options["type_v"] = cls._GGML_CACHE_TYPES.get(entry.type_v, entry.type_v)

        attempts = [
            dict(n_gpu_layers=-1, flash_attn=True),
            dict(n_gpu_layers=0, flash_attn=True),
            dict(n_gpu_layers=0, flash_attn=False),
        ]
        logger.info("Loading model...")
        last_error: Optional[Exception] = None
        for attempt in attempts:
            try:
                model = Llama(**options, **attempt)
            except Exception as e:  # noqa: BLE001 - llama.cpp reports load failures as assorted types
                logger.warning("Model load failed with %s: %s", attempt, e)
                last_error = e
                continue
            logger.info("Model loaded successfully with %s.", attempt)
            return model
        raise ProviderError(f"Could not load model from {model_path}: {last_error}") from last_error

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenize(text.encode("utf-8")))

    def _activate(self, session_state: Optional["SessionState"]) -> None:
        """Make the model's KV cache belong to ``session_state``.

        Sessions sharing one model would otherwise evaluate their whole prompt
        again each time another session ran in between.  The outgoing
        session's cache is parked on its state and the incoming one restored.
        A call without session state (one-shot completions) parks nothing.
        """
        if session_state is self._active_state:
            return
        if self._active_state is not None:
            self._active_state.model.runtime[self._KV_STATE_KEY] = self.model.save_state()
        self.model.reset()
        saved = session_state.model.runtime.get(self._KV_STATE_KEY) if session_state is not None else None
        if saved is not None:
            self.model.load_state(saved)
        self._active_state = session_state

    def _infer(
        self,
        conversation: Conversation,
        *,
        session_state: Optional["SessionState"],
        tool_choice: ToolChoiceMode = "auto",
        **kwargs,
    ) -> Iterator[str]:
        started = time.monotonic()
        chunks = 0
        with self._lock:
            self._activate(session_state)

            request = dict(kwargs)
            tools = [t.to_dict() for t in conversation.tools.values()]
            if tools and tool_choice != "none":
                request["tools"] = tools
                request.setdefault("tool_choice", tool_choice)
            else:
                request["tool_choice"] = "none"

            request.setdefault("max_tokens", self.max_tokens)
            stream = self.model.create_chat_completion(messages=conversation.messages, stream=True, **request)
            for chunk in stream:
                content = chunk["choices"][0]["delta"].get("content")
                if content:
                    chunks += 1
                    yield content

        elapsed = time.monotonic() - started
        if chunks and elapsed:
            self.logger.info("Generated %d chunks in %.1fs (%.0f/s)", chunks, elapsed, chunks / elapsed)


class LiteLLMProvider(LLM):
    """OpenAI-format adapter for provider-backed models supported by LiteLLM."""

    supported_providers = frozenset({"openai", "gemini", "ollama"})
    _provider_key_env = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    _allowed_generation_params = frozenset(
        {"frequency_penalty", "presence_penalty", "seed", "stop", "temperature", "top_p"}
    )
    _MAX_TOOL_NAME_CHARS = 64
    _MAX_TOOL_ARGUMENT_CHARS = 65_536
    _MAX_TOOL_CALL_INDEX = 128

    def __init__(
        self,
        model: str,
        *,
        provider: str,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 60.0,
        n_ctx: int = 16_384,
        max_tokens: int = 1_024,
        completion_fn: Optional[Callable[..., Iterable[Any]]] = None,
    ):
        self.logger = get_logger(self.__class__.__name__)

        if provider not in self.supported_providers:
            raise ProviderError(f"Unsupported LiteLLM provider: {provider!r}.")
        if not isinstance(model, str) or not 1 <= len(model.strip()) <= 256:
            raise ProviderError("Model must be a non-empty string up to 256 characters.")
        if "/" in model and not model.startswith(f"{provider}/"):
            raise ProviderError("The model prefix must match its configured provider.")
        if not isinstance(timeout, (int, float)) or not 1 <= timeout <= 300:
            raise ProviderError("Provider timeout must be between 1 and 300 seconds.")
        if not isinstance(n_ctx, int) or not 512 <= n_ctx <= 2_000_000:
            raise ProviderError("n_ctx must be between 512 and 2000000.")
        if not isinstance(max_tokens, int) or not 1 <= max_tokens < n_ctx:
            raise ProviderError("max_tokens must be positive and smaller than n_ctx.")
        if api_base is not None:
            self._validate_api_base(api_base)
        local_endpoint = api_base is not None and self._is_loopback(api_base)

        key_env = self._provider_key_env.get(provider)
        api_key = api_key or (os.environ.get(key_env) if key_env else None)
        if key_env and not api_key:
            # Local servers (llama.cpp, Ollama) don't need API keys
            if not local_endpoint:
                raise ProviderError(f"{key_env} is required for the {provider} provider.")
            api_key = "sk-no-key-required"
        if api_key is not None and not (isinstance(api_key, str) and api_key.strip()):
            raise ProviderError("Provider API keys must be non-blank strings.")

        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = float(timeout)
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        local_openai_base = api_base if provider == "openai" and local_endpoint else None
        self._create_stream = self._bind_backend(completion_fn, local_openai_base)

    def _bind_backend(
        self, completion_fn: Optional[Callable], local_openai_base: Optional[str]
    ) -> Callable[..., Iterable[Any]]:
        """Return a callable that opens a chat-completion stream for (messages, tools, params)."""
        if completion_fn is not None:
            if not callable(completion_fn):
                raise ProviderError("completion_fn must be callable.")
            return self._litellm_stream(completion_fn)

        if local_openai_base is not None:
            # For local OpenAI-compatible servers (llama.cpp, Ollama), use the
            # OpenAI SDK directly — LiteLLM's CustomStreamWrapper has issues with
            # synchronous iteration on Windows.
            from openai import OpenAI

            # A bare host gets the conventional /v1; a base that names a path is used as given.
            base_url = local_openai_base.rstrip("/")
            if not urlparse(base_url).path:
                base_url += "/v1"
            client = OpenAI(base_url=base_url, api_key=self.api_key)
            # The "openai/" routing prefix is LiteLLM's; the server knows the model by its own id.
            model = self.model.removeprefix(f"{self.provider}/")
            return lambda **request: client.chat.completions.create(model=model, **request)

        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        from litellm import completion

        return self._litellm_stream(completion)

    def _litellm_stream(self, completion: Callable) -> Callable[..., Iterable[Any]]:
        # LiteLLM routes on the "provider/model" form; a bare name only works for
        # the few models it happens to recognise.
        model = self.model if self.model.startswith(f"{self.provider}/") else f"{self.provider}/{self.model}"
        fixed: Dict[str, Any] = {"model": model, "num_retries": 0}
        if self.api_key:
            fixed["api_key"] = self.api_key
        if self.api_base:
            fixed["api_base"] = self.api_base
        return lambda **request: completion(**fixed, **request)

    @staticmethod
    def _is_loopback(api_base: str) -> bool:
        hostname = (urlparse(api_base).hostname or "").lower()
        if hostname == "localhost":
            return True
        try:
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False

    @staticmethod
    def _validate_api_base(api_base: str) -> None:
        if not isinstance(api_base, str) or len(api_base) > 2_048:
            raise ProviderError("api_base must be a URL up to 2048 characters.")
        parsed = urlparse(api_base)
        if not parsed.hostname or parsed.username or parsed.password:
            raise ProviderError("api_base must be an absolute URL without credentials.")
        if parsed.scheme == "https":
            return
        if parsed.scheme != "http":
            raise ProviderError("api_base must use HTTPS, or HTTP for loopback only.")

        # ASVS 12.3.1 / 12.3.2: custom remote provider endpoints must use
        # certificate-validated HTTPS; plaintext is limited to local runtimes.
        if not LiteLLMProvider._is_loopback(api_base):
            raise ProviderError("Plaintext provider endpoints are allowed on loopback only.")

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        """Read ``name`` from a response part, whether the SDK returned a dict or an object."""
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def _infer(
        self,
        conversation: Conversation,
        *,
        session_state: Optional["SessionState"],
        tool_choice: ToolChoiceMode = "auto",
        **kwargs,
    ) -> Iterator[StreamItem]:
        unexpected = set(kwargs) - self._allowed_generation_params
        if unexpected:
            raise LLMError("Unsupported generation parameters: " + ", ".join(sorted(unexpected)))
        if not isinstance(tool_choice, (str, dict)):
            raise LLMError("tool_choice must be a string or object.")
        if isinstance(tool_choice, str) and tool_choice not in {"auto", "none", "required"}:
            raise LLMError("Unsupported tool_choice value.")

        request: Dict[str, Any] = {
            "messages": conversation.to_openai_messages(),
            "stream": True,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        tools = [tool.to_dict() for tool in conversation.tools.values()]
        if tools and tool_choice != "none":
            request["tools"] = tools
            request["tool_choice"] = tool_choice

        yield from self._decode_stream(self._create_stream(**request))

    def _decode_stream(self, stream: Iterable[Any]) -> Iterator[StreamItem]:
        """Yield text as it streams, then the tool calls reassembled from their fragments."""
        calls: List[Dict[str, Any]] = []  # in order of arrival
        current: Dict[int, Dict[str, Any]] = {}  # the call each index is still adding to
        for chunk in stream:
            choices = self._field(chunk, "choices") or ()
            delta = self._field(choices[0], "delta") if choices else None
            if delta is None:
                continue

            content = self._field(delta, "content")
            if isinstance(content, str) and content:
                yield content
            # Reasoning (``reasoning_content``) is the model thinking, not its reply: it is not spoken.

            for part in self._field(delta, "tool_calls") or ():
                index = self._field(part, "index")
                if index is None:
                    index = 0
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= self._MAX_TOOL_CALL_INDEX:
                    raise LLMError("Provider returned an invalid tool call index.")
                call_id = self._field(part, "id")
                call_id = call_id if isinstance(call_id, str) and call_id else None
                entry = current.get(index)
                # Some providers number every call 0: a different id means another call, not more of this one.
                if entry is None or (call_id and entry["id"] and call_id != entry["id"]):
                    if len(calls) > self._MAX_TOOL_CALL_INDEX:
                        raise LLMError("Provider returned too many tool calls.")
                    entry = {"id": None, "name": "", "arguments": "", "object": None}
                    calls.append(entry)
                    current[index] = entry
                if call_id:
                    entry["id"] = call_id
                function = self._field(part, "function") or {}
                name = self._field(function, "name")
                arguments = self._field(function, "arguments")
                if isinstance(name, str) and name != entry["name"]:  # a provider may repeat the name on every delta
                    entry["name"] += name
                    if len(entry["name"]) > self._MAX_TOOL_NAME_CHARS:
                        raise LLMError(f"Tool name exceeds {self._MAX_TOOL_NAME_CHARS} characters.")
                if isinstance(arguments, dict):
                    entry["object"] = arguments
                elif isinstance(arguments, str):
                    entry["arguments"] += arguments
                    # ASVS 2.2.1: bound model-generated serialized tool input.
                    if len(entry["arguments"]) > self._MAX_TOOL_ARGUMENT_CHARS:
                        raise LLMError(f"Tool arguments exceed {self._MAX_TOOL_ARGUMENT_CHARS} characters.")

        for entry in calls:
            if entry["name"]:
                yield self._assemble_call(entry)

    def _assemble_call(self, entry: Dict[str, Any]) -> ToolCall | MalformedToolCall:
        arguments = entry["object"]
        if arguments is None:
            try:
                arguments = json.loads(entry["arguments"]) if entry["arguments"] else {}
            except json.JSONDecodeError:
                arguments = None
        # ASVS 1.5.2 / 15.3.5: accept only a JSON object as tool input.
        if not isinstance(arguments, dict):
            self.logger.warning("Provider returned malformed arguments for tool %s.", entry["name"])
            return MalformedToolCall(raw=entry["arguments"], name=entry["name"])
        if entry["id"]:
            return ToolCall(name=entry["name"], arguments=arguments, id=entry["id"])
        return ToolCall(name=entry["name"], arguments=arguments)


class LLMProviders:
    """Builds the LLM selected by configuration."""

    BACKENDS = ("local", "litellm")

    @classmethod
    def create(cls, backend: str, model: str, config: Optional[AppConfig] = None) -> LLM:
        """Create the provider for ``backend``.

        ``model`` names the entry under ``llm.local`` for the local backend;
        the LiteLLM backend takes its model from ``llm.litellm``.
        """
        llm_config = (config or get_config()).llm
        if backend == "local":
            entry = llm_config.local.get(model)
            if entry is None:
                raise ProviderError(f"Missing local model config for '{model}'.")
            return GGUFLLM.from_config(model, entry)
        if backend == "litellm":
            if llm_config.litellm is None:
                raise ProviderError("The litellm backend needs an `llm.litellm` configuration section.")
            return LiteLLMProvider(**llm_config.litellm.model_dump())
        raise ProviderError(f"Unknown LLM backend: {backend!r}. Expected one of: {', '.join(cls.BACKENDS)}.")
