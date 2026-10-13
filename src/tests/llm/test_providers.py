import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from voxpipe.core.config import load_config
from voxpipe.core.exceptions import LLMError, ProviderError
from voxpipe.core.models import LiteLLMConfig
from voxpipe.llm.conversation import Conversation
from voxpipe.llm.decoders import GemmaE2BDecoder
from voxpipe.llm.model import GGUFLLM, LiteLLMProvider, LLMProviders
from voxpipe.llm.state import SessionState
from voxpipe.llm.tools import MalformedToolCall, Tool, ToolCall


def text_chunk(content):
    return {"choices": [{"delta": {"content": content}}]}


def tool_chunk(index=0, name=None, arguments=None, call_id=None):
    function = {k: v for k, v in (("name", name), ("arguments", arguments)) if v is not None}
    part = {"index": index, "function": function}
    if call_id:
        part["id"] = call_id
    return {"choices": [{"delta": {"tool_calls": [part]}}]}


class FakeLlama:
    """Stands in for llama_cpp.Llama, recording what the provider asks of it."""

    def __init__(self, reply=("This is a test.",)):
        self.reply = reply
        self.requests = []
        self.log = []

    def create_chat_completion(self, **request):
        self.requests.append(request)
        return iter([text_chunk(piece) for piece in self.reply])

    def tokenize(self, data):
        return data.split()

    def save_state(self):
        self.log.append("save")
        return f"state-{len(self.log)}"

    def load_state(self, state):
        self.log.append(f"load {state}")

    def reset(self):
        self.log.append("reset")


class TestGGUFLLM(unittest.TestCase):
    def _llm(self, model=None, **kwargs):
        return GGUFLLM(model or FakeLlama(), n_ctx=4096, max_tokens=128, **kwargs)

    def test_streams_content(self):
        llm = self._llm(FakeLlama(reply=("This is ", "a test.")))
        conversation = Conversation()
        conversation.add_user_message("Hello")
        self.assertEqual(list(llm(conversation)), ["This is ", "a test."])

    def test_request_carries_messages_and_token_limit(self):
        model = FakeLlama()
        conversation = Conversation()
        conversation.set_system_message("sys")
        conversation.add_user_message("Hello")
        list(self._llm(model)(conversation, temperature=0.1))
        request = model.requests[0]
        self.assertEqual(request["messages"], conversation.messages)
        self.assertEqual(request["max_tokens"], 128)
        self.assertEqual(request["temperature"], 0.1)
        self.assertTrue(request["stream"])
        self.assertEqual(request["tool_choice"], "none", "no tools registered")

    def test_tools_are_offered_unless_tool_choice_is_none(self):
        model = FakeLlama()
        conversation = Conversation()
        conversation.tools = [Tool(name="t", description="d")]
        llm = self._llm(model)
        list(llm(conversation))
        list(llm(conversation, tool_choice="none"))
        self.assertEqual(model.requests[0]["tools"], [conversation.tools["t"].to_dict()])
        self.assertEqual(model.requests[0]["tool_choice"], "auto")
        self.assertNotIn("tools", model.requests[1])
        self.assertEqual(model.requests[1]["tool_choice"], "none")

    def test_decoder_is_applied(self):
        model = FakeLlama(reply=("<|tool_call>", "call:ping{}", "<tool_call|>"))
        llm = self._llm(model, decoder=GemmaE2BDecoder())
        self.assertEqual(list(llm(Conversation())), [ToolCall(name="ping", arguments={})])

    def test_count_tokens_uses_the_model_tokenizer(self):
        self.assertEqual(self._llm().count_tokens("one two three"), 3)

    def test_lock_is_released_when_the_stream_is_abandoned(self):
        llm = self._llm(FakeLlama(reply=("a", "b", "c")))
        stream = llm(Conversation())
        next(stream)
        stream.close()
        self.assertFalse(llm._lock.locked())

    def test_kv_state_follows_the_session(self):
        model = FakeLlama()
        llm = self._llm(model)
        first, second = SessionState(), SessionState()
        conversation = Conversation()

        list(llm(conversation, session_state=first))
        list(llm(conversation, session_state=first))
        self.assertEqual(model.log, ["reset"], "same session: the live cache is reused")

        list(llm(conversation, session_state=second))
        self.assertEqual(model.log, ["reset", "save", "reset"])
        self.assertEqual(first.model.runtime, {"llama_state": "state-2"})

        list(llm(conversation, session_state=first))
        self.assertEqual(model.log[3:], ["save", "reset", "load state-2"])

    def test_one_shot_calls_park_the_session_but_keep_no_state_of_their_own(self):
        model = FakeLlama()
        llm = self._llm(model)
        state = SessionState()
        list(llm(Conversation(), session_state=state))
        list(llm(Conversation(), session_state=None))
        list(llm(Conversation(), session_state=state))
        self.assertEqual(model.log, ["reset", "save", "reset", "reset", "load state-2"])

    def test_from_config_requires_a_configured_model(self):
        with self.assertRaises(ProviderError):
            GGUFLLM.from_config("NoSuchModel")

    def test_from_config_rejects_unknown_decoder_before_loading(self):
        entry = load_config().llm.local["Qwen3"].model_copy(update={"decoder": "nonsense"})
        with self.assertRaises(ProviderError):
            GGUFLLM.from_config("Qwen3", entry)


class TestLiteLLMProvider(unittest.TestCase):
    def _provider(self, chunks=(), **kwargs):
        completion = MagicMock(return_value=iter(chunks))
        kwargs.setdefault("model", "test")
        kwargs.setdefault("provider", "ollama")
        return LiteLLMProvider(completion_fn=completion, **kwargs), completion

    def test_empty_conversation(self):
        llm, _ = self._provider()
        self.assertEqual("".join(llm(Conversation())), "")

    def test_request(self):
        llm, completion = self._provider([text_chunk("This is a test.")], api_base="http://localhost:11434")
        conversation = Conversation()
        conversation.add_user_message("Hello")

        self.assertEqual("".join(llm(conversation, temperature=0.3)), "This is a test.")

        request = completion.call_args.kwargs
        self.assertEqual(request["model"], "ollama/test", "LiteLLM routes on the provider prefix")
        self.assertEqual(request["api_base"], "http://localhost:11434")
        self.assertEqual(request["messages"], [{"role": "user", "content": "Hello"}])
        self.assertEqual(request["timeout"], 60.0)
        self.assertEqual(request["num_retries"], 0)
        self.assertEqual(request["max_tokens"], 1024)
        self.assertEqual(request["temperature"], 0.3)
        self.assertTrue(request["stream"])
        self.assertNotIn("tools", request)

    def test_model_prefixing(self):
        for provider, model, expected in (
            ("openai", "gpt-test", "openai/gpt-test"),
            ("gemini", "gemini-test", "gemini/gemini-test"),
            ("ollama", "ollama/qwen3:latest", "ollama/qwen3:latest"),
        ):
            with self.subTest(provider=provider):
                llm, completion = self._provider(model=model, provider=provider, api_key="test-key")
                self.assertEqual(llm.model, model)
                list(llm(Conversation()))
                self.assertEqual(completion.call_args.kwargs["model"], expected)

    def test_tools_and_tool_history_use_openai_format(self):
        llm, completion = self._provider()
        conversation = Conversation()
        conversation.tools = [Tool(name="lookup", description="d")]
        conversation.add_user_message("go")
        call = ToolCall(name="lookup", arguments={"q": "x"}, id="call_1")
        conversation.add_assistant_message("", tool_calls=[call])
        conversation.add_tool_message('{"a": 1}', tool_call=call)

        list(llm(conversation))
        request = completion.call_args.kwargs
        self.assertEqual(request["tools"], [conversation.tools["lookup"].to_dict()])
        self.assertEqual(request["tool_choice"], "auto")
        self.assertEqual(request["messages"], conversation.to_openai_messages())
        self.assertEqual(request["messages"][2]["tool_call_id"], "call_1")

        list(llm(conversation, tool_choice="none"))
        self.assertNotIn("tools", completion.call_args.kwargs)

    def test_invalid_requests_are_rejected(self):
        llm, _ = self._provider()
        for kwargs in ({"top_k": 5}, {"tool_choice": "sometimes"}, {"tool_choice": 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(LLMError):
                list(llm(Conversation(), **kwargs))

    def test_stream_reassembles_fragmented_tool_calls(self):
        llm, _ = self._provider([
            text_chunk("Checking. "),
            tool_chunk(name="pi", arguments='{"value":', call_id="call_abc"),
            tool_chunk(name="ng", arguments="1}"),
            tool_chunk(index=1, name="pong", arguments=""),
        ])
        items = list(llm(Conversation()))
        self.assertEqual(
            items, ["Checking. ", ToolCall(name="ping", arguments={"value": 1}), ToolCall(name="pong", arguments={})]
        )
        self.assertEqual(items[1].id, "call_abc", "the provider's call id is kept for the reply")

    def test_malformed_tool_arguments_are_not_executable(self):
        for arguments in ("[1, 2]", "{not json", '"text"'):
            with self.subTest(arguments=arguments):
                llm, _ = self._provider([tool_chunk(name="ping", arguments=arguments)])
                self.assertEqual(list(llm(Conversation())), [MalformedToolCall(raw=arguments, name="ping")])

    def test_oversized_tool_calls_are_rejected(self):
        cases = [
            [tool_chunk(name="x" * 65, arguments="{}")],
            [tool_chunk(name="ping", arguments="x" * 65_537)],
            [tool_chunk(index=999, name="ping", arguments="{}")],
        ]
        for chunks in cases:
            llm, _ = self._provider(chunks)
            with self.assertRaises(LLMError):
                list(llm(Conversation()))

    def test_reasoning_is_not_spoken(self):
        llm, _ = self._provider([
            {"choices": [{"delta": {"reasoning_content": "The user greets me, so I should..."}}]},
            text_chunk("Hello!"),
        ])
        self.assertEqual(list(llm(Conversation())), ["Hello!"])

    def test_object_style_chunks(self):
        delta = types.SimpleNamespace(content="Hi", tool_calls=None)
        chunk = types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])
        empty = types.SimpleNamespace(choices=[])
        llm, _ = self._provider([empty, chunk])
        self.assertEqual(list(llm(Conversation())), ["Hi"])

    def test_constructor_validation(self):
        bad = [
            dict(provider="anthropic"),
            dict(model=""),
            dict(model="openai/gpt", provider="ollama"),
            dict(timeout=0),
            dict(n_ctx=10),
            dict(max_tokens=0),
            dict(api_base="http://192.0.2.10:11434"),  # plaintext to a remote host
            dict(api_base="ftp://localhost"),
            dict(api_base="https://user:pw@example.com"),
            dict(provider="openai"),  # no key and no local endpoint
            dict(provider="openai", api_key="   "),
        ]
        with patch.dict(os.environ, {}, clear=True):
            for kwargs in bad:
                with self.subTest(kwargs=kwargs), self.assertRaises(ProviderError):
                    self._provider(**kwargs)
            with self.assertRaises(ProviderError):
                LiteLLMProvider(model="test", provider="ollama", completion_fn="not callable")

    def test_api_key_resolution(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "from-env"}, clear=True):
            self.assertEqual(self._provider(provider="openai")[0].api_key, "from-env")
            self.assertEqual(self._provider(provider="openai", api_key="explicit")[0].api_key, "explicit")
        with patch.dict(os.environ, {}, clear=True):
            local, completion = self._provider(provider="gemini", api_base="http://127.0.0.1:8080")
            self.assertEqual(local.api_key, "sk-no-key-required")
            self.assertIsNone(self._provider(provider="ollama")[0].api_key)

    def test_litellm_import_uses_local_cost_map_by_default(self):
        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock(return_value=iter([]))

        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"litellm": fake_litellm}):
            llm = LiteLLMProvider(model="test", provider="ollama")
            self.assertEqual(os.environ["LITELLM_LOCAL_MODEL_COST_MAP"], "True")
            list(llm(Conversation()))
        self.assertEqual(fake_litellm.completion.call_args.kwargs["model"], "ollama/test")

    def test_local_openai_endpoints_use_the_openai_sdk(self):
        client = MagicMock()
        client.chat.completions.create.return_value = iter([text_chunk("hi")])
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = MagicMock(return_value=client)

        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"openai": fake_openai}):
            llm = LiteLLMProvider(model="local-model", provider="openai", api_base="http://localhost:8080")
            self.assertEqual(list(llm(Conversation())), ["hi"])

        fake_openai.OpenAI.assert_called_once_with(base_url="http://localhost:8080/v1", api_key="sk-no-key-required")
        self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], "local-model")


class TestLLMProviders(unittest.TestCase):
    def _config(self, **litellm):
        config = load_config()
        section = LiteLLMConfig(**litellm) if litellm else None
        return config.model_copy(update={"llm": config.llm.model_copy(update={"litellm": section})})

    def test_litellm_backend_is_built_from_its_config_section(self):
        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock()
        config = self._config(provider="ollama", model="test", api_base="http://127.0.0.1:11434", max_tokens=256)

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            provider = LLMProviders.create("litellm", "ignored", config=config)

        self.assertIsInstance(provider, LiteLLMProvider)
        self.assertEqual((provider.model, provider.api_base, provider.max_tokens), ("test", "http://127.0.0.1:11434", 256))

    def test_litellm_backend_needs_its_section(self):
        with self.assertRaises(ProviderError):
            LLMProviders.create("litellm", "ignored", config=self._config())

    def test_local_backend_needs_a_configured_model(self):
        with self.assertRaises(ProviderError):
            LLMProviders.create("local", "NoSuchModel", config=self._config())

    def test_unknown_backend(self):
        for backend in ("", "__class__", "remote"):
            with self.subTest(backend=backend), self.assertRaises(ProviderError):
                LLMProviders.create(backend, "Qwen3", config=self._config())


if __name__ == "__main__":
    unittest.main()
