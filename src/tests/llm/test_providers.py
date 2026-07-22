import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from voxpipe.core.exceptions import ProviderError

from voxpipe.llm.conversation import Conversation
from voxpipe.llm.model import (
    LLMProviders,
    LiteLLMProvider,
    GGUFLLM,
)
from voxpipe.llm.tools import ToolCall


class TestLLM(unittest.TestCase):
    @patch("voxpipe.llm.model.os.path.exists", return_value=True)
    @patch("voxpipe.llm.model.GGUFLLM.__init__", return_value=None)
    @patch("voxpipe.llm.model.Llama", create=True)
    def test_gguf_llm(self, mock_llama, mock_init, mock_exists):
        """
        Test a GGUF-backed LLM.
        """
        # Mock the Llama model
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter(
            [{"choices": [{"delta": {"content": "This is a test."}}]}]
        )
        mock_llama.return_value = mock_model

        # Initialize the LLM
        llm = GGUFLLM("Qwen3")
        llm.model = mock_model
        llm.max_tokens = 128
        llm._last_state = None
        llm._lock = MagicMock()
        llm._parse = MagicMock(side_effect=lambda x: x)
        llm.logger = MagicMock()

        # Create a conversation
        conversation = Conversation()
        conversation.add_user_message("Hello")

        # Get the response
        response = "".join(llm._infer(conversation, session_state={}))

        # Check the response
        self.assertEqual(response, "This is a test.")

    def test_empty_conversation(self):
        """
        Test that the LLM returns an empty string for an empty conversation.
        """
        completion = MagicMock(return_value=iter([]))

        # Initialize the LLM
        llm = LiteLLMProvider(model="test", provider="ollama", completion_fn=completion)

        # Create an empty conversation
        conversation = Conversation()

        # Get the response
        response = "".join(llm(conversation))

        # Check the response
        self.assertEqual(response, "")

    @patch("voxpipe.llm.model.os.path.exists", return_value=True)
    @patch("voxpipe.llm.model.GGUFLLM.__init__", return_value=None)
    @patch("voxpipe.llm.model.Llama", create=True)
    def test_qwen_llm(self, mock_llama, mock_init, mock_exists):
        """
        Test the QwenLLM.
        """
        # Mock the Llama model
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter(
            [{"choices": [{"delta": {"content": "This is a test."}}]}]
        )
        mock_llama.return_value = mock_model

        # Initialize the LLM
        llm = GGUFLLM("Qwen3")
        llm.model = mock_model
        llm.max_tokens = 128
        llm._last_state = None
        llm._lock = MagicMock()
        llm._parse = MagicMock(side_effect=lambda x: x)
        llm.logger = MagicMock()

        # Create a conversation
        conversation = Conversation()
        conversation.add_user_message("Hello")

        # Get the response
        response = "".join(llm._infer(conversation, session_state={}))

        # Check the response
        self.assertEqual(response, "This is a test.")

    def test_ollama_llm(self):
        """
        Test the OllamaLLM.
        """
        completion = MagicMock(
            return_value=iter(
                [
                    {
                        "choices": [
                            {"delta": {"content": "This is a test."}}
                        ]
                    }
                ]
            )
        )

        # Initialize the LLM
        llm = LiteLLMProvider(
            model="test", provider="ollama",
            api_base="http://localhost:11434", completion_fn=completion,
        )

        # Create a conversation
        conversation = Conversation()
        conversation.add_user_message("Hello")

        # Get the response
        response = "".join(llm(conversation))

        # Check the response
        self.assertEqual(response, "This is a test.")
        request = completion.call_args.kwargs
        self.assertEqual(request["model"], "test")
        self.assertEqual(request["api_base"], "http://localhost:11434")
        self.assertEqual(request["timeout"], 60.0)
        self.assertEqual(request["num_retries"], 0)

    def test_litellm_model_prefixing(self):
        completion = MagicMock(return_value=iter([]))
        openai = LiteLLMProvider(
            model="gpt-test", provider="openai",
            api_key="test-openai-key", completion_fn=completion,
        )
        gemini = LiteLLMProvider(
            model="gemini-test", provider="gemini",
            api_key="test-gemini-key", completion_fn=completion,
        )

        self.assertEqual(openai.model, "gpt-test")
        self.assertEqual(gemini.model, "gemini-test")

    def test_remote_plaintext_provider_endpoint_is_rejected(self):
        with self.assertRaises(ProviderError):
            LiteLLMProvider(
                model="test", provider="ollama",
                api_base="http://192.0.2.10:11434",
                completion_fn=MagicMock(),
            )

    def test_litellm_stream_reassembles_fragmented_tool_calls(self):
        completion = MagicMock(
            return_value=iter(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "pi",
                                                "arguments": '{"value":',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "ng",
                                                "arguments": "1}",
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                ]
            )
        )
        llm = LiteLLMProvider(model="test", provider="ollama", completion_fn=completion)

        result = list(llm._infer(Conversation(), session_state={}))

        self.assertEqual(result, [ToolCall(name="ping", arguments={"value": 1})])

    def test_litellm_malformed_tool_arguments_are_not_executed(self):
        completion = MagicMock(
            return_value=iter(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "ping",
                                                "arguments": "[1, 2]",
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            )
        )
        llm = LiteLLMProvider(model="test", provider="ollama", completion_fn=completion)

        from voxpipe.core.exceptions import LLMError
        with self.assertRaises(LLMError):
            list(llm._infer(Conversation(), session_state={}))

    def test_provider_factory_uses_allowlisted_provider_configuration(self):
        completion = MagicMock(return_value=iter([]))

        with patch("voxpipe.llm.model.config.get") as mock_config:
            mock_config.side_effect = lambda key, default=None: {
                "llm.litellm": {
                    "provider": "ollama",
                    "model": "test",
                    "api_base": "http://127.0.0.1:11434",
                    "completion_fn": completion,
                },
            }.get(key, default)
            provider = LLMProviders.create("litellm", "test")

        self.assertIsInstance(provider, LiteLLMProvider)
        self.assertEqual(provider.model, "test")
        with self.assertRaises(ProviderError):
            LLMProviders.get("__class__")

    def test_litellm_import_uses_local_cost_map_by_default(self):
        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(sys.modules, {"litellm": fake_litellm}),
        ):
            LiteLLMProvider(model="test", provider="ollama")
            self.assertEqual(
                os.environ["LITELLM_LOCAL_MODEL_COST_MAP"], "True"
            )
