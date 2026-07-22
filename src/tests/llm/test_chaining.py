"""Tool chaining, RAG integration, and conversation state tests."""
import unittest
from unittest.mock import MagicMock, patch

from voxpipe.llm.tools import ToolCall, Tool, ToolResult


def _make_mock_llm(tokens_per_call, decoder=None):
    """Create a mock LLM that yields different tokens on each _infer call.

    tokens_per_call is a list of lists: tokens_per_call[0] for the first
    infer, tokens_per_call[1] for the second, etc.
    """
    from voxpipe.llm.decoders import GeneralDecoder
    from voxpipe.llm.model import LLM
    from voxpipe.llm.context import DropOldestStrategy
    _decoder = decoder or GeneralDecoder()

    class MockLLM(LLM):
        decoder = _decoder
        _call_count = 0

        def _infer(self, conversation, *, session_state, **kwargs):
            idx = self._call_count
            self._call_count += 1
            tokens = tokens_per_call[min(idx, len(tokens_per_call) - 1)]
            yield from tokens

        def create_context_strategy(self, max_turns=20):
            return DropOldestStrategy(max_turns)

        def count_tokens(self, text):
            return max(1, len(text) // 2)

    llm = MockLLM()
    llm.logger = MagicMock()
    return llm


def _collect(gen):
    text, calls = [], []
    for item in gen:
        if isinstance(item, ToolCall):
            calls.append((item.name, item.arguments))
        elif isinstance(item, str):
            text.append(item)
    return text, calls


class TestToolChaining(unittest.TestCase):
    """Session tool chaining behavior with max_tool_iterations."""

    def _check_tool_called(self, sess, expected_name, expected_args, tool_results):
        """Verify a tool was dispatched by checking side effects + conversation."""
        self.assertEqual(len(tool_results), 1,
                         f"tool {expected_name} should be called once")
        roles = [m["role"] for m in sess.conversation.messages]
        self.assertIn("tool", roles,
                      "tool result should be in conversation")

    def test_single_call_default(self):
        """max_tool_iterations=1: one auto pass, gather, then none pass."""
        from voxpipe.llm.session import Session
        from voxpipe.llm.conversation import Conversation

        tool_results = []
        def my_tool(query: str) -> ToolResult:
            tool_results.append(query)
            return ToolResult(result={"result": f"result for {query}"})

        # First pass: tool call. Second pass: answer.
        tokens = [
            ["Let me check...",
             "<|tool_call>", 'call:my_tool{query:<|"|>hello<|"|>}', "<tool_call|>"],
            ["Final answer."],
        ]
        llm = _make_mock_llm(tokens)
        conv = Conversation()
        conv.set_system_message("You are helpful.")
        conv.tools["my_tool"] = Tool.from_callable("my_tool", my_tool)
        conv.add_user_message("find hello")
        sess = Session(llm=llm, conversation=conv, max_tool_iterations=1)

        text, calls = _collect(sess())
        self.assertEqual(len(tool_results), 1, "tool should be called once")
        self.assertIn("Final", "".join(text))
        sess.close()

    def test_zero_disables_tools(self):
        """max_tool_iterations=0: tools disabled, answer-only pass."""
        from voxpipe.llm.session import Session
        from voxpipe.llm.conversation import Conversation

        tool_results = []
        def my_tool(query: str) -> ToolResult:
            tool_results.append(query)
            return ToolResult(result={"result": f"result for {query}"})

        tokens = [["Direct answer."]]
        llm = _make_mock_llm(tokens)
        conv = Conversation()
        conv.set_system_message("You are helpful.")
        conv.tools["my_tool"] = Tool.from_callable("my_tool", my_tool)
        conv.add_user_message("hello")
        sess = Session(llm=llm, conversation=conv, max_tool_iterations=0)

        text, calls = _collect(sess())
        self.assertEqual(len(tool_results), 0, "tool should NOT be called")
        self.assertIn("Direct", "".join(text))
        sess.close()

    def test_chaining_two_steps(self):
        """max_tool_iterations=2: auto→gather→auto→gather→none."""
        from voxpipe.llm.session import Session
        from voxpipe.llm.conversation import Conversation

        calls = []
        def chain_tool(query: str) -> ToolResult:
            calls.append(query)
            return ToolResult(result={"result": f"chain({query})"})

        # Pass 0: first tool call. Pass 1: second tool call. Pass 2: final answer.
        tokens = [
            ["<|tool_call>", 'call:chain_tool{query:<|"|>step1<|"|>}', "<tool_call|>"],
            ["<|tool_call>", 'call:chain_tool{query:<|"|>step2<|"|>}', "<tool_call|>"],
            ["Final chained answer."],
        ]
        llm = _make_mock_llm(tokens)
        conv = Conversation()
        conv.set_system_message("You are helpful.")
        conv.tools["chain_tool"] = Tool.from_callable("chain_tool", chain_tool)
        conv.add_user_message("chain")
        sess = Session(llm=llm, conversation=conv, max_tool_iterations=2)

        text, _ = _collect(sess())
        self.assertEqual(len(calls), 2, "tool should be called twice")
        self.assertEqual(calls, ["step1", "step2"])
        self.assertIn("chained", "".join(text))
        sess.close()

    def test_chaining_breaks_early_when_no_tool(self):
        """If model stops calling tools, loop breaks before max iterations."""
        from voxpipe.llm.session import Session
        from voxpipe.llm.conversation import Conversation

        calls = []
        def my_tool(query: str) -> ToolResult:
            calls.append(query)
            return ToolResult(result={"result": f"r({query})"})

        # Only the first pass produces a tool call.
        tokens = [
            ["<|tool_call>", 'call:my_tool{query:<|"|>only<|"|>}', "<tool_call|>"],
            ["Answer after first tool."],
        ]
        llm = _make_mock_llm(tokens)
        conv = Conversation()
        conv.set_system_message("You are helpful.")
        conv.tools["my_tool"] = Tool.from_callable("my_tool", my_tool)
        conv.add_user_message("hi")
        sess = Session(llm=llm, conversation=conv, max_tool_iterations=5)

        text, _ = _collect(sess())
        self.assertEqual(len(calls), 1, "should break after first gather returns empty")
        self.assertIn("Answer", "".join(text))
        sess.close()

    def test_tool_error_does_not_chain(self):
        """Tool error should not trigger another tool pass."""
        from voxpipe.llm.session import Session
        from voxpipe.llm.conversation import Conversation

        def broken_tool(q: str) -> str:
            raise ValueError("broken")

        tokens = [
            ["<|tool_call>", 'call:broken_tool{query:<|"|>x<|"|>}', "<tool_call|>"],
            ["Error recovery answer."],
        ]
        llm = _make_mock_llm(tokens)
        conv = Conversation()
        conv.set_system_message("You are helpful.")
        conv.tools["broken_tool"] = Tool.from_callable("broken_tool", broken_tool)
        conv.add_user_message("test")
        sess = Session(llm=llm, conversation=conv, max_tool_iterations=3)

        text, _ = _collect(sess())
        self.assertEqual(len([m for m in sess.conversation.messages if m["role"] == "tool"]), 2,
                         "tool error message + error instruction")
        sess.close()


class TestRAGToolIntegration(unittest.TestCase):
    """Session + RAG retrieve tool integration."""

    def _make_rag_conv(self):
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        rag_tool = Tool.from_callable("retrieve", lambda q: ToolResult(result={"evidence": "[graph] test evidence"}))
        rag_tool.instruction = "Call when the user asks about entities."
        conv.tools["retrieve"] = rag_tool
        conv.set_system_message(
            "You are a game assistant. "
            "- Call 'retrieve' when the user asks about entities."
        )
        return conv

    def test_retrieve_tool_used_on_knowledge_query(self):
        """Model should call retrieve for entity questions."""
        from voxpipe.llm.session import Session

        tool_results = []
        def retrieve(query) -> ToolResult:
            tool_results.append(query)
            return ToolResult(result={"evidence": "[graph] Elden Ring is a game"})

        tokens = [
            ["<|tool_call>", 'call:retrieve{query:<|"|>Elden Ring<|"|>}', "<tool_call|>"],
            ["The Elden Ring is a fictional artifact."],
        ]
        llm = _make_mock_llm(tokens)
        conv = self._make_rag_conv()
        # Replace the tool with one that tracks calls
        conv.tools["retrieve"] = Tool.from_callable("retrieve", retrieve)
        conv.add_user_message("What is the Elden Ring?")
        sess = Session(llm=llm, conversation=conv)

        text, _ = _collect(sess())
        self.assertEqual(len(tool_results), 1)
        self.assertIn("Elden Ring", "".join(text))
        sess.close()

    def test_retrieve_result_in_conversation(self):
        """Tool result should be added as tool message in conversation."""
        from voxpipe.llm.session import Session

        tokens = [
            ["<|tool_call>", 'call:retrieve{query:<|"|>test<|"|>}', "<tool_call|>"],
            ["Answer."],
        ]
        llm = _make_mock_llm(tokens)
        conv = self._make_rag_conv()
        conv.add_user_message("test query")
        sess = Session(llm=llm, conversation=conv)

        list(sess())

        roles = [m["role"] for m in conv.messages]
        self.assertIn("tool", roles, "tool result should be in conversation")
        sess.close()

    def test_retrieve_not_called_for_greeting(self):
        """Greetings should skip the retrieve tool."""
        from voxpipe.llm.session import Session

        tokens = [["Hello! How can I help you?"]]
        llm = _make_mock_llm(tokens)
        conv = self._make_rag_conv()
        conv.add_user_message("Hi")
        sess = Session(llm=llm, conversation=conv)

        text, calls = _collect(sess())
        self.assertEqual(calls, [])
        self.assertIn("Hello", "".join(text))
        sess.close()


class TestConversationState(unittest.TestCase):
    """Conversation history management across tool calls."""

    def test_messages_are_appended_in_order(self):
        """Conversation messages should be: system, user, tool, tool, assistant."""
        from voxpipe.llm.session import Session
        from voxpipe.llm.conversation import Conversation

        tokens = [
            ["<|tool_call>", 'call:test_tool{query:<|"|>q<|"|>}', "<tool_call|>"],
            ["The answer is 42."],
        ]
        llm = _make_mock_llm(tokens)
        conv = Conversation()
        conv.set_system_message("You are helpful.")
        conv.tools["test_tool"] = Tool.from_callable("test_tool", lambda q: ToolResult(result={"answer": "42"}))
        conv.add_user_message("what is the answer")
        sess = Session(llm=llm, conversation=conv)

        list(sess())

        roles = [m["role"] for m in conv.messages]
        expected = ["system", "user", "tool", "tool", "assistant"]
        self.assertEqual(roles, expected, f"got {roles}")
        sess.close()

    def test_context_pruning_across_turns(self):
        """Multiple turns should not exceed max_turns."""
        from voxpipe.llm.session import Session
        from voxpipe.llm.conversation import Conversation

        tokens = [["OK."]]
        llm = _make_mock_llm(tokens)
        conv = Conversation()
        conv.set_system_message("You are helpful.")
        # Set max_turns low to test pruning
        sess = Session(llm=llm, conversation=conv, max_turns=3)

        for i in range(5):
            conv.add_user_message(f"turn {i}")
            list(sess())

        # Should have system + at most max_turns*2 + 1 visible messages
        self.assertLessEqual(len(conv.messages), 7,
                             f"too many messages: {len(conv.messages)}")
        sess.close()
