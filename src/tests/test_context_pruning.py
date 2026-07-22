"""Context pruning tests for DropOldestStrategy."""
import unittest
from unittest.mock import MagicMock, PropertyMock
from voxpipe.llm.context import DropOldestStrategy
from voxpipe.llm.conversation import Conversation


class _MockLLM:
    def __init__(self, n_ctx=100, max_tokens=10):
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.logger = MagicMock()

    def count_tokens(self, text):
        return len(text) + 4  # +4 overhead per message


class TestContextPruning(unittest.TestCase):

    def test_prunes_when_over_token_limit(self):
        llm = _MockLLM(n_ctx=50, max_tokens=10)
        strat = DropOldestStrategy(max_turns=20)
        conv = Conversation()
        for i in range(6):
            conv.add_user_message(f"msg{i}")
        strat.trim(conv, llm)
        self.assertLessEqual(len(conv.messages), 5)

    def test_no_pruning_when_under_limit(self):
        llm = _MockLLM(n_ctx=100, max_tokens=10)
        strat = DropOldestStrategy(max_turns=20)
        conv = Conversation()
        for i in range(3):
            conv.add_user_message(f"msg{i}")
        strat.trim(conv, llm)
        self.assertEqual(len(conv.messages), 3)

    def test_prunes_when_over_turn_limit(self):
        llm = _MockLLM(n_ctx=9999, max_tokens=10)
        strat = DropOldestStrategy(max_turns=2)
        conv = Conversation()
        for i in range(5):
            conv.add_user_message(f"msg{i}")
        strat.trim(conv, llm)
        self.assertLessEqual(len(conv.messages), 4)

    def test_keeps_at_least_one_message(self):
        llm = _MockLLM(n_ctx=10, max_tokens=5)
        strat = DropOldestStrategy(max_turns=1)
        conv = Conversation()
        conv.add_user_message("hello")
        strat.trim(conv, llm)
        self.assertGreaterEqual(len(conv.messages), 1)

    def test_empty_conversation(self):
        llm = _MockLLM(n_ctx=50, max_tokens=10)
        strat = DropOldestStrategy(max_turns=5)
        conv = Conversation()
        strat.trim(conv, llm)
        self.assertEqual(len(conv.messages), 0)


if __name__ == "__main__":
    unittest.main()
