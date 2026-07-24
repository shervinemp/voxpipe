"""Context pruning tests for MemoryHandler and MaxTurnsStrategy."""
import unittest
from voxpipe.llm.context import MaxTurnsStrategy, MemoryHandler
from voxpipe.llm.conversation import Conversation


class TestContextPruning(unittest.TestCase):

    def test_no_pruning_when_under_limit(self):
        handler = MemoryHandler(strategy=MaxTurnsStrategy(max_turns=20))
        conv = Conversation()
        for i in range(3):
            conv.add_user_message(f"msg{i}")
        handler.pre_process(conv)
        self.assertEqual(conv.visible_count(), 3)

    def test_prunes_when_over_turn_limit(self):
        handler = MemoryHandler(strategy=MaxTurnsStrategy(max_turns=2), amortize_k=1)
        conv = Conversation()
        for i in range(5):
            conv.add_user_message(f"msg{i}")
        handler.pre_process(conv)
        self.assertLessEqual(conv.visible_count(), 4)

    def test_keeps_at_least_one_message(self):
        handler = MemoryHandler(strategy=MaxTurnsStrategy(max_turns=1), amortize_k=1)
        conv = Conversation()
        conv.add_user_message("hello")
        handler.pre_process(conv)
        self.assertGreaterEqual(conv.visible_count(), 1)

    def test_empty_conversation(self):
        handler = MemoryHandler(strategy=MaxTurnsStrategy(max_turns=5))
        conv = Conversation()
        handler.pre_process(conv)
        self.assertEqual(conv.visible_count(), 0)


if __name__ == "__main__":
    unittest.main()
