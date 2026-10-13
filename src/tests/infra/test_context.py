"""Tests for the max-turns eviction strategy."""
import unittest


class TestMaxTurnsStrategy(unittest.TestCase):
    def setUp(self):
        from voxpipe.llm.context import MaxTurnsStrategy
        self.strat = MaxTurnsStrategy(max_turns=3)

    def _conv(self, n_msgs):
        from voxpipe.llm.conversation import Conversation
        c = Conversation()
        for i in range(n_msgs):
            c.add_user_message(f"msg{i}")
        return c

    def test_does_not_blip_below_max_turns(self):
        c = self._conv(3)
        self.assertFalse(self.strat(c))

    def test_blips_above_max_turns(self):
        c = self._conv(10)
        self.assertTrue(self.strat(c))

    def test_empty_conversation(self):
        from voxpipe.llm.conversation import Conversation
        c = Conversation()
        self.assertFalse(self.strat(c))
