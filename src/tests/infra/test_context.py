"""Tests for context pruning strategy and consumer-producer base."""
import unittest
from unittest.mock import MagicMock, PropertyMock


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


class TestConsumerProducer(unittest.TestCase):
    def test_enable_disable(self):
        from voxpipe.streaming.splitter import ConsumerProducer
        class Impl(ConsumerProducer):
            def _consume(self, value): self.last = value
            def _produce(self): yield from [1, 2, 3]
        impl = Impl()
        impl("hello")
        self.assertEqual(impl.last, "hello")
        impl.disable()
        impl("world")
        self.assertEqual(impl.last, "hello")
        impl.enable()
        impl("again")
        self.assertEqual(impl.last, "again")

    def test_passthrough(self):
        from voxpipe.streaming.splitter import ConsumerProducer
        class Impl(ConsumerProducer):
            def _consume(self, value): self.last = value
            def _produce(self): yield from []
        impl = Impl()
        impl.disable_w_passthrough("fixed")
        impl("input")
        self.assertEqual(impl.last, "fixed")

    def test_iter(self):
        from voxpipe.streaming.splitter import ConsumerProducer
        class Impl(ConsumerProducer):
            def _consume(self, value): pass
            def _produce(self): yield from [10, 20]
        impl = Impl()
        self.assertEqual(list(impl), [10, 20])
