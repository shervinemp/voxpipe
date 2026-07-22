"""Tests for context pruning strategy and consumer-producer base."""
import unittest
from unittest.mock import MagicMock, PropertyMock


class TestDropOldestStrategy(unittest.TestCase):
    def setUp(self):
        from voxpipe.llm.context import DropOldestStrategy
        self.strat = DropOldestStrategy(max_turns=3)

    def _conv(self, n_msgs):
        from voxpipe.llm.conversation import Conversation
        c = Conversation()
        for i in range(n_msgs):
            c.add_user_message(f"msg{i}")
        return c

    def _llm(self):
        m = MagicMock()
        m.count_tokens.return_value = 10
        type(m).n_ctx = PropertyMock(return_value=4096)
        type(m).max_tokens = PropertyMock(return_value=512)
        return m

    def test_does_not_trim_below_max_turns(self):
        c = self._conv(3)
        self.strat.trim(c, self._llm())
        self.assertEqual(len(c._messages), 3)

    def test_trims_to_max_turns(self):
        c = self._conv(10)
        self.strat.trim(c, self._llm())
        self.assertLessEqual(c.visible_count(), 6)

    def test_empty_conversation(self):
        from voxpipe.llm.conversation import Conversation
        c = Conversation()
        self.strat.trim(c, self._llm())
        self.assertEqual(len(c._messages), 0)


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
