import unittest

from voxpipe.llm.context import (
    CompositeStrategy,
    ContextHandler,
    MaxTurnsStrategy,
    MemoryHandler,
    TokenBudgetStrategy,
)
from voxpipe.llm.conversation import Conversation
from voxpipe.llm.session import Session
from voxpipe.llm.tools import Tool, ToolCall
from voxpipe.storage import MemoryStore, RAMStorage

from ._fakes import ScriptedLLM


def archived(storage):
    return [(record.meta["role"], record.content) for _, record in storage._entries]


class TestStrategies(unittest.TestCase):
    def _conv(self, *contents):
        conv = Conversation()
        for content in contents:
            conv.add_user_message(content)
        return conv

    def test_token_budget_evicts_oldest_until_it_fits(self):
        strategy = TokenBudgetStrategy(count_tokens=len, budget=30, per_message_overhead=0)
        self.assertEqual(strategy(self._conv("a" * 10, "b" * 10, "c" * 10)), 0)
        self.assertEqual(strategy(self._conv("a" * 10, "b" * 10, "c" * 10, "d" * 10)), 1)
        self.assertEqual(strategy(self._conv("a" * 25, "b" * 25, "c" * 10)), 2)

    def test_token_budget_never_evicts_the_newest_message(self):
        strategy = TokenBudgetStrategy(count_tokens=len, budget=5, per_message_overhead=0)
        self.assertEqual(strategy(self._conv("a" * 50)), 0)
        self.assertEqual(strategy(self._conv("a" * 50, "b" * 50)), 1)

    def test_token_budget_counts_system_prompt_tools_and_overhead(self):
        conv = self._conv("a" * 10, "b" * 10)
        self.assertEqual(TokenBudgetStrategy(len, budget=20, per_message_overhead=0)(conv), 0)
        self.assertEqual(TokenBudgetStrategy(len, budget=20, per_message_overhead=1)(conv), 1)
        conv.set_system_message("s" * 5)
        self.assertEqual(TokenBudgetStrategy(len, budget=20, per_message_overhead=0)(conv), 1)
        conv.set_system_message("")
        conv.tools = [Tool(name="t", description="d")]
        self.assertEqual(TokenBudgetStrategy(len, budget=20, per_message_overhead=0)(conv), 1)

    def test_token_budget_counts_recalled_context(self):
        conv = self._conv("a" * 10, "b" * 10)
        conv.last_message.context = "c" * 10
        self.assertEqual(TokenBudgetStrategy(len, budget=25, per_message_overhead=0)(conv), 1)

    def test_composite_takes_the_most_demanding(self):
        conv = self._conv(*["m"] * 6)
        strategy = CompositeStrategy(MaxTurnsStrategy(4), MaxTurnsStrategy(2), MaxTurnsStrategy(10))
        self.assertEqual(strategy(conv), 4)
        self.assertEqual(CompositeStrategy()(conv), 0)


class TestMemoryHandler(unittest.TestCase):
    def test_base_handler_hooks_do_nothing(self):
        handler = ContextHandler()
        conv = Conversation()
        self.assertFalse(handler.pre_process(conv))
        self.assertFalse(handler.post_process(conv))
        handler.finalize(conv)

    def test_trim_and_auto_archive(self):
        storage = RAMStorage()
        mem = MemoryStore(storage, session_id="test")
        handler = MemoryHandler(strategy=MaxTurnsStrategy(max_turns=2), memory_store=mem, amortize_k=1)

        conv = Conversation()
        for i in range(10):
            conv.add_user_message(f"message turn {i}")

        self.assertTrue(handler.pre_process(conv))
        self.assertEqual(conv.visible_count(), 2)
        self.assertEqual(len(conv.all_messages), 2, "amortize_k=1 drops hidden messages at once")
        self.assertEqual(archived(storage), [("user", f"message turn {i}") for i in range(8)])
        self.assertEqual(storage._entries[0][0], "session:test")

    def test_hard_trim_is_amortised(self):
        storage = RAMStorage()
        handler = MemoryHandler(MaxTurnsStrategy(max_turns=2), MemoryStore(storage), amortize_k=3, recall_top_k=0)
        conv = Conversation()
        conv.add_user_message("m0")
        conv.add_user_message("m1")
        for i in range(2, 5):
            conv.add_user_message(f"m{i}")
            handler.pre_process(conv)
            if i < 4:
                self.assertEqual(len(conv.all_messages), i + 1, "hidden, not yet dropped")
                self.assertEqual(archived(storage), [])
        self.assertEqual([m.content for m in conv.all_messages], ["m3", "m4"])
        self.assertEqual(archived(storage), [("user", "m0"), ("user", "m1"), ("user", "m2")])

    def test_archives_spoken_turns_only_and_never_recalled_context(self):
        storage = RAMStorage()
        handler = MemoryHandler(MaxTurnsStrategy(1), MemoryStore(storage), amortize_k=1, recall_top_k=0)
        conv = Conversation()
        conv.add_user_message("what is the weather", context="(Earlier: user lives in Oslo)")
        call = ToolCall(name="weather")
        conv.add_assistant_message("", tool_calls=[call])
        conv.add_tool_message('{"sky": "clear"}', tool_call=call)
        conv.add_tool_message("Now answer.")
        conv.add_assistant_message("Clear skies.")
        conv.add_user_message("thanks")

        handler.pre_process(conv)

        self.assertEqual(archived(storage), [("user", "what is the weather"), ("assistant", "Clear skies.")])

    def test_auto_archive_can_be_disabled(self):
        storage = RAMStorage()
        handler = MemoryHandler(MaxTurnsStrategy(1), MemoryStore(storage), amortize_k=1, auto_archive=False)
        conv = Conversation()
        conv.add_user_message("first message")
        conv.add_user_message("second message")
        handler.pre_process(conv)
        handler.finalize(conv)
        self.assertEqual(archived(storage), [])

    def test_recall_attaches_context_to_the_newest_user_message(self):
        mem = MemoryStore(RAMStorage())
        mem.store("my favourite colour is teal", role="user")
        mem.store("the meeting moved to friday", role="assistant")
        handler = MemoryHandler(MaxTurnsStrategy(20), mem)

        conv = Conversation()
        message = conv.add_user_message("remind me of my favourite colour")
        self.assertTrue(handler.pre_process(conv))

        self.assertEqual(message.context, "(Earlier: my favourite colour is teal)")
        self.assertEqual(message.content, "remind me of my favourite colour")
        self.assertEqual(conv.messages[-1]["content"], f"{message.context}\n{message.content}")

    def test_recall_runs_once_per_message_and_keeps_existing_context(self):
        mem = MemoryStore(RAMStorage())
        mem.store("my favourite colour is teal")
        handler = MemoryHandler(MaxTurnsStrategy(20), mem)
        conv = Conversation()
        message = conv.add_user_message("my favourite colour again?", context="(User interrupted.)")

        handler.pre_process(conv)
        self.assertFalse(handler.pre_process(conv))
        self.assertEqual(message.context, "(Earlier: my favourite colour is teal)\n(User interrupted.)")

    def test_recall_skips_what_is_already_in_context_and_short_queries(self):
        mem = MemoryStore(RAMStorage())
        mem.store("my favourite colour is teal")
        handler = MemoryHandler(MaxTurnsStrategy(20), mem)

        conv = Conversation()
        conv.add_user_message("my favourite colour is teal")
        conv.add_assistant_message("Noted.")
        repeat = conv.add_user_message("what is my favourite colour")
        self.assertFalse(handler.pre_process(conv))
        self.assertIsNone(repeat.context)

        short = Conversation()
        short.add_user_message("hm?")
        self.assertFalse(handler.pre_process(short))

    def test_recall_only_after_user_messages_and_respects_top_k(self):
        mem = MemoryStore(RAMStorage())
        for i in range(5):
            mem.store(f"colour fact number {i}")
        conv = Conversation()
        conv.add_user_message("colour facts")
        conv.add_assistant_message("colour facts are fun")
        self.assertFalse(MemoryHandler(MaxTurnsStrategy(20), mem).pre_process(conv))

        asked = Conversation()
        message = asked.add_user_message("tell me colour facts")
        MemoryHandler(MaxTurnsStrategy(20), mem, recall_top_k=3).pre_process(asked)
        self.assertEqual(message.context.count("(Earlier:"), 3)

        silent = Conversation()
        silent.add_user_message("tell me colour facts")
        self.assertFalse(MemoryHandler(MaxTurnsStrategy(20), mem, recall_top_k=0).pre_process(silent))

    def test_finalize_archives_everything_exactly_once(self):
        storage = RAMStorage()
        handler = MemoryHandler(MaxTurnsStrategy(20), MemoryStore(storage))
        conv = Conversation()
        conv.add_user_message("remember the milk")
        conv.add_assistant_message("I will.")

        handler.finalize(conv)
        handler.finalize(conv)

        self.assertEqual(archived(storage), [("user", "remember the milk"), ("assistant", "I will.")])
        self.assertTrue(all(m.archived for m in conv.all_messages))
        restored = Conversation.from_dict(conv.to_dict())
        handler.finalize(restored)
        self.assertEqual(len(archived(storage)), 2, "a saved and reloaded conversation is not archived again")

    def test_failing_memory_backend_does_not_break_the_turn(self):
        class Broken:
            def store(self, content, **kwargs):
                raise OSError("disk full")

            def retrieve(self, query, **kwargs):
                raise OSError("disk gone")

        handler = MemoryHandler(MaxTurnsStrategy(1), MemoryStore(Broken()), amortize_k=1)
        conv = Conversation()
        conv.add_user_message("first message here")
        conv.add_user_message("second message here")
        with self.assertLogs("voxpipe.llm.context", level="WARNING") as logs:
            self.assertTrue(handler.pre_process(conv))
            handler.finalize(conv)
        self.assertEqual(conv.visible_count(), 1)
        self.assertGreaterEqual(len(logs.records), 2)


class TestMemoryAcrossSessions(unittest.TestCase):
    """The whole loop: a session's turns reach memory and inform a later session."""

    def _session(self, storage, *scripts):
        llm = ScriptedLLM(*scripts)
        handler = MemoryHandler(MaxTurnsStrategy(20), MemoryStore(storage))
        session = Session(llm=llm, context_handler=handler)
        return session, llm

    def test_closed_session_is_recalled_by_the_next(self):
        storage = RAMStorage()
        first, _ = self._session(storage, ["Noted, teal it is."])
        list(first("my favourite colour is teal"))
        first.close()

        second, llm = self._session(storage, ["It is teal."])
        self.addCleanup(second.close)
        list(second("what was my favourite colour"))

        prompt = llm.calls[0].messages[-1]["content"]
        self.assertIn("(Earlier: my favourite colour is teal)", prompt)
        self.assertTrue(prompt.endswith("what was my favourite colour"))

    def test_reset_archives_the_conversation_it_discards(self):
        storage = RAMStorage()
        session, _ = self._session(storage, ["Sure."])
        self.addCleanup(session.close)
        list(session("call me Ishmael please"))
        session.reset()
        self.assertEqual(archived(storage), [("user", "call me Ishmael please"), ("assistant", "Sure.")])


if __name__ == "__main__":
    unittest.main()
