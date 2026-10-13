import threading
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


def conversation_of(turns):
    """A conversation of ``turns`` question-and-answer turns, q0/a0 first."""
    conv = Conversation()
    for i in range(turns):
        conv.add_user_message(f"q{i}")
        conv.add_assistant_message(f"a{i}")
    return conv


class TestMaxTurnsStrategy(unittest.TestCase):
    def test_nothing_to_evict_up_to_the_limit(self):
        strategy = MaxTurnsStrategy(3)
        self.assertEqual(strategy(Conversation()), 0)
        self.assertEqual(strategy(conversation_of(3)), 0)

    def test_over_the_limit_cuts_back_to_trim_to(self):
        strategy = MaxTurnsStrategy(4, trim_to=2)
        conv = conversation_of(5)
        self.assertEqual(strategy(conv), 6, "three turns of two messages each")
        conv.evict_oldest(strategy(conv))
        self.assertEqual([m.content for m in conv.history], ["q3", "a3", "q4", "a4"])

    def test_a_turn_counts_once_however_many_messages_it_took(self):
        conv = Conversation()
        for i in range(2):
            conv.add_user_message(f"q{i}")
            call = ToolCall(name="lookup")
            conv.add_assistant_message("", tool_calls=[call])
            conv.add_tool_message("{}", tool_call=call)
            conv.add_tool_message("Now answer.")
            conv.add_assistant_message(f"a{i}")
        self.assertEqual(MaxTurnsStrategy(2)(conv), 0, "ten messages, but only two turns")
        self.assertEqual(MaxTurnsStrategy(1)(conv), 5)

    def test_the_turn_being_asked_counts(self):
        conv = conversation_of(2)
        conv.add_user_message("q2")
        self.assertEqual(MaxTurnsStrategy(2, trim_to=2)(conv), 2)

    def test_trim_to_defaults_to_sixty_percent_of_the_limit(self):
        self.assertEqual(MaxTurnsStrategy(10).trim_to, 6)
        self.assertEqual(MaxTurnsStrategy(20).trim_to, 12)
        self.assertEqual(MaxTurnsStrategy(1).trim_to, 1)

    def test_rejects_limits_that_make_no_sense(self):
        for arguments in [dict(max_turns=0), dict(max_turns=4, trim_to=0), dict(max_turns=4, trim_to=5)]:
            with self.subTest(arguments), self.assertRaises(ValueError):
                MaxTurnsStrategy(**arguments)

    def test_the_start_of_the_prompt_holds_still_between_cuts(self):
        """The point of cutting below the limit: a prompt-prefix cache survives most turns."""
        conv = Conversation()
        handler = MemoryHandler(MaxTurnsStrategy(10))
        moved = []
        for i in range(40):
            first_before = conv.history[0] if conv.history else None
            conv.add_user_message(f"q{i}")
            handler.pre_process(conv)
            conv.add_assistant_message(f"a{i}")
            if first_before is not None and conv.history[0] is not first_before:
                moved.append(i)
        self.assertEqual(moved, [10, 15, 20, 25, 30, 35], "one cut every five turns, not one per turn")
        self.assertLessEqual(len(conv.history), 20)


class TestTokenBudgetStrategy(unittest.TestCase):
    def _conv(self, *contents):
        conv = Conversation()
        for content in contents:
            conv.add_user_message(content)
        return conv

    def test_nothing_to_evict_within_the_budget(self):
        strategy = TokenBudgetStrategy(count_tokens=len, budget=30, per_message_overhead=0)
        self.assertEqual(strategy(Conversation()), 0)
        self.assertEqual(strategy(self._conv("a" * 10, "b" * 10, "c" * 10)), 0)

    def test_over_budget_evicts_oldest_down_to_trim_to(self):
        fits_again = TokenBudgetStrategy(len, budget=30, per_message_overhead=0, trim_to=30)
        self.assertEqual(fits_again(self._conv("a" * 10, "b" * 10, "c" * 10, "d" * 10)), 1)
        self.assertEqual(fits_again(self._conv("a" * 25, "b" * 25, "c" * 10)), 2)

        with_headroom = TokenBudgetStrategy(len, budget=30, per_message_overhead=0, trim_to=10)
        self.assertEqual(with_headroom(self._conv("a" * 10, "b" * 10, "c" * 10, "d" * 10)), 3)

    def test_trim_to_defaults_to_sixty_percent_of_the_room_for_messages(self):
        strategy = TokenBudgetStrategy(len, budget=100, per_message_overhead=0)
        conv = self._conv(*["m" * 10] * 11)
        self.assertEqual(strategy(conv), 5, "110 tokens cut back to 60")

        conv.set_system_message("s" * 50)
        self.assertEqual(strategy(conv), 8, "50 fixed + 60% of the 50 left = 80, so messages shrink to 30")

    def test_never_evicts_the_newest_message(self):
        strategy = TokenBudgetStrategy(count_tokens=len, budget=5, per_message_overhead=0)
        self.assertEqual(strategy(self._conv("a" * 50)), 0)
        self.assertEqual(strategy(self._conv("a" * 50, "b" * 50)), 1)

    def test_counts_system_prompt_rules_tools_and_overhead(self):
        def strategy(**kwargs):
            return TokenBudgetStrategy(len, budget=20, trim_to=20, **kwargs)

        conv = self._conv("a" * 10, "b" * 10)
        self.assertEqual(strategy(per_message_overhead=0)(conv), 0)
        self.assertEqual(strategy(per_message_overhead=1)(conv), 1)
        conv.set_system_message("s" * 5)
        self.assertEqual(strategy(per_message_overhead=0)(conv), 1)
        conv.set_system_message("")
        conv.rules = ["r"]
        self.assertEqual(strategy(per_message_overhead=0)(conv), 1)
        conv.rules = []
        conv.tools = [Tool(name="t", description="d")]
        self.assertEqual(strategy(per_message_overhead=0)(conv), 1)

    def test_counts_recalled_context(self):
        conv = self._conv("a" * 10, "b" * 10)
        conv.last_message.context = "c" * 10
        self.assertEqual(TokenBudgetStrategy(len, budget=25, per_message_overhead=0, trim_to=25)(conv), 1)

    def test_rejects_a_trim_target_above_the_budget(self):
        with self.assertRaises(ValueError):
            TokenBudgetStrategy(len, budget=10, trim_to=11)


class TestCompositeStrategy(unittest.TestCase):
    def test_takes_the_most_demanding(self):
        conv = conversation_of(6)
        strategy = CompositeStrategy(
            MaxTurnsStrategy(4, trim_to=4), MaxTurnsStrategy(2, trim_to=2), MaxTurnsStrategy(10)
        )
        self.assertEqual(strategy(conv), 8)
        self.assertEqual(CompositeStrategy()(conv), 0)


class TestMemoryHandler(unittest.TestCase):
    def test_base_handler_hooks_do_nothing(self):
        handler = ContextHandler()
        conv = Conversation()
        self.assertFalse(handler.pre_process(conv))
        self.assertFalse(handler.post_process(conv))
        handler.finalize(conv)

    def test_leaves_a_conversation_within_bounds_alone(self):
        handler = MemoryHandler(MaxTurnsStrategy(5))
        self.assertFalse(handler.pre_process(Conversation()))
        conv = conversation_of(3)
        self.assertFalse(handler.pre_process(conv))
        self.assertEqual(len(conv.history), 6)

    def test_evicts_and_archives_what_the_strategy_no_longer_wants(self):
        storage = RAMStorage()
        handler = MemoryHandler(MaxTurnsStrategy(2, trim_to=2), MemoryStore(storage, session_id="test"))

        conv = Conversation()
        for i in range(10):
            conv.add_user_message(f"message turn {i}")

        self.assertTrue(handler.pre_process(conv))
        self.assertEqual([m.content for m in conv.history], ["message turn 8", "message turn 9"])
        self.assertEqual(archived(storage), [("user", f"message turn {i}") for i in range(8)])
        self.assertEqual(storage._entries[0][0], "session:test")

    def test_works_without_a_memory(self):
        handler = MemoryHandler(MaxTurnsStrategy(1))
        conv = conversation_of(3)
        self.assertTrue(handler.pre_process(conv))
        self.assertEqual([m.content for m in conv.history], ["q2", "a2"])
        handler.finalize(conv)

    def test_any_storage_serves_as_memory(self):
        """The handler needs only ``store`` and ``retrieve``; a MemoryStore is one way to scope them."""
        storage = RAMStorage()
        handler = MemoryHandler(MaxTurnsStrategy(1), storage)
        conv = conversation_of(2)
        handler.pre_process(conv)
        self.assertEqual(archived(storage), [("user", "q0"), ("assistant", "a0")])

    def test_archives_spoken_turns_only_and_never_recalled_context(self):
        storage = RAMStorage()
        handler = MemoryHandler(MaxTurnsStrategy(1), MemoryStore(storage), recall_top_k=0)
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
        handler = MemoryHandler(MaxTurnsStrategy(1), MemoryStore(storage), auto_archive=False)
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
        self.assertTrue(all(m.archived for m in conv.history))
        restored = Conversation.from_dict(conv.to_dict())
        handler.finalize(restored)
        self.assertEqual(len(archived(storage)), 2, "a saved and reloaded conversation is not archived again")

    def test_failing_memory_backend_does_not_break_the_turn(self):
        class Broken:
            def store(self, content, **kwargs):
                raise OSError("disk full")

            def retrieve(self, query, **kwargs):
                raise OSError("disk gone")

        handler = MemoryHandler(MaxTurnsStrategy(1), Broken())
        conv = Conversation()
        conv.add_user_message("first message here")
        conv.add_user_message("second message here")
        with self.assertLogs("voxpipe.llm.context", level="WARNING") as logs:
            self.assertTrue(handler.pre_process(conv))
            handler.finalize(conv)
        self.assertEqual(len(conv.history), 1)
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

    def test_turns_pushed_out_of_a_long_conversation_are_archived_as_they_go(self):
        storage = RAMStorage()
        llm = ScriptedLLM(["Noted."])
        handler = MemoryHandler(MaxTurnsStrategy(2, trim_to=1), MemoryStore(storage), recall_top_k=0)
        session = Session(llm=llm, context_handler=handler)
        self.addCleanup(session.close)

        for i in range(3):
            list(session(f"fact number {i}"))

        self.assertEqual(
            archived(storage),
            [("user", "fact number 0"), ("assistant", "Noted."), ("user", "fact number 1"), ("assistant", "Noted.")],
        )
        self.assertEqual([m["content"] for m in llm.calls[-1].messages], ["fact number 2"])


class TestArchivingUnderFailure(unittest.TestCase):
    def test_messages_a_failing_memory_refused_are_offered_again(self):
        storage = RAMStorage()

        class Flaky:
            down = True

            def store(self, content, **kwargs):
                if self.down:
                    raise OSError("database is locked")
                return storage.store(content, **kwargs)

            def retrieve(self, query, **kwargs):
                return []

        memory = Flaky()
        handler = MemoryHandler(MaxTurnsStrategy(1), memory, recall_top_k=0)
        conv = conversation_of(2)
        with self.assertLogs("voxpipe.llm.context", level="WARNING"):
            handler.pre_process(conv)
        self.assertEqual(archived(storage), [], "the backend was down when q0 and a0 were evicted")

        memory.down = False
        conv.add_user_message("q2")
        handler.pre_process(conv)
        self.assertEqual(
            archived(storage), [("user", "q0"), ("assistant", "a0"), ("user", "q1"), ("assistant", "a1")]
        )

    def test_a_failed_finalize_is_retried_without_duplicates(self):
        storage = RAMStorage()
        attempts = []

        class FailsOnce:
            def store(self, content, **kwargs):
                attempts.append(content)
                if len(attempts) == 2:
                    raise OSError("disk full")
                return storage.store(content, **kwargs)

            def retrieve(self, query, **kwargs):
                return []

        handler = MemoryHandler(MaxTurnsStrategy(20), FailsOnce())
        conv = conversation_of(2)
        with self.assertLogs("voxpipe.llm.context", level="WARNING"):
            handler.finalize(conv)
        handler.finalize(conv)
        self.assertEqual(
            archived(storage), [("user", "q0"), ("assistant", "a0"), ("user", "q1"), ("assistant", "a1")]
        )

    def test_reset_mid_reply_still_archives_what_was_heard(self):
        storage = RAMStorage()
        resume = threading.Event()

        def script(conversation):
            yield "The capital "
            resume.wait(timeout=2)
            yield "is Paris."

        session = Session(llm=ScriptedLLM(script), context_handler=MemoryHandler(MaxTurnsStrategy(20), storage))
        self.addCleanup(session.close)
        stream = session("what is the capital of France")
        self.assertEqual(next(stream), "The capital ")

        session.reset()  # archives the conversation while the turn has yet to record its partial reply
        resume.set()
        self.assertEqual(list(stream), [])

        self.assertEqual(
            archived(storage), [("user", "what is the capital of France"), ("assistant", "The capital ")]
        )


class TestStrategyArguments(unittest.TestCase):
    def test_token_budget_counts_tool_calls_and_their_arguments(self):
        conv = Conversation()
        conv.add_user_message("q")
        call = ToolCall(name="save", arguments={"text": "x" * 100})
        conv.add_assistant_message("", tool_calls=[call])
        conv.add_tool_message("{}", tool_call=call)
        conv.add_user_message("next")
        self.assertGreater(TokenBudgetStrategy(len, budget=50, per_message_overhead=0, trim_to=50)(conv), 0)

    def test_a_budget_must_be_positive(self):
        for budget in (0, -5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                TokenBudgetStrategy(len, budget=budget)

    def test_turn_limits_must_be_whole_numbers(self):
        for arguments in [dict(max_turns=2.5), dict(max_turns=True), dict(max_turns=10, trim_to=2.5)]:
            with self.subTest(arguments), self.assertRaises(TypeError):
                MaxTurnsStrategy(**arguments)


if __name__ == "__main__":
    unittest.main()
