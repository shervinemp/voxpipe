import unittest
from unittest.mock import MagicMock

from voxpipe.llm.tools import ToolCall


class TestSessionIntegration(unittest.TestCase):
    """End-to-end Session flow with a mocked LLM streaming output."""

    def _make_mock_llm(self, tokens: list[str], decoder=None):
        from voxpipe.llm.decoders import GeneralDecoder
        from voxpipe.llm.model import LLM
        from voxpipe.llm.context import DropOldestStrategy
        _decoder = decoder or GeneralDecoder()

        class MockLLM(LLM):
            decoder = _decoder
            def _infer(self, conversation, *, session_state, **kwargs):
                yield from tokens
            def create_context_strategy(self, max_turns=20):
                return DropOldestStrategy(max_turns)
            def count_tokens(self, text):
                return max(1, len(text) // 2)

        llm = MockLLM()
        llm.logger = MagicMock()
        return llm

    def _collect(self, gen):
        text, calls = [], []
        for item in gen:
            if isinstance(item, ToolCall):
                calls.append((item.name, item.arguments))
            elif isinstance(item, str):
                text.append(item)
        return text, calls

    def test_session_plain_text(self):
        from voxpipe.llm.session import Session
        llm = self._make_mock_llm(["Hello! ", "How ", "can ", "I ", "help?"])
        sess = Session(llm=llm)
        text, calls = self._collect(sess("Hi."))
        self.assertEqual("".join(text), "Hello! How can I help?")
        self.assertEqual(calls, [])
        sess.close()

    def test_session_with_tool_call(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.tools import Tool, ToolResult
        from voxpipe.llm.conversation import Conversation
        tool_results = []
        def my_tool(query: str) -> ToolResult:
            tool_results.append(query)
            return ToolResult(result={"result": f"result for {query}"})
        tokens = [
            "Let me check...",
            "<|tool_call>", 'call:my_tool{query:<|"|>hello<|"|>}', "<tool_call|>",
        ]
        llm = self._make_mock_llm(tokens)
        conv = Conversation()
        conv.set_system_message("You are a helpful assistant.")
        conv.tools["my_tool"] = Tool.from_callable("my_tool", my_tool)
        conv.add_user_message("find hello")
        sess = Session(llm=llm, conversation=conv)
        text, calls = self._collect(sess())
        self.assertEqual(len(tool_results), 1)
        self.assertEqual(tool_results[0], "hello")
        msgs = sess.conversation.messages
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        self.assertIn("result for hello", tool_msgs[0]["content"])
        sess.close()

    def test_session_second_pass_uses_tool_results(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.tools import Tool, ToolResult
        from voxpipe.llm.conversation import Conversation
        from voxpipe.llm.decoders import GeneralDecoder

        class TwoPassLLM:
            decoder = GeneralDecoder()
            _call_count = 0
            def create_context_strategy(self, max_turns=20):
                from voxpipe.llm.context import DropOldestStrategy
                return DropOldestStrategy(max_turns)
            def count_tokens(self, text):
                return max(1, len(text) // 2)
            def __call__(self, conversation, session_state=None, **kwargs):
                self._call_count += 1
                if self._call_count == 1:
                    tokens = ["<|tool_call>", 'call:retrieve{query:<|"|>test<|"|>}', "<tool_call|>"]
                else:
                    tokens = ["Based on retrieval, the answer is 42."]
                yield from self.decoder(iter(tokens))

        def retrieve(query: str) -> ToolResult:
            return ToolResult(result={"answer": "42"})
        conv = Conversation()
        conv.set_system_message("You are a helpful assistant.")
        conv.tools["retrieve"] = Tool.from_callable("retrieve", retrieve)
        conv.add_user_message("what is the answer?")
        llm = TwoPassLLM()
        llm.logger = MagicMock()
        sess = Session(llm=llm, conversation=conv)
        text, calls = self._collect(sess())
        self.assertIn("Based on retrieval, the answer is 42.", "".join(text))
        sess.close()

    def test_session_recovers_from_tool_error(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.tools import Tool
        from voxpipe.llm.conversation import Conversation
        def broken_tool(**kwargs):
            raise RuntimeError("tool exploded")
        tokens = ["<|tool_call>", 'call:broken_tool{x:1}', "<tool_call|>"]
        llm = self._make_mock_llm(tokens)
        conv = Conversation()
        conv.tools["broken_tool"] = Tool.from_callable("broken_tool", broken_tool)
        conv.add_user_message("do something")
        sess = Session(llm=llm, conversation=conv)
        text, calls = self._collect(sess())
        msgs = sess.conversation.messages
        self.assertTrue(any("Tool Error" in m["content"] for m in msgs if m["role"] == "tool"))
        sess.close()

    def test_session_unknown_tool(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.conversation import Conversation
        # Tool name not in conversation.tools — triggers retry, then fallback
        tokens = ["<|tool_call>", 'call:nonexistent{args:{}}', "<tool_call|>"]
        llm = self._make_mock_llm(tokens)
        conv = Conversation()
        conv.add_user_message("test")
        sess = Session(llm=llm, conversation=conv)
        text, calls = self._collect(sess())
        combined = "".join(text)
        self.assertIn("persistent error", combined)

    def test_session_lock_serializes_access(self):
        from voxpipe.llm.session import Session
        import threading, time
        call_order = []

        class LockTestLLM:
            def create_context_strategy(self, max_turns=20):
                from voxpipe.llm.context import DropOldestStrategy
                return DropOldestStrategy(max_turns)
            def count_tokens(self, text):
                return max(1, len(text) // 2)
            def __call__(self, conversation, session_state=None, **kwargs):
                call_order.append("enter")
                time.sleep(0.05)
                call_order.append("exit")
                return iter(["x"])

        llm = LockTestLLM()
        llm.logger = MagicMock()
        sess = Session(llm=llm)
        results = []
        def run():
            for chunk in sess("hello"):
                results.append(chunk)
        t1 = threading.Thread(target=run)
        t2 = threading.Thread(target=run)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertEqual(call_order, ["enter", "exit", "enter", "exit"])
        sess.close()

    def test_session_retry_on_parse_error(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.conversation import Conversation
        tokens = ["<|tool_call>", "call:bad_tool{invalid!!!}}", "<tool_call|>"]
        llm = self._make_mock_llm(tokens)
        conv = Conversation()
        conv.add_user_message("do it")
        sess = Session(llm=llm, conversation=conv)
        text, calls = self._collect(sess())
        self.assertIn("persistent error", "".join(text))
        sess.close()

    def test_session_empty_query(self):
        from voxpipe.llm.session import Session
        llm = self._make_mock_llm([])
        sess = Session(llm=llm)
        text, calls = self._collect(sess())
        sess.close()

    def test_session_gather_timeout(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.tools import Tool
        from voxpipe.llm.conversation import Conversation
        import time

        def slow_tool(**kwargs):
            time.sleep(20)
            return "too late"

        tokens = ["<|tool_call>", 'call:slow_tool{x:1}', "<tool_call|>"]
        llm = self._make_mock_llm(tokens)
        conv = Conversation()
        conv.tools["slow_tool"] = Tool.from_callable("slow_tool", slow_tool)
        conv.add_user_message("do slow")
        sess = Session(llm=llm, conversation=conv)
        t0 = time.monotonic()
        text, calls = self._collect(sess())
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 15)
        msgs = sess.conversation.messages
        self.assertTrue(any("Tool Error" in m["content"] for m in msgs if m["role"] == "tool"))
        sess.close()

    def test_toolcall_passthrough_ge(self):
        from voxpipe.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        tc = ToolCall(name="direct", arguments={"from": "test"})
        result = list(d(iter([tc])))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "direct")


class TestSessionMethods(unittest.TestCase):
    """Session methods not covered by integration tests."""

    def _make_mock_llm(self, tokens):
        from voxpipe.llm.model import LLM
        from voxpipe.llm.decoders import GeneralDecoder
        from voxpipe.llm.context import DropOldestStrategy
        _decoder = GeneralDecoder()
        class M(LLM):
            decoder = _decoder
            def _infer(self, conversation, *, session_state, **kwargs):
                yield from tokens
            def create_context_strategy(self, max_turns=20):
                return DropOldestStrategy(max_turns)
            def count_tokens(self, text):
                return max(1, len(text) // 2)
        llm = M()
        llm.logger = MagicMock()
        return llm

    def test_complete_once_returns_string(self):
        from voxpipe.llm.session import Session
        llm = self._make_mock_llm(["hello world"])
        sess = Session(llm=llm)
        result = sess.complete_once("test", system="sys")
        self.assertEqual(result, "hello world")
        sess.close()

    def test_complete_once_empty_query_raises(self):
        from voxpipe.llm.session import Session, LLMError
        llm = self._make_mock_llm([])
        sess = Session(llm=llm)
        with self.assertRaises(LLMError):
            sess.complete_once("")
        with self.assertRaises(LLMError):
            sess.complete_once("   ")
        sess.close()

    def test_complete_once_does_not_mutate_conversation(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        conv.add_user_message("persistent")
        llm = self._make_mock_llm(["result"])
        sess = Session(llm=llm, conversation=conv)
        sess.complete_once("isolated")
        self.assertEqual(len(conv._messages), 1)
        self.assertEqual(conv.messages[-1]["content"], "persistent")
        sess.close()

    def test_complete_once_with_system(self):
        from voxpipe.llm.session import Session
        llm = self._make_mock_llm(["answer"])
        sess = Session(llm=llm)
        result = sess.complete_once("q", system="You are helpful.")
        self.assertEqual(result, "answer")
        sess.close()

    def test_reset_clears_conversation_and_state(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        conv.add_user_message("old")
        llm = self._make_mock_llm([])
        sess = Session(llm=llm, conversation=conv)
        sess._session_state["k"] = "v"
        sess.reset()
        self.assertEqual(len(sess.conversation._messages), 0)
        self.assertEqual(sess._session_state, {})
        sess.close()

    def test_reset_with_new_conversation(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.conversation import Conversation
        new_conv = Conversation()
        new_conv.add_user_message("new")
        llm = self._make_mock_llm([])
        sess = Session(llm=llm)
        sess.reset(new_conv)
        self.assertEqual(len(sess.conversation._messages), 1)
        self.assertEqual(sess.conversation.messages[-1]["content"], "new")
        sess.close()

    def test_save_load_round_trip(self):
        import tempfile, os, json
        from voxpipe.llm.session import Session
        from pathlib import Path

        llm = self._make_mock_llm([])
        sess = Session(llm=llm)
        sess.conversation.add_user_message("hello")
        sess._session_state["note"] = "test"

        with tempfile.TemporaryDirectory() as tmp:
            sess.save(tmp)
            self.assertTrue(os.path.exists(os.path.join(tmp, "conversation.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "state.json")))

            sess2 = Session.load(tmp, llm=llm)
            self.assertEqual(len(sess2.conversation._messages), 1)
            self.assertEqual(sess2.conversation.messages[-1]["content"], "hello")
            self.assertEqual(sess2._session_state.get("note"), "test")
        sess.close()
        sess2.close()

    def test_save_kv_cache(self):
        import tempfile, os
        from voxpipe.llm.session import Session
        llm = self._make_mock_llm([])
        sess = Session(llm=llm)
        sess._session_state["model_state"] = b"\x00\x01\x02"
        with tempfile.TemporaryDirectory() as tmp:
            sess.save(tmp, save_kv_cache=True)
            self.assertTrue(os.path.exists(os.path.join(tmp, "kv_cache.bin")))
        sess.close()

    def test_close_stops_tool_caller(self):
        from voxpipe.llm.session import Session
        llm = self._make_mock_llm([])
        sess = Session(llm=llm)
        t = sess.tool_caller._loop_thread
        self.assertTrue(t.is_alive())
        sess.close()
        t.join(timeout=2)
        self.assertFalse(t.is_alive())

    def test_gather_empty_queue(self):
        from voxpipe.llm.session import Session
        llm = self._make_mock_llm([])
        sess = Session(llm=llm)
        result = sess.tool_caller.gather()
        self.assertEqual(result, {})
        sess.close()

    def test_lock_released_after_full_consumption(self):
        from voxpipe.llm.session import Session
        llm = self._make_mock_llm(["hello", " world"])
        sess = Session(llm=llm)
        gen = sess("Hi.")
        list(gen)
        self.assertFalse(sess._lock.locked(), "Lock still held after generator consumed")
        sess.close()

    def test_lock_released_after_close(self):
        from voxpipe.llm.session import Session
        llm = self._make_mock_llm(["hello", " world"])
        sess = Session(llm=llm)
        gen = sess("Hi.")
        try:
            next(gen)
        except StopIteration:
            pass
        finally:
            gen.close()
        self.assertFalse(sess._lock.locked(), "Lock still held after partial iteration + close")
        sess.close()

    def test_concurrent_call_blocks_until_lock_released(self):
        import threading
        from voxpipe.llm.session import Session
        llm = self._make_mock_llm(["hello", " world"])
        sess = Session(llm=llm)

        results = []
        def consume(gen):
            list(gen)
            results.append("done")

        gen1 = sess("First.")
        t = threading.Thread(target=consume, args=(gen1,), daemon=True)
        t.start()
        t.join(timeout=5)
        self.assertIn("done", results)
        self.assertFalse(sess._lock.locked(), "Lock still held after threaded consumption")
        sess.close()


class TestSessionConfirm(unittest.TestCase):
    """_confirm registration and ToolChoice handling."""

    def test_register_confirm_adds_tool(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.tools import Tool, ToolChoice
        from voxpipe.llm.conversation import Conversation
        def fn(x: int) -> ToolChoice:
            return ToolChoice(result=str(x))
        conv = Conversation()
        conv.tools["test"] = Tool.from_callable("test", fn)
        llm = MagicMock()
        llm.create_context_strategy = MagicMock(return_value=MagicMock())
        llm.count_tokens = MagicMock(return_value=1)
        sess = Session(llm=llm, conversation=conv)
        try:
            list(sess("go"))
            self.assertIn("_confirm", sess.conversation.tools)
            confirm = sess.conversation.tools["_confirm"]
            self.assertEqual(confirm.name, "_confirm")
            self.assertIsNotNone(confirm.callback)
        finally:
            sess.close()

    def test_no_register_when_no_choice(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.tools import Tool, ToolResult
        from voxpipe.llm.conversation import Conversation
        def fn(x: int) -> ToolResult:
            return ToolResult(result=str(x))
        conv = Conversation()
        conv.tools["test"] = Tool.from_callable("test", fn)
        llm = MagicMock()
        llm.create_context_strategy = MagicMock(return_value=MagicMock())
        llm.count_tokens = MagicMock(return_value=1)
        sess = Session(llm=llm, conversation=conv)
        try:
            list(sess("go"))
            self.assertNotIn("_confirm", sess.conversation.tools)
        finally:
            sess.close()

    def test_register_idempotent(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.tools import Tool, ToolChoice
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        conv.tools["_confirm"] = Tool(name="_confirm", description="existing")
        conv.tools["test"] = Tool.from_callable("test", lambda x: ToolChoice(result=str(x)))
        llm = MagicMock()
        llm.create_context_strategy = MagicMock(return_value=MagicMock())
        llm.count_tokens = MagicMock(return_value=1)
        sess = Session(llm=llm, conversation=conv)
        try:
            list(sess("go"))
            self.assertIn("_confirm", sess.conversation.tools)
            self.assertEqual(sess.conversation.tools["_confirm"].description, "existing")
        finally:
            sess.close()

    def test_on_confirm_dispatches_to_tool(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.tools import Tool, ToolResult, ToolCall, ToolChoice
        from voxpipe.llm.conversation import Conversation
        captured = []
        def fn(x: int) -> ToolResult:
            captured.append(x)
            return ToolResult(result={"x": x})
        conv = Conversation()
        conv.tools["test"] = Tool.from_callable("test", fn)
        llm = MagicMock()
        llm.create_context_strategy = MagicMock(return_value=MagicMock())
        llm.count_tokens = MagicMock(return_value=1)
        sess = Session(llm=llm, conversation=conv)
        try:
            call = ToolCall(name="test", arguments={"x": 42})
            choice = ToolChoice(result={"uid": "tc_1234", "allow": [True, False], "remember": [True, False]})
            conv.get_meta("test")["pending"] = {"tc_1234": (call, choice)}
            result = sess._on_confirm(uid="tc_1234", choice={"allow": True, "remember": False})
            self.assertIsInstance(result, ToolResult)
            self.assertEqual(captured, [42])
            self.assertIn("42", result.result)
        finally:
            sess.close()

    def test_on_confirm_permission_remember(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.tools import Tool, ToolResult, ToolCall, ToolChoice
        from voxpipe.llm.conversation import Conversation
        captured = []
        def fn(x: int) -> ToolResult:
            captured.append(x)
            return ToolResult(result={"x": x})
        conv = Conversation()
        conv.tools["test"] = Tool.from_callable("test", fn)
        llm = MagicMock()
        llm.create_context_strategy = MagicMock(return_value=MagicMock())
        llm.count_tokens = MagicMock(return_value=1)
        sess = Session(llm=llm, conversation=conv)
        try:
            call = ToolCall(name="test", arguments={"x": 99})
            choice = ToolChoice(result={"uid": "tc_5678", "allow": [True, False], "remember": [True, False]})
            conv.get_meta("test")["pending"] = {"tc_5678": (call, choice)}
            result = sess._on_confirm(uid="tc_5678", choice={"allow": True, "remember": True})
            self.assertIsInstance(result, ToolResult)
            self.assertEqual(captured, [99])
            self.assertTrue(conv.get_meta("test").get("_permission"))
        finally:
            sess.close()

    def test_on_confirm_mismatched_keys_raises(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.tools import ToolCall, ToolChoice
        from voxpipe.core.exceptions import ToolError
        llm = MagicMock()
        llm.create_context_strategy = MagicMock(return_value=MagicMock())
        llm.count_tokens = MagicMock(return_value=1)
        sess = Session(llm=llm)
        try:
            call = ToolCall(name="test", arguments={})
            choice = ToolChoice(result={"uid": "tc_9999", "allow": [True, False], "remember": [True, False]})
            sess.conversation.get_meta("test")["pending"] = {"tc_9999": (call, choice)}
            with self.assertRaises(ToolError):
                sess._on_confirm(uid="tc_9999", choice={"allow": True})  # Missing 'remember'
        finally:
            sess.close()

    def test_on_confirm_unknown_uid_raises(self):
        from voxpipe.llm.session import Session
        from voxpipe.core.exceptions import ToolError
        llm = MagicMock()
        llm.create_context_strategy = MagicMock(return_value=MagicMock())
        llm.count_tokens = MagicMock(return_value=1)
        sess = Session(llm=llm)
        try:
            with self.assertRaises(ToolError):
                sess._on_confirm(uid="nonexistent", choice={"allow": True, "remember": False})
        finally:
            sess.close()

    def test_tool_choice_injects_assistant_message(self):
        from voxpipe.llm.session import Session
        from voxpipe.llm.tools import Tool, ToolChoice
        from voxpipe.llm.conversation import Conversation
        from voxpipe.llm.decoders import GeneralDecoder

        class ChoiceLLM:
            decoder = GeneralDecoder()
            called = False
            def create_context_strategy(self, max_turns=20):
                from voxpipe.llm.context import DropOldestStrategy
                return DropOldestStrategy(max_turns)
            def count_tokens(self, text):
                return max(1, len(text) // 2)
            def __call__(self, conversation, session_state=None, **kwargs):
                if not self.called:
                    self.called = True
                    from voxpipe.llm.tools import ToolCall
                    yield ToolCall(name="test", arguments={"x": 5})
                else:
                    yield "Got it."

        captured = []
        def fn(x: int) -> ToolChoice:
            captured.append(x)
            return ToolChoice(result={"slot": ["1", "2", "3"]}, speech="Which slot?")
        conv = Conversation()
        conv.set_system_message("You are helpful.")
        conv.tools["test"] = Tool.from_callable("test", fn)
        llm = ChoiceLLM()
        llm.logger = MagicMock()
        sess = Session(llm=llm, conversation=conv)
        try:
            text = list(sess("choose"))
            roles = [m["role"] for m in conv.messages]
            self.assertIn("tool", roles)
            assistant_msgs = [m for m in conv.messages if m["role"] == "assistant"]
            self.assertTrue(any("_confirm" in m["content"] for m in assistant_msgs),
                            "assistant message should mention _confirm")
            self.assertIn("Which slot?", "".join(text), "speech should be yielded")
        finally:
            sess.close()
