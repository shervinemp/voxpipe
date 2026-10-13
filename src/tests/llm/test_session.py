import asyncio
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock

from voxpipe.core.exceptions import LLMError, ToolError
from voxpipe.llm.conversation import Conversation
from voxpipe.llm.decoders import GeneralDecoder
from voxpipe.llm.session import Session, TextDelta, ToolCallEvent, ToolCaller, ToolResultEvent
from voxpipe.llm.tools import MalformedToolCall, Tool, ToolCall, ToolChoice, ToolResult, permission

from ._fakes import ScriptedLLM


def gemma_call(name, args="{}"):
    """The chunks a Gemma-style model streams for one tool call."""
    return ["<|tool_call>", f"call:{name}{args}", "<tool_call|>"]


def tool_messages(session):
    return [m["content"] for m in session.conversation.messages if m["role"] == "tool"]


class SessionCase(unittest.TestCase):
    def make(self, *scripts, tools=(), decoder=None, **session_kwargs):
        """A session over a scripted LLM; pass a decoder when the scripts hold tagged tool-call text."""
        llm = ScriptedLLM(*scripts, decoder=decoder)
        session = Session(llm=llm, tools=tools, **session_kwargs)
        self.addCleanup(session.close)
        return session, llm


class TestSessionTurns(SessionCase):
    """Text replies, streaming and history."""

    def test_plain_text(self):
        session, _ = self.make(["Hello! ", "How ", "can ", "I ", "help?"])
        self.assertEqual("".join(session("Hi.")), "Hello! How can I help?")
        self.assertEqual(
            session.conversation.messages,
            [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello! How can I help?"}],
        )

    def test_text_is_streamed_chunk_by_chunk(self):
        session, _ = self.make(["Hello ", "there. ", "How ", "are ", "you?"])
        self.assertEqual(list(session("hi")), ["Hello ", "there. ", "How ", "are ", "you?"])

    def test_continue_without_query(self):
        session, _ = self.make(["answer"])
        session.conversation.add_user_message("question")
        self.assertEqual("".join(session()), "answer")
        self.assertEqual([m["role"] for m in session.conversation.messages], ["user", "assistant"])

    def test_empty_reply_records_nothing(self):
        session, _ = self.make([])
        self.assertEqual(list(session("hello")), [])
        self.assertEqual([m["role"] for m in session.conversation.messages], ["user"])

    def test_unanswered_user_message_is_extended_not_duplicated(self):
        session, llm = self.make([], ["Paris is sunny, London is wet."])
        list(session("what's the weather in Paris"))
        list(session("and in London?"))
        self.assertEqual(
            llm.calls[1].messages, [{"role": "user", "content": "what's the weather in Paris\nand in London?"}]
        )

    def test_context_is_shown_to_the_model_but_not_stored_as_content(self):
        session, llm = self.make(["ok"])
        list(session("stop", context="(User interrupted you.)"))
        self.assertEqual(llm.calls[0].messages[0]["content"], "(User interrupted you.)\nstop")
        self.assertEqual(session.conversation.history[0].content, "stop")

    def test_generation_parameters_reach_the_llm(self):
        session, llm = self.make(["ok"])
        list(session("hi", temperature=0.2))
        self.assertEqual(llm.calls[0].kwargs, {"temperature": 0.2})

    def test_session_state_reaches_the_llm(self):
        session, llm = self.make(["ok"])
        list(session("hi"))
        self.assertIs(llm.calls[0].state, session.state)

    def test_llm_failure_rolls_the_turn_back(self):
        def explode(conversation):
            raise RuntimeError("backend down")

        session, _ = self.make(["first reply"], explode)
        list(session("one"))
        with self.assertRaises(LLMError):
            list(session("two"))
        self.assertEqual([m["content"] for m in session.conversation.messages], ["one", "first reply"])
        self.assertFalse(session._lock.locked())


class TestSessionTools(SessionCase):
    """Tool dispatch, result handling and the pass budget."""

    def test_tool_call_then_answer(self):
        seen = []

        def my_tool(query: str) -> ToolResult:
            seen.append(query)
            return ToolResult(result={"result": f"result for {query}"})

        session, llm = self.make(
            ["Let me check...", *gemma_call("my_tool", '{query:<|"|>hello<|"|>}')],
            ["Final answer."],
            tools=[Tool.from_callable("my_tool", my_tool)],
            decoder=GeneralDecoder(),
        )
        self.assertEqual("".join(session("find hello")), "Let me check...Final answer.")
        self.assertEqual(seen, ["hello"])
        self.assertEqual([c.tool_choice for c in llm.calls], ["auto", "none"])
        self.assertEqual(
            [(m["role"], m["content"]) for m in session.conversation.messages],
            [
                ("user", "find hello"),
                ("assistant", "Let me check..."),
                ("tool", 'my_tool: {"result": "result for hello"}'),
                ("tool", "Now, generate an answer based only on the returned responses."),
                ("assistant", "Final answer."),
            ],
        )

    def test_openai_format_pairs_calls_with_results(self):
        session, llm = self.make(
            [ToolCall("lookup", {"q": "x"}, id="call_1")],
            ["done"],
            tools=[Tool.from_callable("lookup", lambda q: {"answer": 42})],
        )
        list(session("go"))
        second_pass = llm.calls[1].openai_messages
        self.assertEqual(second_pass[1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(second_pass[1]["tool_calls"][0]["function"], {"name": "lookup", "arguments": '{"q": "x"}'})
        self.assertEqual(second_pass[2]["role"], "tool")
        self.assertEqual(second_pass[2]["tool_call_id"], "call_1")
        self.assertIn('{"answer": 42}', second_pass[2]["content"])
        self.assertIn("Now, generate an answer", second_pass[2]["content"], "notes fold into the tool result")
        self.assertEqual(len(second_pass), 3)

    def test_same_tool_called_twice_keeps_both_results(self):
        session, _ = self.make(
            [ToolCall("weather", {"city": "Paris"}), ToolCall("weather", {"city": "London"})],
            ["ok"],
            tools=[Tool.from_callable("weather", lambda city: {"city": city})],
        )
        list(session("weather in paris and london"))
        self.assertEqual(tool_messages(session)[:2], ['weather: {"city": "Paris"}', 'weather: {"city": "London"}'])

    def test_async_tool(self):
        async def lookup(q: str):
            await asyncio.sleep(0)
            return {"answer": 42}

        session, _ = self.make(
            [ToolCall("lookup", {"q": "x"})], ["final"], tools=[Tool.from_callable("lookup", lookup)]
        )
        self.assertEqual(list(session("go")), ["final"])
        self.assertEqual(tool_messages(session)[0], 'lookup: {"answer": 42}')

    def test_tool_speech_is_the_answer(self):
        """A tool that says its own result needs no second model pass."""
        session, llm = self.make(
            [ToolCall("lights", {})],
            ["The lights have been turned on for you."],
            tools=[Tool.from_callable("lights", lambda: ToolResult({"ok": True}, speech="Lights are on."))],
        )
        self.assertEqual(list(session("lights on")), ["Lights are on."])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(session.conversation.messages[-1], {"role": "assistant", "content": "Lights are on."})

    def test_mixed_speech_and_silent_tools_get_one_assistant_turn(self):
        tools = [
            Tool.from_callable("lights", lambda: ToolResult({"ok": True}, speech="Lights are on.")),
            Tool.from_callable("weather", lambda: {"sky": "clear"}),
        ]
        session, llm = self.make([ToolCall("lights", {}), ToolCall("weather", {})], ["And it is clear out."], tools=tools)
        self.assertEqual(list(session("lights and weather")), ["Lights are on.", "And it is clear out."])
        self.assertEqual(
            session.conversation.messages[-1],
            {"role": "assistant", "content": "Lights are on. And it is clear out."},
        )

    def test_tool_error_is_reported_to_the_model(self):
        def broken_tool(**kwargs):
            raise RuntimeError("tool exploded")

        session, _ = self.make(
            gemma_call("broken_tool", "{x:1}"),
            ["Sorry."],
            tools=[Tool.from_callable("broken_tool", broken_tool)],
            decoder=GeneralDecoder(),
        )
        list(session("do something"))
        self.assertEqual(
            tool_messages(session),
            [
                "broken_tool: Tool Error: tool exploded",
                "One or more tools failed to execute. "
                "Please inform the user of the error and suggest an alternative action.",
            ],
        )

    def test_unknown_tool(self):
        session, _ = self.make(gemma_call("nonexistent", "{args:{}}"), decoder=GeneralDecoder())
        self.assertIn("persistent error", "".join(session("test")))
        self.assertIn("'nonexistent' is not a recognized tool", tool_messages(session)[0])

    def test_unparseable_tool_call(self):
        session, _ = self.make(gemma_call("bad_tool", "{invalid!!!}}"), decoder=GeneralDecoder())
        self.assertIn("persistent error", "".join(session("do it")))
        self.assertIn("Could not parse tool call", tool_messages(session)[0])

    def test_tool_call_during_answer_pass_is_ignored(self):
        ran = []
        session, _ = self.make(
            ["Here you go. ", ToolCall("t", {})],
            tools=[Tool.from_callable("t", lambda: ran.append(1))],
            max_tool_iterations=0,
        )
        self.assertEqual(list(session("hi")), ["Here you go. "])
        self.assertEqual(ran, [])

    def test_tool_timeout(self):
        def slow_tool(**kwargs):
            time.sleep(1.5)
            return "too late"

        session, _ = self.make(
            gemma_call("slow_tool", "{x:1}"),
            ["Sorry."],
            tools=[Tool.from_callable("slow_tool", slow_tool)],
            decoder=GeneralDecoder(),
            tool_timeout=0.2,
        )
        started = time.monotonic()
        list(session("do slow"))
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(tool_messages(session)[0], "slow_tool: Tool Error: slow_tool timed out")

    def test_tools_in_one_pass_run_concurrently(self):
        def nap():
            time.sleep(0.3)
            return {"ok": True}

        session, _ = self.make(
            [ToolCall("nap", {}), ToolCall("nap", {}), ToolCall("nap", {})],
            ["done"],
            tools=[Tool.from_callable("nap", nap)],
        )
        started = time.monotonic()
        list(session("go"))
        self.assertLess(time.monotonic() - started, 0.8)

    def test_events_expose_tool_activity(self):
        session, _ = self.make(
            ["Checking. ", ToolCall("lookup", {"q": "x"})],
            ["Done."],
            tools=[Tool.from_callable("lookup", lambda q: {"answer": 42})],
        )
        events = list(session.events("go"))
        self.assertEqual([type(e) for e in events], [TextDelta, ToolCallEvent, ToolResultEvent, TextDelta])
        self.assertEqual(events[1].call.name, "lookup")
        self.assertEqual(events[2].result.raw_result, {"answer": 42})
        self.assertIsNone(events[2].error)


class TestSessionConcurrency(SessionCase):
    """Turn serialisation, cancellation and reset."""

    def test_turns_do_not_interleave(self):
        order = []

        def slow(conversation):
            order.append("enter")
            time.sleep(0.05)
            order.append("exit")
            return ["x"]

        session, _ = self.make(slow)
        threads = [threading.Thread(target=lambda: list(session("hello"))) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(order, ["enter", "exit", "enter", "exit"])

    def test_lock_released_after_full_consumption(self):
        session, _ = self.make(["hello", " world"])
        list(session("Hi."))
        self.assertFalse(session._lock.locked())

    def test_closing_the_stream_ends_the_turn_and_keeps_what_was_said(self):
        session, _ = self.make(["hello", " world"])
        stream = session("Hi.")
        self.assertEqual(next(stream), "hello")
        stream.close()
        self.assertFalse(session._lock.locked())
        self.assertEqual(session.conversation.messages[-1], {"role": "assistant", "content": "hello"})

    def test_cancel_stops_the_turn_at_the_next_chunk(self):
        session, _ = self.make(["one ", "two ", "three"])
        stream = session("count")
        self.assertEqual(next(stream), "one ")
        session.cancel()
        self.assertEqual(list(stream), [])
        self.assertEqual(session.conversation.messages[-1], {"role": "assistant", "content": "one "})
        self.assertEqual("".join(session("again")), "one two three", "the next turn is unaffected")

    def test_reset_does_not_wait_for_the_turn_in_flight(self):
        session, _ = self.make(["partial ", "answer"])
        session.conversation.set_system_message("You are helpful.")
        stream = session("hi")
        next(stream)

        worker = threading.Thread(target=session.reset)
        worker.start()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive(), "reset() blocked behind the running turn")

        self.assertEqual(list(stream), [], "the reset turn stops at its next chunk")
        self.assertEqual(session.conversation.messages, [{"role": "system", "content": "You are helpful."}])


class TestInstructionsRulesAndTools(SessionCase):
    """What the model is told and may do is set on the session."""

    LIGHTS = Tool(name="lights", description="", instruction="Call 'lights' to change the lighting.")

    def test_the_model_receives_instructions_tool_instructions_and_rules(self):
        session, llm = self.make(["ok"], instructions="You are a butler.", rules=["Be brief."], tools=[self.LIGHTS])
        list(session("hello"))
        prompt = "You are a butler.\n\nRules:\n- Call 'lights' to change the lighting.\n- Be brief."
        self.assertEqual(llm.calls[0].messages[0], {"role": "system", "content": prompt})
        self.assertEqual(session.system_prompt, prompt)

    def test_a_bare_session_has_no_system_prompt(self):
        session, llm = self.make(["ok"])
        list(session("hello"))
        self.assertEqual(session.system_prompt, "")
        self.assertEqual([m["role"] for m in llm.calls[0].messages], ["user"])

    def test_they_can_be_changed_between_turns(self):
        session, llm = self.make(["ok"])
        session.instructions = "You are a butler."
        session.rules = ("Be brief.",)
        session.tools.add(self.LIGHTS, Tool(name="plain", description=""))
        self.assertEqual((session.instructions, session.rules, list(session.tools)), (
            "You are a butler.", ["Be brief."], ["lights", "plain"],
        ))

        list(session("hello"))
        self.assertIn("- Call 'lights' to change the lighting.\n- Be brief.", llm.calls[0].messages[0]["content"])

        session.tools = [Tool(name="plain", description="")]
        session.rules.clear()
        list(session("again"))
        self.assertEqual(llm.calls[1].messages[0]["content"], "You are a butler.")

    def test_they_are_the_conversations_own(self):
        """One copy: a conversation set up by hand and the session's view of it cannot disagree."""
        conversation = Conversation()
        conversation.set_system_message("You are a butler.")
        conversation.tools = [self.LIGHTS]
        session = Session(llm=ScriptedLLM([]), conversation=conversation, rules=["Be brief."])
        self.addCleanup(session.close)

        self.assertEqual(session.instructions, "You are a butler.")
        self.assertIs(session.tools, conversation.tools)
        self.assertEqual(conversation.rules, ["Be brief."])

    def test_constructor_arguments_add_to_a_conversation_that_brings_its_own(self):
        conversation = Conversation()
        conversation.set_system_message("old")
        conversation.tools = [Tool(name="plain", description="")]
        session = Session(llm=ScriptedLLM([]), conversation=conversation, instructions="new", tools=[self.LIGHTS])
        self.addCleanup(session.close)
        self.assertEqual((session.instructions, list(session.tools)), ("new", ["plain", "lights"]))

    def test_the_confirm_tool_adds_no_rule(self):
        tool = Tool.from_callable("unlock", lambda door: {"unlocked": door})
        tool.requires_permission = True
        session, llm = self.make(["ok"], instructions="You are a butler.", tools=[tool])
        list(session("hello"))
        self.assertIn("_confirm", session.tools)
        self.assertEqual(llm.calls[0].messages[0]["content"], "You are a butler.")


class TestSessionLifecycle(SessionCase):
    def test_reset_keeps_instructions_rules_and_tools_but_clears_the_rest(self):
        session, _ = self.make([], instructions="sys", rules=["Be brief."], tools=[Tool(name="t", description="")])
        session.conversation.add_user_message("old")
        session.state.user.set("k", "v")
        session.state.tools.set_permission("t", True)

        session.reset()

        self.assertEqual(session.conversation.history, [])
        self.assertEqual((session.instructions, session.rules, list(session.tools)), ("sys", ["Be brief."], ["t"]))
        self.assertIsNone(session.state.user.get("k"))
        self.assertIsNone(session.state.tools.get_permission("t"))

    def test_reset_with_new_conversation_replaces_everything(self):
        session, _ = self.make([], tools=[Tool(name="t", description="")])
        replacement = Conversation()
        replacement.add_user_message("new")
        session.reset(replacement)
        self.assertIs(session.conversation, replacement)
        self.assertEqual(session.conversation.tools, {})

    def test_empty_conversation_argument_is_kept(self):
        conversation = Conversation()
        session = Session(llm=ScriptedLLM([]), conversation=conversation)
        self.addCleanup(session.close)
        self.assertIs(session.conversation, conversation)

    def test_save_load_round_trip(self):
        session, llm = self.make([])
        session.conversation.add_user_message("hello")
        session.state.user.set("note", "test")

        with tempfile.TemporaryDirectory() as tmp:
            session.save(tmp)
            self.assertTrue(os.path.exists(os.path.join(tmp, "conversation.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "state.json")))
            restored = Session.load(tmp, llm=llm, max_tool_iterations=3)
            self.addCleanup(restored.close)

        self.assertEqual(restored.conversation.messages, [{"role": "user", "content": "hello"}])
        self.assertEqual(restored.state.user.get("note"), "test")
        self.assertEqual(restored.max_tool_iterations, 3)

    def test_load_restores_instructions_and_rules_and_takes_the_tools_again(self):
        session, llm = self.make([], instructions="You are a butler.", rules=["Be brief."])
        session.tools.add(Tool(name="lights", description=""))
        session.state.model.runtime["cache"] = object()

        with tempfile.TemporaryDirectory() as tmp:
            session.save(tmp)
            self.assertEqual(sorted(os.listdir(tmp)), ["conversation.json", "state.json"])
            restored = Session.load(tmp, llm=llm, tools=[Tool(name="lights", description="")])
            self.addCleanup(restored.close)

        self.assertEqual((restored.instructions, restored.rules), ("You are a butler.", ["Be brief."]))
        self.assertEqual(list(restored.tools), ["lights"])

    def test_close_stops_tool_caller(self):
        session = Session(llm=ScriptedLLM([]))
        thread = session.tool_caller._loop_thread
        self.assertTrue(thread.is_alive())
        session.close()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_context_manager_closes(self):
        with Session(llm=ScriptedLLM([])) as session:
            thread = session.tool_caller._loop_thread
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_tool_caller_requires_start(self):
        caller = ToolCaller()
        with self.assertRaises(LLMError):
            caller.submit(Tool.from_callable("t", lambda: None), {})


class TestSessionConfirm(SessionCase):
    """ToolChoice handling, the _confirm tool and permissions."""

    @staticmethod
    def unlock_tool(ran):
        def unlock(door: str):
            ran.append(door)
            return {"unlocked": door}

        tool = Tool.from_callable("unlock", unlock)
        tool.requires_permission = True
        return tool

    @staticmethod
    def confirm(tool_name, choice):
        """A script that answers whatever choice is pending for ``tool_name``."""

        def script(conversation):
            notice = next(m.content for m in reversed(conversation.history) if m.name == tool_name)
            uid = notice.split('"uid": "')[1].split('"')[0]
            return [ToolCall("_confirm", {"uid": uid, "choice": choice})]

        return script

    def test_confirm_tool_offered_only_when_needed(self):
        def choose(x: int) -> ToolChoice:
            return ToolChoice(result={"x": [x]})

        with_choice, _ = self.make([], tools=[Tool.from_callable("choose", choose)])
        list(with_choice("go"))
        self.assertIsNotNone(with_choice.conversation.tools["_confirm"].callback)

        without, _ = self.make([], tools=[Tool.from_callable("plain", lambda x: {"x": x})])
        list(without("go"))
        self.assertNotIn("_confirm", without.conversation.tools)

    def test_existing_confirm_tool_is_left_alone(self):
        tools = [Tool(name="_confirm", description="existing"), self.unlock_tool([])]
        session, _ = self.make([], tools=tools)
        list(session("go"))
        self.assertEqual(session.conversation.tools["_confirm"].description, "existing")

    def test_choice_with_speech_asks_the_user_directly(self):
        def book(x: int) -> ToolChoice:
            return ToolChoice(result={"slot": ["1", "2", "3"]}, speech="Which slot?")

        session, llm = self.make([ToolCall("book", {"x": 5})], ["unused"], tools=[Tool.from_callable("book", book)])
        self.assertEqual(list(session("choose")), ["Which slot?"])
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("Ask the user, then call _confirm.", tool_messages(session))
        self.assertEqual(session.conversation.messages[-1], {"role": "assistant", "content": "Which slot?"})

    def test_choice_without_speech_lets_the_model_ask(self):
        def book(x: int) -> ToolChoice:
            return ToolChoice(result={"slot": ["1", "2", "3"]})

        session, llm = self.make(
            [ToolCall("book", {"x": 5})], ["Which slot would you like?"], tools=[Tool.from_callable("book", book)]
        )
        self.assertEqual(list(session("choose")), ["Which slot would you like?"])
        self.assertIn("Ask the user, then call _confirm.", tool_messages(session))

    def test_choice_resolved_on_the_next_turn(self):
        booked = []

        def book(x: int, slot: str = "") -> ToolResult | ToolChoice:
            if not slot:
                return ToolChoice(result={"slot": ["1", "2"]}, speech="Which slot?")
            booked.append((x, slot))
            return ToolResult({"booked": slot})

        session, _ = self.make(
            [ToolCall("book", {"x": 5})],
            self.confirm("book", {"slot": "2"}),
            ["Booked."],
            tools=[Tool.from_callable("book", book)],
        )
        list(session("book it"))
        self.assertEqual("".join(session("slot two")), "Booked.")
        self.assertEqual(booked, [(5, "2")])
        self.assertFalse(session.state.tools.has_pending())

    def test_permission_is_requested_then_remembered(self):
        ran = []
        session, _ = self.make(
            [ToolCall("unlock", {"door": "front"})],
            self.confirm("unlock", {"allow": True, "remember": True}),
            ["Unlocked."],
            [ToolCall("unlock", {"door": "back"})],
            ["Unlocked again."],
            tools=[self.unlock_tool(ran)],
        )
        self.assertEqual(list(session("unlock the front door")), ["Permission requested to run unlock."])
        self.assertEqual(ran, [])

        list(session("yes, always"))
        self.assertEqual(ran, ["front"])
        self.assertTrue(session.state.tools.get_permission("unlock"))

        list(session("unlock the back door"))
        self.assertEqual(ran, ["front", "back"], "a remembered permission must not be asked for again")

    def test_permission_decorator(self):
        ran = []

        @permission
        def wipe(target: str):
            ran.append(target)
            return {"wiped": target}

        session, _ = self.make(
            [ToolCall("wipe", {"target": "cache"})],
            self.confirm("wipe", {"allow": True}),
            ["Done."],
            tools=[Tool.from_callable("wipe", wipe)],
        )
        list(session("wipe the cache"))
        self.assertEqual(ran, [])
        list(session("yes"))
        self.assertEqual(ran, ["cache"])
        self.assertIsNone(session.state.tools.get_permission("wipe"), "not remembered unless asked")

    def test_permission_denied_and_remembered(self):
        ran = []
        session, _ = self.make(
            [ToolCall("unlock", {"door": "front"})],
            self.confirm("unlock", {"allow": False, "remember": True}),
            [ToolCall("unlock", {"door": "back"})],
            tools=[self.unlock_tool(ran)],
        )
        list(session("unlock"))
        self.assertIn("Action unlock was cancelled.", list(session("never")))
        self.assertIn("Permission for unlock was denied.", list(session("unlock the back door")))
        self.assertEqual(ran, [])

    def test_model_cannot_answer_its_own_question(self):
        """With several tool passes per turn, _confirm in the same turn must not run the tool."""
        erased = []

        def erase(disk: str, sure: str = "") -> ToolResult | ToolChoice:
            if not sure:
                return ToolChoice(result={"sure": ["yes", "no"]})
            erased.append(disk)
            return ToolResult({"erased": disk})

        session, _ = self.make(
            [ToolCall("erase", {"disk": "C"})],
            self.confirm("erase", {"sure": "yes"}),
            ["Are you sure you want to erase C?"],
            tools=[Tool.from_callable("erase", erase)],
            max_tool_iterations=2,
        )
        self.assertEqual(list(session("erase my disk")), ["Are you sure you want to erase C?"])
        self.assertEqual(erased, [])
        self.assertTrue(any("The user has not answered yet" in m for m in tool_messages(session)))
        self.assertTrue(session.state.tools.has_pending(), "the question stays open for the user's real answer")

    def _pending(self, session, tool_name="test", arguments=None, options=None, uid="tc_1"):
        call = ToolCall(name=tool_name, arguments=arguments or {})
        choice = ToolChoice(result=options or {"allow": [True, False], "remember": [True, False]}, uid=uid)
        session.state.tools.add_pending(call, choice)
        return uid

    def test_on_confirm_runs_the_held_tool(self):
        captured = []

        def record(x: int) -> dict:
            captured.append(x)
            return {"x": x}

        session, _ = self.make([], tools=[Tool.from_callable("test", record)])
        uid = self._pending(session, arguments={"x": 42})
        result = asyncio.run(session._on_confirm(uid=uid, choice={"allow": True, "remember": False}))
        self.assertEqual(captured, [42])
        self.assertIn("42", result.result)
        self.assertIsNone(session.state.tools.get_permission("test"))

    def test_on_confirm_accepts_json_string_and_missing_remember(self):
        captured = []
        session, _ = self.make([], tools=[Tool.from_callable("test", lambda: captured.append(1))])
        asyncio.run(session._on_confirm(uid=self._pending(session), choice='{"allow": "true"}'))
        self.assertEqual(captured, [1])

    def test_on_confirm_mismatched_keys_raises_and_keeps_the_request(self):
        session, _ = self.make([], tools=[Tool.from_callable("test", lambda slot: None)])
        uid = self._pending(session, options={"slot": ["1", "2"]})
        with self.assertRaises(ToolError):
            asyncio.run(session._on_confirm(uid=uid, choice={"seat": "1"}))
        self.assertIsNotNone(session.state.tools.get_pending(uid))

    def test_on_confirm_permission_rejects_stray_keys(self):
        session, _ = self.make([], tools=[Tool.from_callable("test", lambda: None)])
        uid = self._pending(session)
        for choice in ({"remember": True}, {"allow": True, "door": "front"}):
            with self.subTest(choice=choice), self.assertRaises(ToolError):
                asyncio.run(session._on_confirm(uid=uid, choice=choice))

    def test_on_confirm_unknown_uid_raises(self):
        session, _ = self.make([])
        with self.assertRaises(ToolError):
            asyncio.run(session._on_confirm(uid="nonexistent", choice={"allow": True}))

    def test_on_confirm_uses_the_pending_calls_own_tool(self):
        ran = []
        tools = [
            Tool.from_callable("first", lambda: ran.append("first")),
            Tool.from_callable("second", lambda: ran.append("second")),
        ]
        session, _ = self.make([], tools=tools)
        asyncio.run(session._on_confirm(uid=self._pending(session, tool_name="first"), choice={"allow": True}))
        self.assertEqual(ran, ["first"])


class TestSessionContextHandler(SessionCase):
    def test_hooks_run_around_the_turn_and_on_close(self):
        handler = MagicMock()
        session, _ = self.make(["ok"], context_handler=handler)
        list(session("hi"))
        handler.pre_process.assert_called_once_with(session.conversation)
        handler.post_process.assert_called_once_with(session.conversation)

        previous = session.conversation
        session.reset()
        handler.finalize.assert_called_once_with(previous)


class TestToolRounds(SessionCase):
    """A round of tool calls reaches history whole, however the turn ends."""

    @staticmethod
    def history(session):
        return [(m.role.value, m.content) for m in session.conversation.history]

    def blocking_tool(self, name="slow", result=None):
        """A tool that reports when it starts and then waits to be released."""
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def run():
            started.set()
            release.wait(timeout=3)
            return result or {"ok": True}

        return Tool.from_callable(name, run), started, release

    def test_cancel_while_a_tool_runs_keeps_what_was_said_and_what_was_asked(self):
        tool, started, release = self.blocking_tool()
        session, llm = self.make(["On it. ", ToolCall("slow", {})], ["Done."], tools=[tool], tool_timeout=3)
        heard = []
        worker = threading.Thread(target=lambda: heard.extend(session("go")), daemon=True)
        worker.start()
        self.assertTrue(started.wait(timeout=2))

        session.cancel()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive(), "cancel() must not wait for the tool to finish")

        self.assertEqual(heard, ["On it. "])
        history = self.history(session)
        self.assertEqual(history[:2], [("user", "go"), ("assistant", "On it. ")])
        self.assertIn("slow was interrupted before it finished", history[2][1])

        release.set()
        list(session("next"))
        self.assertEqual(llm.calls[-1].messages[-1], {"role": "user", "content": "next"}, "not merged into 'go'")

    def test_closing_the_stream_at_a_tool_call_does_not_run_the_tool(self):
        ran = []
        session, _ = self.make([ToolCall("act", {})], tools=[Tool.from_callable("act", lambda: ran.append(1))])
        events = session.events("go")
        self.assertIsInstance(next(events), ToolCallEvent)
        events.close()

        self.assertEqual(ran, [])
        self.assertIn("act was not run", self.history(session)[2][1])
        self.assertFalse(session._lock.locked())

    def test_reset_during_a_tool_round_leaves_no_question_behind(self):
        gated = Tool.from_callable("unlock", lambda: {"unlocked": True})
        gated.requires_permission = True
        slow, started, release = self.blocking_tool()
        session, _ = self.make([ToolCall("unlock", {}), ToolCall("slow", {})], tools=[gated, slow], tool_timeout=3)
        worker = threading.Thread(target=lambda: list(session("go")), daemon=True)
        worker.start()
        self.assertTrue(started.wait(timeout=2))

        session.reset()
        worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertFalse(session.state.tools.has_pending(), "the permission request died with its conversation")
        self.assertEqual(session.conversation.history, [])

    def test_results_come_before_the_errors_of_unparseable_calls(self):
        lookup = Tool.from_callable("lookup", lambda: {"answer": 42})
        session, llm = self.make([MalformedToolCall(raw="{"), ToolCall("lookup", {})], ["done"], tools=[lookup])
        list(session("go"))
        sent = llm.calls[1].openai_messages
        self.assertEqual([m["role"] for m in sent], ["user", "assistant", "tool"], "each call is answered first")
        self.assertIn("Could not parse tool call", sent[2]["content"])

    def test_giving_up_after_a_tool_spoke_is_recorded_as_one_reply(self):
        speaks = Tool.from_callable("speaks", lambda: ({"ok": True}, "Lights are on."))
        quiet = Tool.from_callable("quiet", lambda: {"sky": "clear"})
        session, _ = self.make(
            [ToolCall("speaks", {}), ToolCall("quiet", {})], [ToolCall("quiet", {})], tools=[speaks, quiet]
        )
        with self.assertLogs("voxpipe.llm.session", level="WARNING"):
            heard = list(session("go"))

        self.assertEqual(heard[0], "Lights are on.")
        roles = [role for role, _ in self.history(session)]
        self.assertEqual(roles[-2:], ["tool", "assistant"], "two assistant messages in a row break chat templates")
        self.assertTrue(session.conversation.last_message.content.startswith("Lights are on. "))

    def test_a_failed_turn_restores_the_unanswered_message_it_extended(self):
        def backend_down(conversation):
            raise RuntimeError("backend down")

        session, _ = self.make([], backend_down, ["ok"])
        list(session("one"))  # no reply: the next query extends this message
        with self.assertRaises(LLMError):
            list(session("two", context="(User interrupted you.)"))

        unanswered = session.conversation.last_message
        self.assertEqual((unanswered.content, unanswered.context), ("one", None))
        self.assertEqual("".join(session("two")), "ok")
        self.assertEqual(self.history(session)[0], ("user", "one\ntwo"))

    def test_tool_choice_governs_the_tool_passes_only(self):
        tool = Tool.from_callable("lookup", lambda: {"answer": 42})
        session, llm = self.make([ToolCall("lookup", {})], ["done"], tools=[tool])
        list(session("go", tool_choice="required"))
        self.assertEqual([call.tool_choice for call in llm.calls], ["required", "none"])

    def test_context_needs_a_query(self):
        session, _ = self.make(["ok"])
        with self.assertRaises(ValueError):
            list(session(context="(User interrupted you.)"))
        self.assertFalse(session._lock.locked())


class TestFollowUpQuestions(SessionCase):
    def test_a_question_asked_while_answering_another_belongs_to_the_tool(self):
        sent = []

        def send(to: str, body: str = ""):
            if not body:
                return ToolChoice({"body": "What should the message say?"}, speech="What should it say?")
            sent.append((to, body))
            return {"sent": to}

        tool = Tool.from_callable("send", send)
        tool.requires_permission = True
        confirm = TestSessionConfirm.confirm
        session, _ = self.make(
            [ToolCall("send", {"to": "bob"})],
            confirm("send", {"allow": True}),
            confirm("_confirm", {"body": "hi bob"}),
            ["Sent."],
            tools=[tool],
        )

        self.assertEqual(list(session("message bob")), ["Permission requested to run send."])
        self.assertEqual(list(session("yes")), ["What should it say?"])
        self.assertEqual("".join(session("say hi bob")), "Sent.")

        self.assertEqual(sent, [("bob", "hi bob")])
        self.assertFalse(session.state.tools.has_pending())

    def test_confirm_answers_to_the_session_now_running_the_conversation(self):
        gated = Tool.from_callable("unlock", lambda: {"unlocked": True})
        gated.requires_permission = True
        first, _ = self.make([], tools=[gated])
        list(first("hello"))
        self.assertIs(first.tools["_confirm"].callback.__self__, first)

        second = Session(llm=ScriptedLLM([]), conversation=first.conversation)
        self.addCleanup(second.close)
        list(second("hello again"))
        self.assertIs(second.tools["_confirm"].callback.__self__, second)

    def test_a_confirm_tool_supplied_by_the_application_is_left_alone(self):
        own = Tool.from_callable("_confirm", lambda uid, choice: {"ok": True})
        gated = Tool.from_callable("unlock", lambda: {"unlocked": True})
        gated.requires_permission = True
        session, _ = self.make([], tools=[own, gated])
        list(session("hello"))
        self.assertIs(session.tools["_confirm"], own)


class TestSessionConstruction(SessionCase):
    def test_rules_are_not_repeated_when_a_saved_session_is_reloaded_with_them(self):
        session, llm = self.make([], rules=["Be brief."])
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(3):
                session.save(tmp)
                session = Session.load(tmp, llm=llm, rules=["Be brief."])
                self.addCleanup(session.close)
        self.assertEqual(session.rules, ["Be brief."])

    def test_a_single_rule_may_be_given_as_a_string(self):
        session, _ = self.make([], rules="Be brief.")
        self.assertEqual(session.rules, ["Be brief."])

    def test_tools_may_be_given_as_a_mapping(self):
        tool = Tool(name="lights", description="")
        session = Session(llm=ScriptedLLM([]), tools={"lights": tool})
        self.addCleanup(session.close)
        self.assertEqual(dict(session.tools), {"lights": tool})

    def test_a_failed_load_leaves_no_tool_loop_running(self):
        before = {t for t in threading.enumerate() if t.name == "voxpipe-tools"}
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "state.json"), "w", encoding="utf-8") as f:
                f.write("{not json")
            with self.assertRaises(ValueError):
                Session.load(tmp, llm=ScriptedLLM([]))
        self.assertEqual({t for t in threading.enumerate() if t.name == "voxpipe-tools"}, before)

    def test_state_that_cannot_be_saved_leaves_the_saved_conversation_untouched(self):
        session, _ = self.make([])
        session.conversation.add_user_message("first")
        with tempfile.TemporaryDirectory() as tmp:
            session.save(tmp)
            session.conversation.add_user_message("second")
            session.state.user.set("not json", object())
            with self.assertRaises(TypeError):
                session.save(tmp)
            self.assertEqual(len(Conversation.load(os.path.join(tmp, "conversation.json")).history), 1)


if __name__ == "__main__":
    unittest.main()
