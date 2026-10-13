"""Pipeline tests: a real Session over a scripted LLM, with fake audio components."""
import os
import queue
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from voxpipe.core.config import load_config
from voxpipe.core.exceptions import ASRError, ProviderError
from voxpipe.llm.context import MemoryHandler
from voxpipe.llm.session import Session
from voxpipe.llm.tools import Tool, ToolCall
from voxpipe.pipeline.pipeline import Pipeline
from voxpipe.storage import RAMStorage, SQLiteStorage

from ..llm._fakes import ScriptedLLM


class FakeTTS:
    """Records what it is asked to say; like the real one, it refuses text from a run of speech that was cut."""

    def __init__(self):
        self.spoken = []
        self.refused = []
        self.interrupts = 0
        self.is_speaking = False
        self.started = self.stopped = False

    @property
    def generation(self):
        return self.interrupts

    def __call__(self, text, *, generation=None, **options):
        (self.spoken if generation in (None, self.generation) else self.refused).append(text)

    def interrupt(self):
        self.interrupts += 1

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakeASR:
    """A recogniser whose "microphone" is the test: ``hear`` an utterance and it is transcribed."""

    def __init__(self):
        self.vad = SimpleNamespace(on_speech_onset=None, on_audio_level=None)
        self.is_muted = False
        self.started = self.stopped = False
        self._heard = queue.Queue()

    def hear(self, text):
        self._heard.put(text)

    def mute(self):
        self.is_muted = True

    def unmute(self):
        self.is_muted = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        self._heard.put(None)

    def __iter__(self):
        while (text := self._heard.get()) is not None:
            yield text


class FakeHotkeys:
    def __init__(self):
        self.actions = {}
        self.started = self.stopped = False

    def register(self, hotkey, action):
        self.actions[hotkey] = action

    def unregister(self, hotkey):
        self.actions.pop(hotkey, None)

    def is_registered(self, hotkey):
        return hotkey in self.actions

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


class PipelineCase(unittest.TestCase):
    def make(self, *scripts, asr=True, tts=True, session_kwargs=None, **pipeline_kwargs):
        self.llm = ScriptedLLM(*scripts)
        self.session = Session(llm=self.llm, **(session_kwargs or {}))
        self.tts = FakeTTS() if tts else None
        self.asr = FakeASR() if asr else None
        pipe = Pipeline(self.session, asr=self.asr, tts=self.tts, **pipeline_kwargs)
        self.addCleanup(pipe.close)
        self.events = []
        for name in ("pipeline:state", "pipeline:error", "asr:utterance", "tool:call", "tool:result", "session:reset"):
            pipe.events.on(name, lambda *args, _name=name, **kwargs: self.events.append((_name, *args)))
        return pipe

    def states(self):
        return [event[1] for event in self.events if event[0] == "pipeline:state"]

    def last_prompt(self):
        return self.llm.calls[-1].messages[-1]["content"]


class TestResponding(PipelineCase):
    def test_reply_is_spoken_sentence_by_sentence(self):
        pipe = self.make(["Hello there. ", "How are ", "you today? ", "Good."])
        pipe.handle_transcript("hi")
        self.assertEqual(self.tts.spoken, ["Hello there.", "How are you today?", "Good."])
        self.assertEqual(self.states(), ["think", "speak", "speak", "speak", "idle"])
        self.assertEqual(self.last_prompt(), "hi")

    def test_first_sentence_is_spoken_before_generation_finishes(self):
        spoken_when_generating = []

        def script(conversation):
            yield "First sentence. "
            yield "Second "
            spoken_when_generating.extend(self.tts.spoken)
            yield "sentence."

        pipe = self.make(script)
        pipe.handle_transcript("go")
        self.assertEqual(spoken_when_generating, ["First sentence."])
        self.assertEqual(self.tts.spoken, ["First sentence.", "Second sentence."])

    def test_noise_is_dropped_before_the_llm(self):
        pipe = self.make(["unused"])
        for noise in ("", "   ", "...", "?!"):
            pipe.handle_transcript(noise)
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(self.tts.spoken, [])

    def test_works_without_tts(self):
        pipe = self.make(["Just text."], tts=False)
        with self.assertLogs("voxpipe.pipeline.pipeline", level="INFO") as logs:
            pipe.handle_transcript("hi")
        self.assertTrue(any("Just text." in line for line in logs.output))
        self.assertEqual(pipe.status["tts"], "unavailable")

    def test_new_reply_cuts_off_the_tail_of_the_previous_one(self):
        pipe = self.make(["One."])
        pipe.handle_transcript("first")
        pipe.handle_transcript("second")
        self.assertEqual(self.tts.interrupts, 2)

    def test_llm_failure_is_apologised_for_and_leaves_the_pipeline_usable(self):
        def explode(conversation):
            raise RuntimeError("backend down")

        pipe = self.make(explode, ["Back again."])
        with self.assertLogs("voxpipe.pipeline.pipeline", level="ERROR"):
            pipe.handle_transcript("hello")
        self.assertEqual(len(self.tts.spoken), 1)
        self.assertIn("Sorry", self.tts.spoken[0])
        self.assertEqual(self.states()[-1], "idle")
        self.assertEqual(pipe.status["llm"], "idle")
        self.assertTrue(any(event[0] == "pipeline:error" for event in self.events))
        self.assertEqual(self.session.conversation.history, [], "the failed turn left no trace")

        pipe.handle_transcript("hello again")
        self.assertEqual(self.tts.spoken[-1], "Back again.")

    def test_tool_activity_is_re_emitted(self):
        pipe = self.make([ToolCall("lookup", {"q": "x"})], ["Found it."])
        pipe.register_tools(Tool.from_callable("lookup", lambda q: {"answer": 42}))
        pipe.handle_transcript("look it up")
        calls = [event for event in self.events if event[0] == "tool:call"]
        results = [event for event in self.events if event[0] == "tool:result"]
        self.assertEqual(calls[0][1].name, "lookup")
        self.assertEqual(results[0][2].raw_result, {"answer": 42})
        self.assertIsNone(results[0][3])
        self.assertEqual(self.tts.spoken, ["Found it."])


class TestBargeIn(PipelineCase):
    def test_speech_onset_during_a_reply_stops_generation_and_playback(self):
        def script(conversation):
            yield "First sentence. "
            self.asr.vad.on_speech_onset()  # the user starts talking over the reply
            yield "Second sentence. "
            yield "Third sentence."

        pipe = self.make(script, ["Okay, stopping."])
        pipe.handle_transcript("tell me a story")

        self.assertEqual(self.tts.spoken, ["First sentence."])
        self.assertEqual(self.tts.interrupts, 2, "once for the new reply, once for the barge-in")
        self.assertEqual(self.session.conversation.messages[-1], {"role": "assistant", "content": "First sentence. "})
        self.assertEqual(pipe.status["llm"], "idle")

        pipe.handle_transcript("wait, stop")
        self.assertEqual(
            self.last_prompt(), '(User interrupted after: "First sentence.". Continue naturally.)\nwait, stop'
        )
        self.assertEqual(self.session.conversation.history[-2].content, "wait, stop")

    def test_speech_onset_while_audio_is_still_playing_interrupts_it(self):
        pipe = self.make(["A long answer."], ["Sure."])
        pipe.handle_transcript("question")
        self.tts.is_speaking = True  # generation is done, playback is not
        self.asr.vad.on_speech_onset()
        self.assertEqual(self.tts.interrupts, 2)

        pipe.handle_transcript("never mind")
        self.assertIn('(User interrupted after: "A long answer."', self.last_prompt())

    def test_ordinary_follow_up_is_not_marked_as_an_interruption(self):
        pipe = self.make(["It is sunny today."], ["Yes."])
        pipe.handle_transcript("what's the weather")
        self.asr.vad.on_speech_onset()  # long after the reply finished playing
        pipe.handle_transcript("will it stay that way")
        self.assertEqual(self.last_prompt(), "will it stay that way")
        self.assertEqual(self.tts.interrupts, 2, "only the two reply starts; nothing was interrupted")

    def test_stale_interrupt_does_not_cancel_the_next_reply(self):
        pipe = self.make(["Fine, thanks."])
        pipe._interrupt.set()
        pipe.handle_transcript("how are you")
        self.assertEqual(self.tts.spoken, ["Fine, thanks."])

    def test_interrupt_is_safe_when_idle(self):
        pipe = self.make(["ok"])
        pipe.interrupt()
        pipe.handle_transcript("hi")
        self.assertEqual(self.tts.spoken, ["ok"])


class TestCommands(PipelineCase):
    def test_exact_prefix_and_regex(self):
        pipe = self.make(["unused"])
        fired = []
        pipe.register_command("stop listening", lambda: fired.append("exact"))
        pipe.register_command("volume ", lambda: fired.append("prefix"), mode="prefix")
        pipe.register_command(r"timer \d+ minutes?", lambda: fired.append("regex"), mode="regex")

        for utterance in ("Stop listening.", "volume up", "Timer 5 minutes!"):
            pipe.handle_transcript(utterance)
        self.assertEqual(fired, ["exact", "prefix", "regex"])
        self.assertEqual(self.llm.calls, [], "commands bypass the LLM")

        pipe.unregister_command("stop listening")
        pipe.handle_transcript("stop listening")
        self.assertEqual(len(self.llm.calls), 1)

    def test_unknown_mode(self):
        pipe = self.make([])
        with self.assertRaises(ValueError):
            pipe.register_command("x", lambda: None, mode="fuzzy")

    def test_commands_match_after_a_previous_reply(self):
        pipe = self.make(["It is sunny today."])
        fired = []
        pipe.register_command("stop listening", lambda: fired.append(1))
        pipe.handle_transcript("what's the weather")
        self.tts.is_speaking = True
        self.asr.vad.on_speech_onset()
        pipe.handle_transcript("stop listening")
        self.assertEqual(fired, [1])


class TestToolsAndSystemPrompt(PipelineCase):
    def test_system_prompt_lists_tool_instructions(self):
        pipe = self.make([])
        first = Tool.from_callable("lights", lambda: None)
        first.instruction = "Call  lights when\n asked."
        second = Tool.from_callable("weather", lambda: None)
        second.instruction = "Call weather for forecasts."

        pipe.register_tools(first)
        pipe.register_tools([second])

        self.assertEqual(
            self.session.system_prompt,
            "You are a voice-controlled assistant. Respond conversationally.\n"
            "\n"
            "Rules:\n"
            "- Call lights when asked.\n"
            "- Call weather for forecasts.\n"
            "- If the user's message seems incomplete or cut off, ask what they meant.\n"
            "- Do not describe your actions or announce tool usage.\n"
            "- Respond naturally, as if you already know the answer.",
        )
        self.assertEqual(set(self.session.tools), {"lights", "weather"})

    def test_a_session_with_its_own_instructions_gets_no_default_ones(self):
        session = Session(llm=ScriptedLLM([]), instructions="You are a pirate.")
        self.addCleanup(session.close)
        pipe = Pipeline(session)
        tool = Tool.from_callable("lights", lambda: None)
        tool.instruction = "Call lights."
        pipe.register_tools(tool)
        self.assertEqual((session.instructions, session.rules), ("You are a pirate.", []))
        self.assertEqual(session.system_prompt, "You are a pirate.\n\nRules:\n- Call lights.")

    def test_a_session_with_only_rules_of_its_own_gets_no_default_ones(self):
        session = Session(llm=ScriptedLLM([]), rules=["Speak like a pirate."])
        self.addCleanup(session.close)
        Pipeline(session)
        self.assertEqual(session.system_prompt, "Rules:\n- Speak like a pirate.")

    def test_tools_without_a_callback_are_flagged(self):
        pipe = self.make([])
        with self.assertLogs("voxpipe.pipeline.pipeline", level="WARNING"):
            pipe.register_tools(Tool(name="ghost", description="does nothing"))

    def test_reset_keeps_tools_and_prompt(self):
        pipe = self.make(["Hi."])
        pipe.register_tools(Tool.from_callable("lights", lambda: None))
        prompt = self.session.system_prompt
        pipe.handle_transcript("hello")

        pipe.reset()

        self.assertEqual(self.session.conversation.messages, [{"role": "system", "content": prompt}])
        self.assertIn("lights", self.session.tools)
        self.assertIn(("session:reset",), self.events)


class TestHotkeys(PipelineCase):
    def test_no_dispatcher_without_hotkeys(self):
        pipe = self.make([])
        self.assertIsNone(pipe._hotkeys)
        self.assertFalse(self.asr.is_muted)
        self.assertEqual(pipe.status["asr"], "listening")

    def test_push_to_talk(self):
        hotkeys = FakeHotkeys()
        pipe = self.make([], hotkeys=hotkeys, push_to_talk="<ctrl_r>+<shift_r>")
        self.assertTrue(self.asr.is_muted, "muted until the key is held")
        self.assertEqual(pipe.status["asr"], "muted")

        with hotkeys.actions["<ctrl_r>+<shift_r>"]():
            self.assertFalse(self.asr.is_muted)
        self.assertTrue(self.asr.is_muted)

        pipe.push_to_talk = None
        self.assertEqual(hotkeys.actions, {})
        self.assertFalse(self.asr.is_muted, "back to always-on listening")

    def test_press_to_reset(self):
        hotkeys = FakeHotkeys()
        pipe = self.make(["Hi."], hotkeys=hotkeys, press_to_reset="<ctrl_l>+<ctrl_r>")
        pipe.handle_transcript("hello")
        hotkeys.actions["<ctrl_l>+<ctrl_r>"]()
        self.assertEqual(self.session.conversation.history, [])

        pipe.press_to_reset = "<f9>"
        self.assertEqual(list(hotkeys.actions), ["<f9>"])


class TestRunLoop(PipelineCase):
    def test_run_answers_what_the_recogniser_hears_until_stopped(self):
        hotkeys = FakeHotkeys()
        pipe = self.make(["Hello back."], hotkeys=hotkeys)
        runner = threading.Thread(target=pipe.run, daemon=True)
        runner.start()

        self.asr.hear("hello")
        self.assertTrue(wait_until(lambda: self.tts.spoken == ["Hello back."]))
        self.assertTrue(self.asr.started and self.tts.started and hotkeys.started)

        pipe.stop()
        runner.join(timeout=3)
        self.assertFalse(runner.is_alive())
        self.assertTrue(self.asr.stopped and self.tts.stopped and hotkeys.stopped)
        self.assertFalse(self.session.tool_caller._loop_thread)

    def test_run_ends_when_the_recogniser_does(self):
        pipe = self.make(["ok"])
        runner = threading.Thread(target=pipe.run, daemon=True)
        runner.start()
        self.asr.stop()
        runner.join(timeout=3)
        self.assertFalse(runner.is_alive())

    def test_a_failing_turn_does_not_end_the_loop(self):
        pipe = self.make(["Still here."])
        pipe.register_command("crash", lambda: 1 / 0)
        runner = threading.Thread(target=pipe.run, daemon=True)
        runner.start()
        with self.assertLogs("voxpipe.pipeline.pipeline", level="ERROR"):
            self.asr.hear("crash")
            self.asr.hear("are you there")
            self.assertTrue(wait_until(lambda: self.tts.spoken == ["Still here."]))
        pipe.stop()
        runner.join(timeout=3)
        self.assertTrue(any(event[0] == "pipeline:error" for event in self.events))

    def test_user_can_talk_over_a_reply_from_the_recogniser_thread(self):
        release = threading.Event()

        def slow_story(conversation):
            yield "Once upon a time. "
            release.wait(timeout=3)
            yield "The end."

        pipe = self.make(slow_story, ["Okay."])
        runner = threading.Thread(target=pipe.run, daemon=True)
        runner.start()
        self.asr.hear("tell me a story")
        self.assertTrue(wait_until(lambda: self.tts.spoken == ["Once upon a time."]))

        self.asr.vad.on_speech_onset()  # arrives on another thread while the LLM is mid-generation
        release.set()
        self.assertTrue(wait_until(lambda: pipe.status["llm"] == "idle"))
        self.assertEqual(self.tts.spoken, ["Once upon a time."], "the rest of the story is never spoken")

        self.asr.hear("stop please")
        self.assertTrue(wait_until(lambda: self.tts.spoken[-1] == "Okay."))
        self.assertIn("(User interrupted after:", self.last_prompt())
        pipe.stop()
        runner.join(timeout=3)

    def test_run_requires_a_recogniser(self):
        pipe = self.make([], asr=False)
        with self.assertRaises(ASRError):
            pipe.run()
        self.assertEqual(pipe.status["asr"], "unavailable")

    def test_close_is_idempotent_and_survives_a_failing_component(self):
        pipe = self.make([])
        self.tts.stop = MagicMock(side_effect=RuntimeError("device gone"))
        with self.assertLogs("voxpipe.pipeline.pipeline", level="WARNING"):
            pipe.close()
        pipe.close()
        self.assertTrue(self.asr.stopped)
        self.tts.stop.assert_called_once()

    def test_context_manager_closes(self):
        with self.make([]) as pipe:
            pass
        self.assertTrue(self.asr.stopped)
        self.assertIs(pipe.asr, self.asr)


class TestMemory(PipelineCase):
    def test_recalled_memories_and_annotations_do_not_pollute_memory(self):
        storage = RAMStorage()
        storage.store("the user's cat is called Miso", role="user")
        handler = MemoryHandler(strategy=lambda conversation: 0, memory=storage)

        def script(conversation):
            yield "Miso, of course. "
            self.asr.vad.on_speech_onset()
            yield "She is lovely."

        pipe = self.make(script, ["Sorry."], session_kwargs={"context_handler": handler})
        pipe.handle_transcript("what is my cat called")
        self.assertIn("(Earlier: the user's cat is called Miso)", self.llm.calls[0].messages[-1]["content"])
        pipe.handle_transcript("no need to gush about it")
        pipe.close()

        archived = [record.content for _, record in storage._entries]
        self.assertEqual(
            archived,
            [
                "the user's cat is called Miso",
                "what is my cat called",
                "Miso, of course. ",
                "no need to gush about it",
                "Sorry.",
            ],
        )


class TestFromConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = load_config()
        history = base.conversation_history.model_copy(
            update={"db_path": os.path.join(self.tmp.name, "memory", "history.db"), "top_k": 5}
        )
        hotkeys = base.hotkeys.model_copy(update={"push_to_talk": "<f8>"})
        self.config = base.model_copy(update={"conversation_history": history, "hotkeys": hotkeys})
        self.asr, self.tts, self.llm = FakeASR(), FakeTTS(), ScriptedLLM(["Configured."])

    def build(self, config=None, llm=None, tts=None, **kwargs):
        """Run from_config with the model loaders replaced; ``llm`` / ``tts`` override how they behave."""

        def start(target, **behaviour):
            patcher = patch(target, **behaviour)
            self.addCleanup(patcher.stop)
            return patcher.start()

        start("voxpipe.asr.model.ASRProviders.create", return_value=self.asr)
        start("voxpipe.tts.model.TTSProviders.create", **(tts or {"return_value": self.tts}))
        self.create_llm = start("voxpipe.pipeline.pipeline.LLMProviders.create", **(llm or {"return_value": self.llm}))
        start("voxpipe.pipeline.hotkeys.HotkeyDispatcher", new=FakeHotkeys)
        pipe = Pipeline.from_config(config or self.config, **kwargs)
        self.addCleanup(pipe.close)
        return pipe

    def test_assembles_the_configured_pipeline(self):
        pipe = self.build()
        self.assertIs(pipe.asr, self.asr)
        self.assertIs(pipe.tts, self.tts)
        self.assertIs(pipe.session.llm, self.llm)
        self.assertEqual(pipe.session.max_tool_iterations, self.config.llm.max_tool_iterations)
        self.assertEqual(pipe.session.tool_timeout, self.config.llm.tool_timeout)

        handler = pipe.session.context_handler
        self.assertIsInstance(handler, MemoryHandler)
        self.assertIsInstance(pipe.memory, SQLiteStorage)
        self.assertIs(handler.memory, pipe.memory)
        self.assertEqual(handler.recall_top_k, 5)
        self.assertEqual(pipe.push_to_talk, "<f8>")
        self.assertTrue(self.asr.is_muted)

        pipe.handle_transcript("hello")
        self.assertEqual(self.tts.spoken, ["Configured."])

    def test_history_survives_a_restart(self):
        first = self.build()
        first.handle_transcript("my favourite colour is teal")
        first.close()

        self.llm = ScriptedLLM(["Teal."])
        second = self.build()
        second.handle_transcript("what is my favourite colour")
        self.assertIn("(Earlier: my favourite colour is teal)", self.llm.calls[0].messages[-1]["content"])

    def test_history_can_be_disabled(self):
        history = self.config.conversation_history.model_copy(update={"enabled": False})
        pipe = self.build(self.config.model_copy(update={"conversation_history": history}))
        self.assertIsNone(pipe.memory)
        self.assertIsNone(pipe.session.context_handler.memory)
        self.assertFalse(os.path.exists(os.path.dirname(history.db_path)))

    def test_hotkeys_can_be_disabled_or_overridden(self):
        disabled = self.config.model_copy(update={"hotkeys": self.config.hotkeys.model_copy(update={"enable": False})})
        pipe = self.build(disabled)
        self.assertIsNone(pipe.push_to_talk, "the configured hotkey is switched off")
        self.assertFalse(self.asr.is_muted)

        self.asr = FakeASR()
        explicit = self.build(disabled, push_to_talk="<f1>")
        self.assertEqual(explicit.push_to_talk, "<f1>", "an argument wins even over 'enable: false'")
        self.assertTrue(self.asr.is_muted)

        self.asr = FakeASR()
        overridden = self.build(push_to_talk="<f1>", press_to_reset="<f2>")
        self.assertEqual((overridden.push_to_talk, overridden.press_to_reset), ("<f1>", "<f2>"))

    def test_missing_audio_components_degrade_gracefully(self):
        with self.assertLogs("voxpipe.pipeline.pipeline", level="ERROR"):
            pipe = self.build(tts={"side_effect": ProviderError("no speakers")})
        self.assertIsNone(pipe.tts)
        self.assertIs(pipe.asr, self.asr)

    def test_llm_failure_is_fatal_and_releases_what_was_loaded(self):
        with self.assertRaises(ProviderError):
            self.build(llm={"side_effect": ProviderError("model file missing")})
        self.assertTrue(self.asr.stopped)
        self.assertTrue(self.tts.stopped)

    def test_bring_your_own_session(self):
        session = Session(llm=ScriptedLLM(["Mine."]))
        self.addCleanup(session.close)
        pipe = self.build(session=session)
        self.create_llm.assert_not_called()
        self.assertIs(pipe.session, session)
        self.assertIsNone(pipe.memory)
        self.assertIsNone(session.context_handler, "a supplied session keeps its own context handling")


class TestTakingTheFloor(PipelineCase):
    """Moments at which the user starts speaking, or the reply is stopped, around the start of a reply."""

    def test_a_sentence_arriving_after_the_barge_in_is_not_spoken(self):
        pipe = self.make(["First sentence. ", "Second sentence. ", "Third."])
        # The user barges in while the first sentence is being announced: after the pipeline's own
        # interruption check, before the sentence is handed to the synthesiser.
        pipe.events.on("tts:start", lambda sentence: self.asr.vad.on_speech_onset())
        pipe.handle_transcript("tell me a story")
        self.assertEqual(self.tts.spoken, [], "a run of speech that was cut takes no more sentences")
        self.assertEqual(self.tts.refused, ["First sentence."])

    def test_an_interrupt_before_the_turn_exists_stops_it_before_the_model_is_asked(self):
        ran = []
        pipe = self.make([ToolCall("unlock", {})], ["The door is unlocked."])
        pipe.register_tools(Tool.from_callable("unlock", lambda: ran.append("unlocked")))
        pipe.events.on("pipeline:state", lambda state: state == "think" and pipe.interrupt())

        pipe.handle_transcript("unlock the door")

        self.assertEqual((self.llm.calls, ran, self.tts.spoken), ([], [], []))

    def test_an_interrupt_during_the_first_pass_stops_before_any_tool_runs(self):
        ran = []

        def script(conversation):
            pipe.interrupt()
            yield ToolCall("unlock", {})

        pipe = self.make(script, ["The door is unlocked."])
        pipe.register_tools(Tool.from_callable("unlock", lambda: ran.append("unlocked")))
        pipe.handle_transcript("unlock the door")
        self.assertEqual((ran, self.tts.spoken), ([], []))
        self.assertEqual(len(self.llm.calls), 1)

    def test_an_utterance_the_user_has_already_talked_past_waits_for_what_follows(self):
        pipe = self.make(["Lights and fan are on."])
        heard_at = pipe._onsets
        self.asr.vad.on_speech_onset()  # the user is speaking again before this one is answered

        pipe._handle("turn on the lights", heard_at)
        self.assertEqual((self.llm.calls, self.tts.spoken), ([], []), "answering now would talk over them")

        pipe._handle("and the fan", pipe._onsets)
        self.assertEqual(self.last_prompt(), "turn on the lights\nand the fan")
        self.assertEqual(self.tts.spoken, ["Lights and fan are on."])

    def test_a_held_utterance_is_answered_when_what_followed_was_only_noise(self):
        pipe = self.make(["Lights are on."])
        heard_at = pipe._onsets
        self.asr.vad.on_speech_onset()
        pipe._handle("turn on the lights", heard_at)

        pipe._handle("...", pipe._onsets)  # the "speech" was a cough
        self.assertEqual(self.last_prompt(), "turn on the lights")
        self.assertEqual(self.tts.spoken, ["Lights are on."])

    def test_reset_forgets_the_interruption_and_anything_waiting(self):
        def script(conversation):
            yield "Please keep it secret. "
            self.asr.vad.on_speech_onset()
            yield "Never tell."

        pipe = self.make(script, ["Hi!"])
        pipe.handle_transcript("tell me a secret")
        heard_at = pipe._onsets
        self.asr.vad.on_speech_onset()
        pipe._handle("something half said", heard_at)

        pipe.reset()
        pipe.handle_transcript("hello there")
        self.assertEqual(self.last_prompt(), "hello there", "nothing of the forgotten conversation is carried over")

    def test_a_raising_onset_race_does_not_touch_a_replaced_list(self):
        pipe = self.make(["ok"])
        pipe._responding = True
        pipe._response_parts = []
        self.asr.vad.on_speech_onset()  # an empty list of parts is simply no hint
        self.assertIsNone(pipe._interrupted_at)


class TestCommandMatching(PipelineCase):
    def test_patterns_are_normalised_like_utterances(self):
        ran = []
        pipe = self.make(["never asked"])
        pipe.register_command("Stop listening.", lambda: ran.append("exact"))
        pipe.register_command("Volume ", lambda: ran.append("prefix"), mode="prefix")
        pipe.register_command(r"Set timer (\d+)", lambda: ran.append("regex"), mode="regex")

        for utterance in ("stop listening", "volume up!", "set timer 5"):
            pipe.handle_transcript(utterance)

        self.assertEqual(ran, ["exact", "prefix", "regex"])
        self.assertEqual(self.llm.calls, [])

    def test_an_utterance_listener_needs_only_one_argument(self):
        heard = []
        pipe = self.make(["ok"])
        pipe.events.on("asr:utterance", heard.append)
        with self.assertNoLogs("voxpipe.pipeline.events", level="ERROR"):
            pipe.handle_transcript("hello")
        self.assertEqual(heard, ["hello"])


class TestHotkeyAssignment(PipelineCase):
    class StrictHotkeys(FakeHotkeys):
        """Like the real dispatcher, refuses a combination it cannot parse."""

        @staticmethod
        def _check(hotkey):
            from voxpipe.core.exceptions import VoxpipeError

            if "<" not in hotkey and len(hotkey) > 1:
                raise VoxpipeError(f"Invalid hotkey {hotkey!r}")

        def register(self, hotkey, action):
            self._check(hotkey)
            super().register(hotkey, action)

        def unregister(self, hotkey):
            self._check(hotkey)
            super().unregister(hotkey)

        def is_registered(self, hotkey):
            self._check(hotkey)
            return super().is_registered(hotkey)

    def test_a_combination_that_cannot_be_used_changes_nothing(self):
        from voxpipe.core.exceptions import VoxpipeError

        hotkeys = self.StrictHotkeys()
        pipe = self.make([], hotkeys=hotkeys, push_to_talk="<f8>")
        with self.assertRaises(VoxpipeError):
            pipe.push_to_talk = "ctrl+k"  # a typo for "<ctrl>+k"

        self.assertEqual(pipe.push_to_talk, "<f8>")
        self.assertEqual(list(hotkeys.actions), ["<f8>"])
        pipe.push_to_talk = None  # and the pipeline is not stuck with the bad value
        self.assertFalse(self.asr.is_muted)

    def test_a_bad_first_assignment_leaves_the_microphone_open(self):
        from voxpipe.core.exceptions import VoxpipeError

        pipe = self.make([], hotkeys=self.StrictHotkeys())
        with self.assertRaises(VoxpipeError):
            pipe.push_to_talk = "ctrl+k"
        self.assertIsNone(pipe.push_to_talk)
        self.assertFalse(self.asr.is_muted)

    def test_two_actions_cannot_share_a_combination(self):
        from voxpipe.core.exceptions import VoxpipeError

        hotkeys = FakeHotkeys()
        pipe = self.make([], hotkeys=hotkeys, push_to_talk="<f8>")
        with self.assertRaisesRegex(VoxpipeError, "already in use"):
            pipe.press_to_reset = "<f8>"
        self.assertIsNone(pipe.press_to_reset)
        with hotkeys.actions["<f8>"]():
            self.assertFalse(self.asr.is_muted, "push-to-talk still owns the key")

    def test_reassigning_the_same_combination_is_allowed(self):
        hotkeys = FakeHotkeys()
        pipe = self.make([], hotkeys=hotkeys, push_to_talk="<f8>")
        pipe.push_to_talk = "<f8>"
        self.assertEqual(list(hotkeys.actions), ["<f8>"])

    def test_a_dispatcher_created_while_running_is_started(self):
        created = []

        class LateHotkeys(FakeHotkeys):
            def __init__(self):
                super().__init__()
                created.append(self)

        pipe = self.make(["ok"])
        runner = threading.Thread(target=pipe.run, daemon=True)
        runner.start()
        self.assertTrue(wait_until(lambda: pipe._running))
        with patch("voxpipe.pipeline.hotkeys.HotkeyDispatcher", new=LateHotkeys):
            pipe.push_to_talk = "<f8>"
        self.assertTrue(created and created[0].started, "or the key could never unmute the microphone")
        pipe.stop()
        runner.join(timeout=3)


class TestStoppingAndFailing(PipelineCase):
    def test_stop_leaves_waiting_utterances_unanswered(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def slow(conversation):
            yield "Thinking about it. "
            release.wait(timeout=3)
            yield "Done."

        pipe = self.make(slow, ["Should never be said."])
        runner = threading.Thread(target=pipe.run, daemon=True)
        runner.start()
        self.asr.hear("first question")
        self.assertTrue(wait_until(lambda: self.tts.spoken == ["Thinking about it."]))
        self.asr.hear("second question")
        self.asr.hear("third question")
        self.assertTrue(wait_until(lambda: pipe._transcripts.qsize() >= 2))

        pipe.stop()
        release.set()
        runner.join(timeout=3)

        self.assertFalse(runner.is_alive())
        self.assertEqual(len(self.llm.calls), 1, "the backlog is not worked through on the way out")

    def test_run_reports_a_recogniser_that_failed(self):
        class BrokenASR(FakeASR):
            def __iter__(self):
                yield "hello"
                raise RuntimeError("microphone unplugged")

        pipe = self.make(["Hi."])
        pipe.asr = self.asr = BrokenASR()
        with self.assertLogs("voxpipe.pipeline.pipeline", level="ERROR"), self.assertRaisesRegex(
            ASRError, "microphone unplugged"
        ):
            pipe.run()
        self.assertEqual(self.tts.spoken, ["Hi."], "what was heard before the failure is still answered")

    def test_text_can_be_submitted_while_running(self):
        pipe = self.make(["Typed reply."])
        runner = threading.Thread(target=pipe.run, daemon=True)
        runner.start()
        pipe.submit("typed question")
        self.assertTrue(wait_until(lambda: self.tts.spoken == ["Typed reply."]))
        self.assertEqual(self.last_prompt(), "typed question")
        pipe.stop()
        runner.join(timeout=3)

    def test_a_failing_release_does_not_hide_why_start_up_failed(self):
        asr, tts = FakeASR(), FakeTTS()
        asr.stop = MagicMock(side_effect=RuntimeError("stream already gone"))
        config = load_config()
        with patch("voxpipe.asr.model.ASRProviders.create", return_value=asr), patch(
            "voxpipe.tts.model.TTSProviders.create", return_value=tts
        ), patch("voxpipe.pipeline.pipeline.LLMProviders.create", side_effect=ProviderError("model file missing")):
            with self.assertLogs("voxpipe.pipeline.pipeline", level="WARNING"), self.assertRaisesRegex(
                ProviderError, "model file missing"
            ):
                Pipeline.from_config(config)
        self.assertTrue(tts.stopped, "the components after the failing one are still released")


class TestCommandLine(unittest.TestCase):
    def _specs(self, content):
        import json

        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        self.addCleanup(os.unlink, handle.name)
        with handle:
            json.dump(content, handle)
        return handle.name

    def test_tool_files_hold_one_tool_a_list_or_a_tools_object(self):
        from voxpipe.__main__ import _load_tools

        tool = {"name": "lights", "description": "Switch the lights."}
        for content in (tool, [tool], {"tools": [tool]}):
            with self.subTest(content=content):
                self.assertEqual([t.name for t in _load_tools(self._specs(content))], ["lights"])

    def test_a_malformed_tool_file_is_a_tool_error(self):
        from voxpipe.__main__ import _load_tools
        from voxpipe.core.exceptions import ToolError

        for content in (42, {"tools": None}, "lights"):
            with self.subTest(content=content), self.assertRaises(ToolError):
                _load_tools(self._specs(content))


if __name__ == "__main__":
    unittest.main()
