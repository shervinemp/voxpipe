"""End-to-end pipeline integration tests with the real model (no audio devices).

Verifies:
  - a turn completes within timing bounds (no hang)
  - the session is free again after a reply, complete or cut short
  - tool calls during knowledge queries don't deadlock
  - multiple utterances can be processed sequentially

These are non-deterministic (model output varies) and require the GGUF model.
"""
import time
import unittest

from voxpipe.llm.conversation import Conversation
from voxpipe.llm.model import GGUFLLM
from voxpipe.llm.session import Session
from voxpipe.llm.tools import Tool
from voxpipe.pipeline.pipeline import Pipeline

from .._integration import LLAMA, requires_model

pytestmark = requires_model("Gemma4E4B", LLAMA)

SYSTEM_PROMPT = (
    "You are a voice-controlled game assistant. "
    "Respond conversationally and naturally.\n\nRules:\n"
    "- Call 'retrieve' when the user asks about entities.\n"
    "- If the user seems incomplete, ask what they meant."
)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests that run the real model through the pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.llm = GGUFLLM.from_config("Gemma4E4B")
        cls.tool = Tool.from_callable("retrieve", lambda q: "test result")
        cls.tool.instruction = "Call when asked about entities."

    def _session(self, with_tool=True):
        conversation = Conversation()
        conversation.set_system_message(SYSTEM_PROMPT)
        if with_tool:
            conversation.tools = [self.tool]
        session = Session(llm=self.llm, conversation=conversation)
        self.addCleanup(session.close)
        return session

    def _timed(self, pipeline, *utterances):
        started = time.monotonic()
        for utterance in utterances:
            pipeline.handle_transcript(utterance)
        return time.monotonic() - started

    def test_greeting_completes_within_5s(self):
        elapsed = self._timed(Pipeline(self._session()), "Hi, how are you?")
        self.assertLess(elapsed, 5.0, f"turn took {elapsed:.1f}s — hung or too slow")

    def test_knowledge_query_handles_tool_call(self):
        elapsed = self._timed(Pipeline(self._session()), "What is the Elden Ring?")
        self.assertLess(elapsed, 10.0, f"turn took {elapsed:.1f}s — possible deadlock")

    def test_multiple_utterances_sequential(self):
        elapsed = self._timed(Pipeline(self._session()), "Hi.", "What is the Elden Ring?", "Thanks.")
        self.assertLess(elapsed, 20.0, f"3 utterances took {elapsed:.1f}s")

    def test_session_is_free_after_a_full_reply(self):
        session = self._session(with_tool=False)
        list(session("Hi."))
        self.assertFalse(session._lock.locked())

    def test_session_is_free_after_a_reply_is_cut_short(self):
        session = self._session(with_tool=False)
        stream = session("Hi.")
        next(stream, None)
        stream.close()
        self.assertFalse(session._lock.locked())
        self.assertFalse(self.llm._lock.locked(), "the model must be released for the next turn")

    def test_interrupt_stops_the_reply(self):
        session = self._session(with_tool=False)
        pipeline = Pipeline(session)
        pipeline.events.on("tts:start", lambda sentence: pipeline.interrupt())

        elapsed = self._timed(pipeline, "Tell me a very long story about a dragon.")

        self.assertLess(elapsed, 5.0, f"turn took {elapsed:.1f}s despite the interrupt")
        self.assertEqual(len(pipeline._response_parts), 1, "nothing is spoken after the interruption")


if __name__ == "__main__":
    unittest.main()
