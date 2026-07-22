"""End-to-end pipeline integration tests with real model and TTS.

Verifies:
  - _callback completes within timing bounds (no hang)
  - Lock is released after generator consumption
  - Tool calls during knowledge queries don't deadlock
  - Multiple utterances can be processed sequentially

These are non-deterministic (model output varies) and require the GGUF model.
"""
import time, threading
import unittest
from unittest.mock import MagicMock

import pytest

from voxpipe.llm.conversation import Conversation
from voxpipe.llm.session import Session


def _has_model():
    from voxpipe.storage.manager import ensure_downloaded
    try:
        ensure_downloaded("Gemma4E4B")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_model(), reason="Gemma4E4B GGUF not downloaded")


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests that run real model through pipeline callback."""

    _pipe = None

    @classmethod
    def setUpClass(cls):
        from voxpipe.llm import LLMProviders
        from voxpipe.llm.tools import Tool
        from voxpipe.core.config import config
        import logging
        logging.getLogger("voice_control").setLevel(logging.WARNING)

        cls.llm = LLMProviders.create(
            config.get("llm.backend"), config.get("llm.model")
        )
        cls.tool = Tool.from_callable("retrieve", lambda q: "test result")
        cls.tool.instruction = "Call when asked about entities."
        cls.default_conv = Conversation()
        cls.default_conv.tools["retrieve"] = cls.tool
        cls.default_conv.set_system_message(
            "You are a voice-controlled game assistant. "
            "Respond conversationally and naturally.\n\nRules:\n"
            "- Call 'retrieve' when the user asks about entities.\n"
            "- If the user seems incomplete, ask what they meant."
        )

    def _make_pipe(self, session):
        from voxpipe.pipeline.pipeline import Pipeline
        p = Pipeline.__new__(Pipeline)
        p.logger = MagicMock()
        p.events = MagicMock()
        
        p.memory = None
        p._response_parts = []
        p._llm_busy = False
        p._interrupt_event = MagicMock()
        p._interrupt_event.is_set.return_value = False
        p._interrupted_at = None
        p._match_command = MagicMock(return_value=False)
        p.session = session
        p.tts = None
        return p

    def test_greeting_completes_within_5s(self):
        """A greeting must complete _callback in under 5 seconds (no hang)."""
        conv = Conversation()
        conv.tools["retrieve"] = self.tool
        conv.set_system_message(self.default_conv._system)
        session = Session(llm=self.llm, conversation=conv)
        pipe = self._make_pipe(session)

        t0 = time.monotonic()
        pipe._callback("Hi, how are you?")
        dt = time.monotonic() - t0

        session.close()
        self.assertLess(dt, 5.0, f"_callback took {dt:.1f}s — hung or too slow")

    def test_knowledge_query_handles_tool_call(self):
        """A knowledge query may trigger a tool call; must not hang."""
        conv = Conversation()
        conv.tools["retrieve"] = self.tool
        conv.set_system_message(self.default_conv._system)
        session = Session(llm=self.llm, conversation=conv)
        pipe = self._make_pipe(session)

        t0 = time.monotonic()
        pipe._callback("What is the Elden Ring?")
        dt = time.monotonic() - t0

        session.close()
        self.assertLess(dt, 10.0, f"_callback took {dt:.1f}s — possible deadlock")

    def test_multiple_utterances_sequential(self):
        """Multiple utterances in sequence must not deadlock."""
        conv = Conversation()
        conv.tools["retrieve"] = self.tool
        conv.set_system_message(self.default_conv._system)
        session = Session(llm=self.llm, conversation=conv)
        pipe = self._make_pipe(session)

        t0 = time.monotonic()
        pipe._callback("Hi.")
        pipe._callback("What is the Elden Ring?")
        pipe._callback("Thanks.")
        dt = time.monotonic() - t0

        session.close()
        self.assertLess(dt, 20.0, f"3 utterances took {dt:.1f}s")

    def test_lock_released_after_generator(self):
        """Generator lock must be released after session() call completes."""
        conv = Conversation()
        conv.set_system_message(self.default_conv._system)
        session = Session(llm=self.llm, conversation=conv)

        gen = session("Hi.")
        list(gen)

        lock_held = session._lock.locked()
        self.assertFalse(lock_held, "Lock still held after generator consumed")
        session.close()

    def test_lock_released_on_partial_consumption(self):
        """Lock must be released even if generator is only partially consumed."""
        conv = Conversation()
        conv.set_system_message(self.default_conv._system)
        session = Session(llm=self.llm, conversation=conv)

        gen = session("Hi.")
        try:
            next(gen)
        except StopIteration:
            pass
        finally:
            gen.close()

        lock_held = session._lock.locked()
        self.assertFalse(lock_held, "Lock still held after partial iteration + close")
        session.close()

    def test_interrupt_stops_tts_chain(self):
        """Setting interrupt_event mid-stream must break the callback loop."""
        conv = Conversation()
        conv.set_system_message(self.default_conv._system)
        session = Session(llm=self.llm, conversation=conv)

        pipe = self._make_pipe(session)
        pipe._interrupt_event.is_set.side_effect = [False, True]

        t0 = time.monotonic()
        pipe._callback("Hi, how are you?")
        dt = time.monotonic() - t0

        session.close()
        self.assertLess(dt, 5.0, f"_callback took {dt:.1f}s despite interrupt")
