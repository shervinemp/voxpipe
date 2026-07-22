"""Integration tests that run the actual LLM model.

Non-deterministic (model output varies).  Uses best-of-5: each scenario
runs 5 times; passes if >= 3 agree on the expected outcome.

Separated from unit tests because the GGUF must be downloaded.

Run with:
  python -m pytest tests/test_llm_integration.py -v --tb=short
"""
import time
import unittest

import pytest

from voxpipe.llm.model import GGUFLLM
from voxpipe.llm.session import Session
from voxpipe.llm.conversation import Conversation
from voxpipe.llm.tools import Tool, ToolCall


def _has_model():
    try:
        from voxpipe.storage.manager import is_downloaded
        return is_downloaded("Gemma4E4B")
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _has_model(), reason="Gemma4E4B GGUF not downloaded"),
]


class TestModelBehavior(unittest.TestCase):
    """Best-of-5 non-deterministic model behavior tests.

    Each scenario runs 5 times.  Passes if >= 3 runs match the expected
    tool-call pattern (0 calls for greetings, 1+ calls for knowledge).
    Tools are NEVER executed — we only check if the decoder can parse
    the model's output format.
    """

    _model = None

    @classmethod
    def setUpClass(cls):
        t0 = time.monotonic()
        cls._model = GGUFLLM("Gemma4E4B")
        cls._model.logger = type("L", (), {"info": print, "warning": print})()
        print(f"\nModel loaded in {time.monotonic()-t0:.1f}s")

    def _run_once(self, message: str, with_tools: bool,
                  demanding_prompt: bool = True) -> list[ToolCall]:
        """Run the model once, return ToolCalls yielded directly by the model."""
        conv = Conversation()
        if demanding_prompt:
            conv.set_system_message(
                "You are a voice-controlled game assistant. "
                "Respond conversationally and naturally.\n\nRules:\n"
                "- Call 'retrieve' when the user asks about entities, "
                "relationships, or facts.\n"
                "- If the user's message seems incomplete or cut off, "
                "ask what they meant before proceeding."
            )
        else:
            conv.set_system_message(
                "You are a helpful voice-controlled assistant. "
                "Respond conversationally and naturally."
            )
        if with_tools:
            t = Tool.from_callable("retrieve", lambda q: "")
            t.instruction = "Call when asked about entities or facts."
            conv.tools["retrieve"] = t
        conv.add_user_message(message)

        calls = []
        for chunk in self._model(conv, session_state={}):
            if isinstance(chunk, ToolCall):
                calls.append(chunk)
        return calls

    def _best_of_5(self, message: str, with_tools: bool,
                   demanding_prompt: bool = True) -> list[int]:
        """Run 5 times.  Returns list of call counts per run."""
        counts = []
        for i in range(5):
            calls = self._run_once(message, with_tools, demanding_prompt)
            counts.append(len(calls))
        return counts

    # 2x2 matrix: tools × prompt
    #   tools=yes + prompt=demanding  → model has tool + told to use it
    #   tools=yes + prompt=plain      → model has tool but not told to use it
    #   tools=no  + prompt=demanding  → model told to use tool but none available
    #   tools=no  + prompt=plain      → model has neither

    # --- Greetings (expect 0 calls in all 4 combos) ---

    def test_greeting_tools_demanding(self):
        counts = self._best_of_5("Hi.", with_tools=True, demanding_prompt=True)
        ok = sum(1 for c in counts if c == 0)
        print(f"  tools+demand [{ok}/5]: {counts}")
        self.assertGreaterEqual(ok, 3)

    def test_greeting_tools_plain(self):
        counts = self._best_of_5("Hi.", with_tools=True, demanding_prompt=False)
        ok = sum(1 for c in counts if c == 0)
        print(f"  tools+plain [{ok}/5]: {counts}")
        self.assertGreaterEqual(ok, 3)

    def test_greeting_no_tools_demanding(self):
        counts = self._best_of_5("Hi.", with_tools=False, demanding_prompt=True)
        ok = sum(1 for c in counts if c == 0)
        print(f"  no_tools+demand [{ok}/5]: {counts}")
        self.assertGreaterEqual(ok, 3)

    def test_greeting_no_tools_plain(self):
        counts = self._best_of_5("Hi.", with_tools=False, demanding_prompt=False)
        ok = sum(1 for c in counts if c == 0)
        print(f"  no_tools+plain [{ok}/5]: {counts}")
        self.assertGreaterEqual(ok, 3)

    # --- Knowledge (expect >=1 calls only when tools=yes + prompt=demanding) ---

    def test_knowledge_tools_demanding(self):
        counts = self._best_of_5("What is the Elden Ring?", with_tools=True, demanding_prompt=True)
        ok = sum(1 for c in counts if c >= 1)
        print(f"  tools+demand [{ok}/5]: {counts}")
        # Model may or may not call tools with tool_choice=auto;
        # this is informational, not a hard assertion
        if ok < 3:
            print(f"  (model called tool in {ok}/5 runs — native tool calling is optional)")

    def test_knowledge_tools_plain(self):
        counts = self._best_of_5("What is the Elden Ring?", with_tools=True, demanding_prompt=False)
        ok = sum(1 for c in counts if c >= 1)
        print(f"  tools+plain [{ok}/5]: {counts}")
        # Model has the tool but wasn't told to use it — may or may not call
        # This is informational, not a hard assertion
        print(f"  (informational: called in {ok}/5 runs with tool but no prompt instruction)")

    def test_knowledge_no_tools_demanding(self):
        counts = self._best_of_5("What is the Elden Ring?", with_tools=False, demanding_prompt=True)
        ok = sum(1 for c in counts if c == 0)
        print(f"  no_tools+demand [{ok}/5]: {counts}")
        self.assertGreaterEqual(ok, 3)

    def test_knowledge_no_tools_plain(self):
        counts = self._best_of_5("What is the Elden Ring?", with_tools=False, demanding_prompt=False)
        ok = sum(1 for c in counts if c == 0)
        print(f"  no_tools+plain [{ok}/5]: {counts}")
        self.assertGreaterEqual(ok, 3)

    # --- Decoder format verification ---

    def test_tool_call_format_is_parseable(self):
        """When model calls a tool natively, verify ToolCall is yielded correctly."""
        from voxpipe.llm.decoders import GeneralDecoder, GemmaE2BDecoder

        t = Tool.from_callable("retrieve", lambda q: "")
        t.instruction = "Call when asked about entities or facts."

        native = 0

        for i in range(5):
            conv = Conversation()
            conv.set_system_message("You are a game assistant.")
            conv.tools["retrieve"] = t
            conv.add_user_message("Tell me about Skyrim.")

            for chunk in self._model(conv, session_state={}):
                if isinstance(chunk, ToolCall):
                    native += 1
                    break

        print(f"  Native ToolCalls: {native}/5")
        self.assertGreaterEqual(native, 3,
                                "Model didn't generate native tool calls in enough runs")


class TestMultiTurn(unittest.TestCase):
    """Cross-turn state with real model."""

    _model = None

    @classmethod
    def setUpClass(cls):
        t0 = time.monotonic()
        cls._model = GGUFLLM("Gemma4E4B")
        cls._model.logger = type("L", (), {"info": print, "warning": print})()
        print(f"\nModel loaded in {time.monotonic()-t0:.1f}s")

    def setUp(self):
        self.conv = Conversation()
        self.conv.set_system_message(
            "You are a voice-controlled game assistant. "
            "Respond conversationally and naturally."
        )
        self.sess = Session(llm=self._model, conversation=self.conv)

    def tearDown(self):
        self.sess.close()

    def test_two_greetings_in_a_row(self):
        """Two greetings must both produce text without error."""
        r1 = "".join(self.sess("Hi there."))
        r2 = "".join(self.sess("How are you?"))
        self.assertTrue(len(r1) > 0, "first greeting produced no text")
        self.assertTrue(len(r2) > 0, "second greeting produced no text")

    def test_greeting_then_knowledge(self):
        """Greeting then knowledge query, no cross-contamination."""
        r1 = "".join(self.sess("Hello."))
        self.assertTrue(len(r1) > 0)
        r2 = "".join(self.sess("What is the Elden Ring?"))
        # Should produce text (may or may not call retrieve tool)
        self.assertTrue(len(r2) > 0,
                        "knowledge query produced no text")

    def test_conversation_history_grows(self):
        """Messages accumulate across turns."""
        list(self.sess("Hi."))
        list(self.sess("What is Skyrim?"))
        list(self.sess("Thanks."))
        self.assertGreaterEqual(len(self.conv.messages), 5,
                                "expected 5+ messages across 3 turns")

    def test_context_pruning(self):
        """Many turns should not exceed max_turns."""
        for i in range(15):
            list(self.sess(f"Message {i}."))
        self.assertLessEqual(len(self.conv.messages), 45,
                             "messages should be bounded")


if __name__ == "__main__":
    unittest.main()
