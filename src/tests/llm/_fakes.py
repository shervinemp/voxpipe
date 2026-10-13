"""Test doubles shared by the LLM-layer tests."""

from types import SimpleNamespace

from voxpipe.llm.model import LLM


class ScriptedLLM(LLM):
    """Replays one scripted list of stream items per call; the last script repeats.

    A script may also be a callable taking the conversation and returning the
    items, for replies that depend on what happened earlier in the turn.
    Every call is recorded in ``calls``.
    """

    def __init__(self, *scripts, decoder=None):
        self.scripts = list(scripts)
        self.calls = []
        if decoder is not None:
            self.decoder = decoder

    def _infer(self, conversation, *, session_state, tool_choice="auto", **kwargs):
        script = self.scripts[min(len(self.calls), len(self.scripts) - 1)] if self.scripts else []
        self.calls.append(
            SimpleNamespace(
                tool_choice=tool_choice,
                messages=conversation.messages,
                openai_messages=conversation.to_openai_messages(),
                state=session_state,
                kwargs=kwargs,
            )
        )
        if callable(script):
            script = script(conversation)
        yield from script
