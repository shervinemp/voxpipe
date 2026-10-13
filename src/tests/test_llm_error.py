import unittest

from voxpipe.core.exceptions import LLMError
from voxpipe.llm.conversation import Conversation
from voxpipe.llm.model import LLM


class ErrorLLM(LLM):
    def _infer(self, conversation, *, session_state, **kwargs):
        yield "partial "
        raise RuntimeError("Something went wrong!")


class RefusingLLM(LLM):
    def _infer(self, conversation, *, session_state, **kwargs):
        raise LLMError("Unsupported tool_choice value.")
        yield  # unreachable, but makes this a generator like every other _infer


class TestLLMErrorHandling(unittest.TestCase):
    def test_provider_failure_raises_llm_error(self):
        stream = ErrorLLM()(Conversation())
        self.assertEqual(next(stream), "partial ")
        with self.assertRaises(LLMError) as raised:
            next(stream)
        self.assertIn("Something went wrong!", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_llm_error_passes_through_unwrapped(self):
        with self.assertRaisesRegex(LLMError, "^Unsupported tool_choice value.$"):
            list(RefusingLLM()(Conversation()))

    def test_raw_stream_is_closed_when_the_consumer_stops_early(self):
        closed = []

        class Tracked(LLM):
            def _infer(self, conversation, *, session_state, **kwargs):
                try:
                    yield "a"
                    yield "b"
                finally:
                    closed.append(True)

        stream = Tracked()(Conversation())
        next(stream)
        stream.close()
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
