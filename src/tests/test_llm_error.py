import unittest
from unittest.mock import MagicMock
from voxpipe.llm.model import LLM
from voxpipe.llm.conversation import Conversation


class ErrorLLM(LLM):
    def __init__(self):
        super().__init__()
        self.logger = MagicMock()

    def _infer(self, conversation, *, session_state, **kwargs):
        raise RuntimeError("Something went wrong!")


class TestLLMErrorHandling(unittest.TestCase):
    def test_error_handling(self):
        llm = ErrorLLM()
        conv = Conversation()

        # We need to collect the generator output
        result = list(llm(conv))

        # Check that we got the error message
        self.assertIn("Sorry, I encountered an error", "".join(result))


if __name__ == "__main__":
    unittest.main()
