import unittest
from unittest.mock import MagicMock, PropertyMock
from voxpipe.llm.conversation import Conversation
from voxpipe.llm.context import ContextHandler, DropOldestStrategy
from voxpipe.storage import MemoryStore, RAMStorage


class TestContextHandler(unittest.TestCase):
    def _llm(self):
        m = MagicMock()
        m.count_tokens.return_value = 10
        type(m).n_ctx = PropertyMock(return_value=4096)
        type(m).max_tokens = PropertyMock(return_value=512)
        return m

    def test_context_handler_trim_and_auto_archive(self):
        mem = MemoryStore(backend=RAMStorage(), bank="session:test")
        handler = ContextHandler(max_turns=2, memory=mem, auto_archive=True)

        conv = Conversation()
        for i in range(10):
            conv.add_user_message(f"message turn {i}")

        llm = self._llm()
        handler.handle(conv, llm, session_id="test")

        # Verify trimming occurred
        self.assertLessEqual(conv.visible_count(), 6)

        # Verify auto-archival into RAM storage
        archived = mem.retrieve("turn 0")
        self.assertGreater(len(archived), 0)


if __name__ == "__main__":
    unittest.main()
