import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from voxpipe.llm.session import Session
from voxpipe.llm.session_manager import SessionManager


class TestSessionManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.manager = SessionManager(root_dir=self.tmp_dir)
        self.mock_llm = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_load_session(self):
        session = Session(llm=self.mock_llm)
        session.conversation.add_user_message("Hello from test!")
        session.state.tools.set_permission("search", True)

        path = self.manager.save_session(session, "sess_001", meta={"user": "alice"})
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.exists(os.path.join(path, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(path, "conversation.json")))
        self.assertTrue(os.path.exists(os.path.join(path, "state.json")))

        loaded = self.manager.load_session("sess_001", llm=self.mock_llm)
        self.assertEqual(len(loaded.conversation._messages), 1)
        self.assertEqual(loaded.conversation.messages[-1]["content"], "Hello from test!")
        self.assertTrue(loaded.state.tools.get_permission("search"))

    def test_list_sessions(self):
        session1 = Session(llm=self.mock_llm)
        session2 = Session(llm=self.mock_llm)

        self.manager.save_session(session1, "sess_A")
        self.manager.save_session(session2, "sess_B")

        sessions = self.manager.list_sessions()
        session_ids = [s["session_id"] for s in sessions]
        self.assertIn("sess_A", session_ids)
        self.assertIn("sess_B", session_ids)

    def test_delete_session(self):
        session = Session(llm=self.mock_llm)
        self.manager.save_session(session, "sess_to_delete")

        self.assertTrue(self.manager.delete_session("sess_to_delete"))
        self.assertFalse(self.manager.delete_session("sess_to_delete"))

    def test_export_zip(self):
        session = Session(llm=self.mock_llm)
        session.conversation.add_user_message("Zip test message")
        self.manager.save_session(session, "sess_zip")

        zip_dest = os.path.join(self.tmp_dir, "exports", "sess_zip.zip")
        out_path = self.manager.export_zip("sess_zip", zip_dest)

        self.assertTrue(os.path.exists(out_path))
        self.assertTrue(out_path.endswith(".zip"))


if __name__ == "__main__":
    unittest.main()
