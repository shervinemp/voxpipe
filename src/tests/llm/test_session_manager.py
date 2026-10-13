import json
import os
import shutil
import tempfile
import unittest
import zipfile

from voxpipe.core.exceptions import VoxpipeError
from voxpipe.llm.session import Session
from voxpipe.llm.session_manager import SessionManager
from voxpipe.llm.tools import ToolCall, ToolChoice

from ._fakes import ScriptedLLM


class TestSessionManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.manager = SessionManager(root_dir=os.path.join(self.tmp_dir, "root"))
        self.llm = ScriptedLLM([])

    def _session(self, **kwargs):
        session = Session(llm=self.llm, **kwargs)
        self.addCleanup(session.close)
        return session

    def _load(self, session_id, **kwargs):
        session = self.manager.load_session(session_id, llm=self.llm, **kwargs)
        self.addCleanup(session.close)
        return session

    def test_save_and_load_session(self):
        session = self._session()
        session.conversation.add_user_message("Hello from test!")
        session.state.tools.set_permission("search", True)

        path = self.manager.save_session(session, "sess_001", meta={"user": "alice"})
        for name in ("manifest.json", "conversation.json", "state.json"):
            self.assertTrue(os.path.exists(os.path.join(path, name)), name)

        loaded = self._load("sess_001", max_tool_iterations=4)
        self.assertEqual(loaded.conversation.messages[-1]["content"], "Hello from test!")
        self.assertTrue(loaded.state.tools.get_permission("search"))
        self.assertEqual(loaded.max_tool_iterations, 4)

    def test_manifest(self):
        session = self._session()
        self.manager.save_session(session, "sess", meta={"user": "alice"})
        first = self.manager.list_sessions()[0]
        self.manager.save_session(session, "sess")
        second = self.manager.list_sessions()[0]

        self.assertEqual(first["session_id"], "sess")
        self.assertEqual(first["llm_provider"], "ScriptedLLM")
        self.assertEqual(first["custom"], {"user": "alice"})
        self.assertEqual(second["created_at"], first["created_at"], "re-saving keeps the creation time")
        self.assertGreaterEqual(second["saved_at"], first["saved_at"])

    def test_list_sessions_flags_broken_entries(self):
        self.manager.save_session(self._session(), "good")
        os.makedirs(os.path.join(self.manager.sessions_dir, "no_manifest"))
        os.makedirs(os.path.join(self.manager.sessions_dir, "corrupt"))
        with open(os.path.join(self.manager.sessions_dir, "corrupt", "manifest.json"), "w") as f:
            f.write("{not json")

        by_id = {s["session_id"]: s for s in self.manager.list_sessions()}
        self.assertEqual(set(by_id), {"good", "no_manifest", "corrupt"})
        self.assertIs(by_id["no_manifest"]["manifest"], False)
        self.assertIs(by_id["corrupt"]["corrupted"], True)

    def test_delete_session(self):
        self.manager.save_session(self._session(), "sess_to_delete")
        self.assertTrue(self.manager.delete_session("sess_to_delete"))
        self.assertFalse(self.manager.delete_session("sess_to_delete"))

    def test_load_missing_session(self):
        with self.assertRaises(FileNotFoundError):
            self.manager.load_session("nope", llm=self.llm)

    def test_session_ids_cannot_escape_the_sessions_directory(self):
        victim = os.path.join(self.tmp_dir, "victim")
        os.makedirs(victim)
        hostile = [
            os.path.join("..", "..", "victim"),
            "../../victim",
            "..",
            ".",
            "",
            os.path.abspath(victim),
            "a/b",
            "a\\b",
            "sess..x",
            None,
        ]
        operations = [
            self.manager.delete_session,
            lambda sid: self.manager.load_session(sid, llm=self.llm),
            lambda sid: self.manager.save_session(self._session(), sid),
            lambda sid: self.manager.export_zip(sid, os.path.join(self.tmp_dir, "out.zip")),
        ]
        for session_id in hostile:
            for operation in operations:
                with self.subTest(session_id=session_id), self.assertRaises(VoxpipeError):
                    operation(session_id)
        self.assertTrue(os.path.isdir(victim))

    def test_export_and_import_zip(self):
        session = self._session()
        session.conversation.add_user_message("Zip test message")
        session.state.user.set("zip_var", "zip_val")
        self.manager.save_session(session, "sess_zip")

        zip_dest = os.path.join(self.tmp_dir, "exports", "sess_zip.zip")
        self.assertEqual(self.manager.export_zip("sess_zip", zip_dest), zip_dest)
        self.assertEqual(self.manager.import_zip(zip_dest, session_id="sess_imported"), "sess_imported")

        loaded = self._load("sess_imported")
        self.assertEqual(loaded.conversation.messages[-1]["content"], "Zip test message")
        self.assertEqual(loaded.state.user.get("zip_var"), "zip_val")

    def test_import_defaults_to_the_archive_name_and_replaces_existing(self):
        stale = self._session()
        stale.conversation.add_user_message("stale")
        self.manager.save_session(stale, "trip")

        fresh = self._session()
        fresh.conversation.add_user_message("fresh")
        self.manager.save_session(fresh, "source")
        archive = self.manager.export_zip("source", os.path.join(self.tmp_dir, "trip.zip"))

        self.assertEqual(self.manager.import_zip(archive), "trip")
        self.assertEqual(self._load("trip").conversation.messages[-1]["content"], "fresh")

    def test_import_drops_permissions_and_pending_calls(self):
        """An archive from elsewhere must not arrive with consent already given."""
        session = self._session()
        session.state.tools.set_permission("delete_files", True)
        session.state.tools.add_pending(ToolCall(name="delete_files"), ToolChoice(result={"allow": [True, False]}))
        session.state.user.set("name", "mallory")
        self.manager.save_session(session, "source")
        archive = self.manager.export_zip("source", os.path.join(self.tmp_dir, "a.zip"))

        self.manager.import_zip(archive, session_id="imported")
        loaded = self._load("imported")

        self.assertIsNone(loaded.state.tools.get_permission("delete_files"))
        self.assertFalse(loaded.state.tools.has_pending())
        self.assertEqual(loaded.state.user.get("name"), "mallory")

    def test_import_rejects_unexpected_members(self):
        archive = os.path.join(self.tmp_dir, "evil.zip")
        for member in ("../outside.txt", "nested/state.json", "payload.py"):
            with self.subTest(member=member):
                with zipfile.ZipFile(archive, "w") as zf:
                    zf.writestr("conversation.json", json.dumps({"messages": []}))
                    zf.writestr(member, "x")
                with self.assertRaises(VoxpipeError):
                    self.manager.import_zip(archive, session_id="evil")
                self.assertFalse(os.path.exists(os.path.join(self.manager.sessions_dir, "evil")))
        self.assertFalse(os.path.exists(os.path.join(self.manager.root_dir, "outside.txt")))

    def test_import_missing_archive(self):
        with self.assertRaises(FileNotFoundError):
            self.manager.import_zip(os.path.join(self.tmp_dir, "missing.zip"))


class TestImportSafety(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.manager = SessionManager(root_dir=os.path.join(self.tmp_dir, "root"))
        self.llm = ScriptedLLM([])

    def _saved(self, session_id, text):
        session = Session(llm=self.llm)
        self.addCleanup(session.close)
        session.conversation.add_user_message(text)
        self.manager.save_session(session, session_id)

    def _history(self, session_id):
        session = self.manager.load_session(session_id, llm=self.llm)
        self.addCleanup(session.close)
        return [m.content for m in session.conversation.history]

    def test_a_damaged_archive_does_not_cost_the_session_it_would_replace(self):
        self._saved("trip", "precious history")
        archive = os.path.join(self.tmp_dir, "trip.zip")
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("conversation.json", json.dumps({"messages": []}))
            zf.writestr("state.json", "{not json")

        with self.assertRaisesRegex(VoxpipeError, "damaged"):
            self.manager.import_zip(archive)

        self.assertEqual(self._history("trip"), ["precious history"])
        self.assertEqual(os.listdir(self.manager.sessions_dir), ["trip"], "nothing half-unpacked is left behind")

    def test_an_imported_session_is_listed_under_its_new_id(self):
        self._saved("source", "hello")
        archive = self.manager.export_zip("source", os.path.join(self.tmp_dir, "source.zip"))
        self.manager.import_zip(archive, session_id="backup")
        self.assertEqual([s["session_id"] for s in self.manager.list_sessions()], ["backup", "source"])

    def test_ids_the_file_system_would_alias_or_refuse_are_rejected(self):
        session = Session(llm=self.llm)
        self.addCleanup(session.close)
        for session_id in ("notes.", "nul", "CON", "aux.txt", "com1"):
            with self.subTest(session_id=session_id), self.assertRaises(VoxpipeError):
                self.manager.save_session(session, session_id)
        self.assertEqual(os.listdir(self.manager.sessions_dir), [])

    def test_stray_entries_are_not_listed_as_sessions(self):
        self._saved("real", "hello")
        os.makedirs(os.path.join(self.manager.sessions_dir, ".import-leftover"))
        self.assertEqual([s["session_id"] for s in self.manager.list_sessions()], ["real"])


if __name__ == "__main__":
    unittest.main()
