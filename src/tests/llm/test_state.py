import json
import os
import tempfile
import unittest

from voxpipe.llm.state import NO_TURN, SessionState
from voxpipe.llm.tools import ToolCall, ToolChoice


class TestSessionState(unittest.TestCase):
    def test_tools_permission_view(self):
        state = SessionState()
        self.assertIsNone(state.tools.get_permission("delete_file"))

        state.tools.set_permission("delete_file", True)
        self.assertTrue(state.tools.get_permission("delete_file"))

        state.tools.revoke_permission("delete_file")
        self.assertIsNone(state.tools.get_permission("delete_file"))

    def test_pending_choices(self):
        state = SessionState()
        call = ToolCall(name="book", arguments={"x": 1})
        choice = ToolChoice(result={"slot": ["1", "2"]})

        pending = state.tools.add_pending(call, choice, turn=3)
        self.assertTrue(state.tools.has_pending())
        self.assertIs(state.tools.get_pending(choice.uid), pending)
        self.assertEqual(pending.turn, 3)

        self.assertIs(state.tools.remove_pending(choice.uid), pending)
        self.assertIsNone(state.tools.remove_pending(choice.uid))
        self.assertFalse(state.tools.has_pending())

    def test_user_view(self):
        state = SessionState()
        state.user.set("theme", "dark")
        self.assertEqual(state.user.get("theme"), "dark")
        self.assertEqual(state.user.get("missing", "default"), "default")
        self.assertEqual(state.user.pop("theme"), "dark")
        self.assertIsNone(state.user.get("theme"))

    def test_model_view(self):
        state = SessionState()
        state.model.runtime["handle"] = object()
        self.assertIn("handle", state.model.runtime)

    def test_clear(self):
        state = SessionState()
        state.tools.set_permission("t", True)
        state.tools.add_pending(ToolCall(name="t"), ToolChoice(result={"a": [1]}))
        state.user.set("k", "v")
        state.model.runtime["k"] = "v"

        state.clear()

        self.assertIsNone(state.tools.get_permission("t"))
        self.assertFalse(state.tools.has_pending())
        self.assertIsNone(state.user.get("k"))
        self.assertEqual(state.model.runtime, {})

    def test_save_and_load(self):
        call = ToolCall(name="book", arguments={"x": 1})
        choice = ToolChoice(result={"slot": ["1", "2"]}, speech="Which slot?")
        with tempfile.TemporaryDirectory() as tmp:
            state = SessionState()
            state.tools.set_permission("run_script", True)
            state.tools.add_pending(call, choice, turn=7)
            state.user.set("user_id", "usr_100")
            state.model.runtime["live"] = object()

            state.save(tmp)
            self.assertEqual(os.listdir(tmp), ["state.json"])
            with open(os.path.join(tmp, "state.json"), encoding="utf-8") as f:
                self.assertEqual(set(json.load(f)), {"tools", "user"}, "provider objects are never saved")

            restored = SessionState()
            restored.user.set("stale", True)
            restored.model.runtime["stale"] = object()
            restored.load(tmp)

        self.assertTrue(restored.tools.get_permission("run_script"))
        self.assertEqual(restored.user.get("user_id"), "usr_100")
        self.assertIsNone(restored.user.get("stale"), "load replaces, not merges")
        self.assertEqual(restored.model.runtime, {})

        pending = restored.tools.get_pending(choice.uid)
        self.assertEqual((pending.call.name, pending.call.arguments, pending.call.id), ("book", {"x": 1}, call.id))
        self.assertEqual(pending.choice.speech, "Which slot?")
        self.assertEqual(pending.choice.choices_dict, choice.choices_dict)
        self.assertEqual(pending.turn, NO_TURN, "a restored choice belongs to no live turn")

    def test_load_from_empty_directory(self):
        with tempfile.TemporaryDirectory() as empty:
            state = SessionState()
            state.tools.set_permission("t", True)
            state.load(empty)
        self.assertIsNone(state.tools.get_permission("t"))


class TestSavingAwkwardValues(unittest.TestCase):
    def test_a_pending_choice_is_saved_as_the_model_saw_it(self):
        import datetime

        choice = ToolChoice(result={"when": [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)]})
        state = SessionState()
        state.tools.add_pending(ToolCall(name="book"), choice)
        with tempfile.TemporaryDirectory() as tmp:
            state.save(tmp)
            restored = SessionState()
            restored.load(tmp)
        self.assertEqual(restored.tools.get_pending(choice.uid).choice.raw_result["when"], ["2026-01-01", "2026-01-02"])

    def test_a_failed_save_keeps_the_previous_file(self):
        state = SessionState()
        state.user.set("note", "kept")
        with tempfile.TemporaryDirectory() as tmp:
            state.save(tmp)
            state.user.set("bad", object())
            with self.assertRaises(TypeError):
                state.save(tmp)
            with open(os.path.join(tmp, "state.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["user"], {"note": "kept"})
            self.assertEqual(os.listdir(tmp), ["state.json"], "no temporary file is left behind")


if __name__ == "__main__":
    unittest.main()
