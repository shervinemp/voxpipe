import tempfile
import os
import unittest
from voxpipe.llm.state import SessionState


class TestSessionState(unittest.TestCase):
    def test_tools_permission_view(self):
        state = SessionState()
        self.assertIsNone(state.tools.get_permission("delete_file"))

        state.tools.set_permission("delete_file", True)
        self.assertTrue(state.tools.get_permission("delete_file"))

        state.tools.revoke_permission("delete_file")
        self.assertIsNone(state.tools.get_permission("delete_file"))

    def test_user_view(self):
        state = SessionState()
        state.user.set("theme", "dark")
        self.assertEqual(state.user.get("theme"), "dark")
        self.assertEqual(state.user.get("missing", "default"), "default")

    def test_model_view_and_binary(self):
        state = SessionState()
        state.model.set_kv_cache(b"raw_kv_cache_bytes")
        self.assertEqual(state.model.get_kv_cache(), b"raw_kv_cache_bytes")

    def test_dict_backward_compatibility(self):
        state = SessionState()
        state["custom_key"] = "custom_val"
        self.assertEqual(state["custom_key"], "custom_val")
        self.assertIn("custom_key", state)
        self.assertEqual(state.get("custom_key"), "custom_val")

    def test_save_and_load(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            state = SessionState()
            state.tools.set_permission("run_script", True)
            state.user.set("user_id", "usr_100")
            state.model.set_kv_cache(b"model_binary_bytes")

            state.save(tmp_dir, save_binary=True)

            state2 = SessionState()
            state2.load(tmp_dir)

            self.assertTrue(state2.tools.get_permission("run_script"))
            self.assertEqual(state2.user.get("user_id"), "usr_100")
            self.assertEqual(state2.model.get_kv_cache(), b"model_binary_bytes")
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
