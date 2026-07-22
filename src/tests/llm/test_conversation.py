import os
import unittest
from unittest.mock import MagicMock


class TestConversation(unittest.TestCase):
    """Conversation state machine, serialization, and edge cases."""

    def test_messages_includes_system(self):
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        conv.set_system_message("You are a bot.")
        msgs = conv.messages
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "You are a bot.")

    def test_messages_ordering(self):
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        conv.set_system_message("sys")
        conv.add_user_message("user1")
        conv.add_assistant_message("assistant1")
        conv.add_tool_message("tool1")
        roles = [m["role"] for m in conv.messages]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])

    def test_system_empty_by_default(self):
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        self.assertEqual(conv.messages, [])

    def test_cutoff_idx_excludes_old_messages(self):
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        conv.add_user_message("a")
        conv.add_assistant_message("b")
        conv.add_user_message("c")
        conv.cutoff_idx = 2
        roles = [m["role"] for m in conv.messages]
        contents = [m["content"] for m in conv.messages]
        self.assertEqual(roles, ["user"])
        self.assertEqual(contents, ["c"])

    def test_cutoff_idx_zero(self):
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        conv.add_user_message("a")
        conv.cutoff_idx = 0
        self.assertEqual(len(conv.messages), 1)

    def test_to_dict_round_trip(self):
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        conv.set_system_message("sys")
        conv.add_user_message("hello")
        conv.add_assistant_message("hi")
        data = conv.to_dict()
        conv2 = Conversation.from_dict(data)
        self.assertEqual(conv2._system, "sys")
        self.assertEqual(len(conv2._messages), 2)
        self.assertEqual(conv2.messages[1]["content"], "hello")

    def test_save_load_json(self):
        import tempfile, json
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        conv.set_system_message("sys")
        conv.add_user_message("hello")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
            conv.save(path)
        conv2 = Conversation.load(path)
        os.unlink(path)
        self.assertEqual(conv2._system, "sys")
        self.assertEqual(len(conv2._messages), 1)

    def test_clear(self):
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        conv.add_user_message("a")
        conv.add_assistant_message("b")
        conv.clear()
        self.assertEqual(conv.messages, [])

    def test_visible_count(self):
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        conv.add_user_message("a")
        conv.add_assistant_message("b")
        conv.add_user_message("c")
        self.assertEqual(conv.visible_count(), 3)
        conv.cutoff_idx = 2
        self.assertEqual(conv.visible_count(), 1)

    def test_trim_oldest(self):
        from voxpipe.llm.conversation import Conversation
        from unittest.mock import MagicMock
        llm = MagicMock()
        llm.count_tokens.return_value = 10
        conv = Conversation()
        conv.set_system_message("sys")
        conv.add_user_message("a")
        conv.add_assistant_message("b")
        conv.add_user_message("c")
        conv.add_assistant_message("d")
        # Trim 2 oldest visible messages — cutoff advances, messages stay
        trimmed = conv.trim_oldest(2, 20, llm)
        self.assertEqual(trimmed, 28)  # (10+4) * 2
        self.assertEqual(len(conv._messages), 4, "messages stay in list")
        self.assertEqual(conv.cutoff_idx, 2, "cutoff advances past trimmed")
        # Visible count reflects the new cutoff
        self.assertEqual(conv.visible_count(), 2)

    def test_trim_oldest_clamps_to_visible(self):
        from voxpipe.llm.conversation import Conversation
        from unittest.mock import MagicMock
        llm = MagicMock()
        llm.count_tokens.return_value = 5
        conv = Conversation()
        conv.add_user_message("a")
        trimmed = conv.trim_oldest(100, 20, llm)
        self.assertEqual(trimmed, 9)  # (5+4) * 1
        self.assertEqual(len(conv._messages), 1, "message stays in list")
        self.assertEqual(conv.cutoff_idx, 1, "cutoff advances by 1")

    def test_set_token_count(self):
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        conv.add_user_message("hello")
        conv.set_token_count(0, 42)
        self.assertEqual(conv.get_token_count(0), 42)

    def test_get_message_content(self):
        from voxpipe.llm.conversation import Conversation
        conv = Conversation()
        conv.add_user_message("hello world")
        self.assertEqual(conv.get_message_content(0), "hello world")

    def test_tools_setter_from_iterable(self):
        from voxpipe.llm.conversation import Conversation
        from voxpipe.llm.tools import Tool
        t = Tool(name="t1", description="")
        conv = Conversation()
        conv.tools = [t]
        self.assertIn("t1", conv.tools)

    def test_tools_setter_from_dict(self):
        from voxpipe.llm.conversation import Conversation
        from voxpipe.llm.tools import Tool
        t = Tool(name="t1", description="")
        conv = Conversation()
        conv.tools = {"t1": t}
        self.assertIn("t1", conv.tools)

    def test_message_list_slice_assignment(self):
        from voxpipe.llm.conversation import MessageList, Message
        msg_list = MessageList([
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
        ])
        new_msgs = [
            {"role": "user", "content": "updated 1"},
            Message(role=Message.Role.assistant, content="updated 2"),
        ]
        msg_list[0:2] = new_msgs
        self.assertEqual(msg_list[0]["content"], "updated 1")
        self.assertEqual(msg_list[1]["content"], "updated 2")

