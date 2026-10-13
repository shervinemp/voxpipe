import os
import tempfile
import unittest

from voxpipe.llm.conversation import Conversation, Message
from voxpipe.llm.tools import Tool, ToolCall


def exchange(conv, question, answer):
    conv.add_user_message(question)
    conv.add_assistant_message(answer)


class TestConversation(unittest.TestCase):
    """Building, accessors and persistence."""

    def test_messages_includes_system(self):
        conv = Conversation()
        conv.set_system_message("You are a bot.")
        self.assertEqual(conv.messages, [{"role": "system", "content": "You are a bot."}])
        self.assertEqual(conv.system_message, "You are a bot.")

    def test_messages_ordering(self):
        conv = Conversation()
        conv.set_system_message("sys")
        conv.add_user_message("user1")
        conv.add_assistant_message("assistant1")
        conv.add_tool_message("tool1")
        self.assertEqual([m["role"] for m in conv.messages], ["system", "user", "assistant", "tool"])

    def test_empty_by_default(self):
        conv = Conversation()
        self.assertEqual(conv.messages, [])
        self.assertIsNone(conv.last_message)

    def test_add_returns_the_message(self):
        conv = Conversation()
        message = conv.add_user_message("hello")
        self.assertIs(conv.last_message, message)
        self.assertEqual(message.role, Message.Role.user)

    def test_clear_resets_messages_and_window(self):
        conv = Conversation()
        conv.set_system_message("sys")
        for i in range(6):
            conv.add_user_message(f"m{i}")
        conv.cutoff_idx = 4
        conv.clear()
        conv.add_user_message("fresh question")
        self.assertEqual(conv.visible_count(), 1)
        self.assertEqual(conv.messages[-1], {"role": "user", "content": "fresh question"})
        self.assertEqual(conv.system_message, "sys")

    def test_discard_removes_by_identity_and_keeps_the_window(self):
        conv = Conversation()
        first = conv.add_user_message("same text")
        conv.add_assistant_message("reply")
        second = conv.add_user_message("same text")
        conv.cutoff_idx = 2
        conv.discard([first])
        self.assertEqual([m.content for m in conv.all_messages], ["reply", "same text"])
        self.assertEqual(conv.visible, [second])

    def test_tools_setter_from_iterable(self):
        conv = Conversation()
        conv.tools = [Tool(name="t1", description="")]
        self.assertIn("t1", conv.tools)

    def test_tools_setter_from_dict_copies(self):
        source = {"t1": Tool(name="t1", description="")}
        conv = Conversation()
        conv.tools = source
        conv.tools["t2"] = Tool(name="t2", description="")
        self.assertNotIn("t2", source)

    def test_to_dict_round_trip(self):
        conv = Conversation()
        conv.set_system_message("sys")
        conv.add_user_message("hello", context="(Earlier: hi)")
        call = ToolCall(name="lookup", arguments={"q": "x"})
        conv.add_assistant_message("checking", tool_calls=[call])
        conv.add_tool_message('{"a": 1}', tool_call=call)
        conv.add_tool_message("a note")
        conv.add_assistant_message("done").archived = True
        conv.cutoff_idx = 1

        restored = Conversation.from_dict(conv.to_dict())

        self.assertEqual(restored.to_dict(), conv.to_dict())
        self.assertEqual(restored.all_messages, conv.all_messages)
        self.assertEqual(restored.all_messages[1].tool_calls[0].id, call.id)
        self.assertEqual(restored.cutoff_idx, 1)

    def test_from_dict_accepts_plain_role_content_messages(self):
        conv = Conversation.from_dict({"messages": [{"role": "user", "content": "hi"}], "_meta": {"x": {}}})
        self.assertEqual(conv.messages, [{"role": "user", "content": "hi"}])

    def test_save_load_json(self):
        conv = Conversation()
        conv.set_system_message("sys")
        conv.add_user_message("héllo")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "conversation.json")
            conv.save(path)
            restored = Conversation.load(path)
        self.assertEqual(restored.messages, conv.messages)


class TestWireFormats(unittest.TestCase):
    def _tool_round(self):
        conv = Conversation()
        conv.set_system_message("sys")
        conv.add_user_message("weather?", context="(Earlier: user lives in Oslo)")
        call = ToolCall(name="weather", arguments={"city": "Oslo"}, id="call_1")
        conv.add_assistant_message("", tool_calls=[call])
        conv.add_tool_message('{"sky": "clear"}', tool_call=call)
        conv.add_tool_message("Now answer.")
        conv.add_assistant_message("Clear skies.")
        return conv

    def test_flat_format(self):
        self.assertEqual(
            self._tool_round().messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "(Earlier: user lives in Oslo)\nweather?"},
                {"role": "tool", "content": 'weather: {"sky": "clear"}'},
                {"role": "tool", "content": "Now answer."},
                {"role": "assistant", "content": "Clear skies."},
            ],
        )

    def test_flat_format_keeps_text_said_before_a_tool_call(self):
        conv = Conversation()
        conv.add_assistant_message("Let me check.", tool_calls=[ToolCall(name="t")])
        self.assertEqual(conv.messages, [{"role": "assistant", "content": "Let me check."}])

    def test_openai_format(self):
        self.assertEqual(
            self._tool_round().to_openai_messages(),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "(Earlier: user lives in Oslo)\nweather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": '{"city": "Oslo"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "weather", "content": '{"sky": "clear"}\n\nNow answer.'},
                {"role": "assistant", "content": "Clear skies."},
            ],
        )

    def test_openai_format_note_without_a_result_becomes_a_user_turn(self):
        conv = Conversation()
        conv.add_user_message("do it")
        conv.add_tool_message("Tool Error: Could not parse tool call.")
        self.assertEqual(
            conv.to_openai_messages()[-1], {"role": "user", "content": "Tool Error: Could not parse tool call."}
        )


class TestContextWindow(unittest.TestCase):
    def test_cutoff_idx_excludes_old_messages(self):
        conv = Conversation()
        conv.add_user_message("a")
        conv.add_assistant_message("b")
        conv.add_user_message("c")
        conv.cutoff_idx = 2
        self.assertEqual(conv.messages, [{"role": "user", "content": "c"}])
        self.assertEqual(conv.visible_count(), 1)
        self.assertEqual(len(conv.all_messages), 3)

    def test_cutoff_idx_is_clamped(self):
        conv = Conversation()
        conv.add_user_message("a")
        conv.cutoff_idx = 99
        self.assertEqual(conv.cutoff_idx, 1)
        conv.cutoff_idx = -5
        self.assertEqual(conv.cutoff_idx, 0)

    def test_hide_oldest_on_a_turn_boundary(self):
        conv = Conversation()
        exchange(conv, "q1", "a1")
        exchange(conv, "q2", "a2")
        self.assertEqual(conv.hide_oldest(2), 2)
        self.assertEqual([m.content for m in conv.visible], ["q2", "a2"])
        self.assertEqual(len(conv.all_messages), 4, "hidden messages stay in the history")

    def test_hide_oldest_rounds_forward_to_the_next_user_turn(self):
        conv = Conversation()
        exchange(conv, "q1", "a1")
        exchange(conv, "q2", "a2")
        self.assertEqual(conv.hide_oldest(1), 2)
        self.assertEqual(conv.visible[0].content, "q2")

    def test_hide_oldest_never_strands_the_current_exchange(self):
        conv = Conversation()
        exchange(conv, "q1", "a1")
        conv.add_user_message("q2")
        call = ToolCall(name="t")
        conv.add_assistant_message("", tool_calls=[call])
        conv.add_tool_message("{}", tool_call=call)
        self.assertEqual(conv.hide_oldest(4), 2)
        self.assertEqual(conv.visible[0].content, "q2", "tool results keep the call they answer")

    def test_hide_oldest_clamps_and_ignores_non_positive_counts(self):
        conv = Conversation()
        conv.add_user_message("only")
        self.assertEqual(conv.hide_oldest(0), 0)
        self.assertEqual(conv.hide_oldest(-3), 0)
        self.assertEqual(conv.hide_oldest(100), 0, "the last user turn is never hidden")
        self.assertEqual(conv.visible_count(), 1)

    def test_drop_hidden(self):
        conv = Conversation()
        exchange(conv, "q1", "a1")
        exchange(conv, "q2", "a2")
        conv.hide_oldest(2)
        dropped = conv.drop_hidden()
        self.assertEqual([m.content for m in dropped], ["q1", "a1"])
        self.assertEqual([m.content for m in conv.all_messages], ["q2", "a2"])
        self.assertEqual(conv.cutoff_idx, 0)
        self.assertEqual(conv.drop_hidden(), [])


if __name__ == "__main__":
    unittest.main()
