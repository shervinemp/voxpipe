import os
import tempfile
import unittest

from voxpipe.llm.conversation import Conversation, Message
from voxpipe.llm.tools import Tool, ToolCall, ToolSet


def turn(conv, question, answer):
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

    def test_clear_forgets_messages_only(self):
        conv = Conversation()
        conv.set_system_message("sys")
        conv.rules = ["Be brief."]
        conv.tools = [Tool(name="t1", description="")]
        for i in range(6):
            conv.add_user_message(f"m{i}")
        conv.clear()
        conv.add_user_message("fresh question")
        self.assertEqual([m.content for m in conv.history], ["fresh question"])
        self.assertEqual((conv.system_message, conv.rules, list(conv.tools)), ("sys", ["Be brief."], ["t1"]))

    def test_blank_copy_keeps_prompt_rules_and_tools_but_shares_nothing(self):
        conv = Conversation()
        conv.set_system_message("sys")
        conv.rules = ["Be brief."]
        conv.tools = [Tool(name="t1", description="")]
        conv.add_user_message("hello")

        copy = conv.blank_copy()
        self.assertEqual(copy.history, [])
        self.assertEqual((copy.system_message, copy.rules, list(copy.tools)), ("sys", ["Be brief."], ["t1"]))

        copy.rules.append("Be kind.")
        copy.tools.add(Tool(name="t2", description=""))
        self.assertEqual((conv.rules, list(conv.tools)), (["Be brief."], ["t1"]))

    def test_discard_removes_by_identity(self):
        conv = Conversation()
        first = conv.add_user_message("same text")
        conv.add_assistant_message("reply")
        second = conv.add_user_message("same text")
        conv.discard([first])
        self.assertEqual([m.content for m in conv.history], ["reply", "same text"])
        self.assertIs(conv.history[-1], second)

    def test_history_is_a_copy(self):
        conv = Conversation()
        conv.add_user_message("hello")
        conv.history.clear()
        self.assertEqual(len(conv.history), 1)

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

    def test_tools_add_keys_by_name_and_replaces(self):
        conv = Conversation()
        first, other = Tool(name="t1", description="old"), Tool(name="t2", description="")
        conv.tools.add(first, other)
        replacement = Tool(name="t1", description="new")
        conv.tools.add(replacement)
        self.assertIsInstance(conv.tools, ToolSet)
        self.assertEqual(list(conv.tools), ["t1", "t2"])
        self.assertIs(conv.tools["t1"], replacement)

    def test_to_dict_round_trip(self):
        conv = Conversation()
        conv.set_system_message("sys")
        conv.add_user_message("hello", context="(Earlier: hi)")
        call = ToolCall(name="lookup", arguments={"q": "x"})
        conv.add_assistant_message("checking", tool_calls=[call])
        conv.add_tool_message('{"a": 1}', tool_call=call)
        conv.add_tool_message("a note")
        conv.add_assistant_message("done").archived = True
        conv.rules = ["Be brief."]

        restored = Conversation.from_dict(conv.to_dict())

        self.assertEqual(restored.to_dict(), conv.to_dict())
        self.assertEqual(restored.history, conv.history)
        self.assertEqual(restored.history[1].tool_calls[0].id, call.id)
        self.assertEqual(restored.rules, ["Be brief."])

    def test_from_dict_accepts_plain_role_content_messages(self):
        conv = Conversation.from_dict({"messages": [{"role": "user", "content": "hi"}], "_meta": {"x": {}}})
        self.assertEqual(conv.messages, [{"role": "user", "content": "hi"}])

    def test_from_dict_keeps_messages_an_older_version_had_hidden(self):
        saved = {"messages": [{"role": "user", "content": f"m{i}"} for i in range(3)], "cutoff_idx": 2}
        conv = Conversation.from_dict(saved)
        self.assertEqual([m.content for m in conv.history], ["m0", "m1", "m2"])

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


class TestSystemPrompt(unittest.TestCase):
    def test_is_the_system_message_when_there_are_no_rules(self):
        conv = Conversation()
        self.assertEqual(conv.system_prompt, "")
        conv.set_system_message("You are a bot.")
        conv.tools = [Tool(name="plain", description="no instruction")]
        self.assertEqual(conv.system_prompt, "You are a bot.")

    def test_lists_tool_instructions_then_standing_rules(self):
        conv = Conversation()
        conv.set_system_message("You are a bot.")
        conv.rules = ["Be brief."]
        conv.tools = [
            Tool(name="lights", description="", instruction="Call  'lights' to\n   change the lighting."),
            Tool(name="plain", description=""),
        ]
        self.assertEqual(
            conv.system_prompt,
            "You are a bot.\n\nRules:\n- Call 'lights' to change the lighting.\n- Be brief.",
        )
        self.assertEqual(conv.system_message, "You are a bot.", "the message as set is left alone")

    def test_rules_stand_alone_without_a_system_message(self):
        conv = Conversation()
        conv.rules = ["Be brief."]
        self.assertEqual(conv.system_prompt, "Rules:\n- Be brief.")

    def test_follows_the_tools_as_they_change(self):
        conv = Conversation()
        conv.tools.add(Tool(name="lights", description="", instruction="Use lights."))
        self.assertIn("- Use lights.", conv.system_prompt)
        del conv.tools["lights"]
        self.assertEqual(conv.system_prompt, "")

    def test_both_wire_formats_send_it(self):
        conv = Conversation()
        conv.set_system_message("You are a bot.")
        conv.rules = ["Be brief."]
        conv.add_user_message("hi")
        expected = {"role": "system", "content": "You are a bot.\n\nRules:\n- Be brief."}
        self.assertEqual(conv.messages[0], expected)
        self.assertEqual(conv.to_openai_messages()[0], expected)


class TestEviction(unittest.TestCase):
    def test_evicts_on_a_turn_boundary(self):
        conv = Conversation()
        turn(conv, "q1", "a1")
        turn(conv, "q2", "a2")
        evicted = conv.evict_oldest(2)
        self.assertEqual([m.content for m in evicted], ["q1", "a1"])
        self.assertEqual([m.content for m in conv.history], ["q2", "a2"])
        self.assertEqual(conv.messages[0], {"role": "user", "content": "q2"})

    def test_rounds_forward_to_the_next_user_message(self):
        conv = Conversation()
        turn(conv, "q1", "a1")
        turn(conv, "q2", "a2")
        self.assertEqual(len(conv.evict_oldest(1)), 2)
        self.assertEqual(conv.history[0].content, "q2")

    def test_never_strands_the_current_turn(self):
        conv = Conversation()
        turn(conv, "q1", "a1")
        conv.add_user_message("q2")
        call = ToolCall(name="t")
        conv.add_assistant_message("", tool_calls=[call])
        conv.add_tool_message("{}", tool_call=call)
        self.assertEqual(len(conv.evict_oldest(4)), 2)
        self.assertEqual(conv.history[0].content, "q2", "tool results keep the call they answer")

    def test_clamps_and_ignores_non_positive_counts(self):
        conv = Conversation()
        conv.add_user_message("only")
        self.assertEqual(conv.evict_oldest(0), [])
        self.assertEqual(conv.evict_oldest(-3), [])
        self.assertEqual(conv.evict_oldest(100), [], "the last user message is never evicted")
        self.assertEqual(len(conv.history), 1)


if __name__ == "__main__":
    unittest.main()
