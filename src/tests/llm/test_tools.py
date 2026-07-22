import unittest


class TestToolResultChoice(unittest.TestCase):
    """ToolChoice and ToolResult behavior."""

    def test_tool_choice_result_prefix(self):
        from voxpipe.llm.tools import ToolChoice
        c = ToolChoice(result={"slot": ["1", "2"]})
        self.assertIn("Available options:", c.result)
        self.assertIn("slot", c.result)
        self.assertIn("1", c.result)

    def test_tool_choice_inherits_speech(self):
        from voxpipe.llm.tools import ToolChoice
        c = ToolChoice(result={"item": "test"}, speech="Which one?")
        self.assertEqual(c.speech, "Which one?")

    def test_tool_choice_no_speech(self):
        from voxpipe.llm.tools import ToolChoice
        c = ToolChoice(result={"slots": ["a", "b"]})
        self.assertIsNone(c.speech)

    def test_tool_choice_is_tool_result(self):
        from voxpipe.llm.tools import ToolChoice, ToolResult
        self.assertTrue(issubclass(ToolChoice, ToolResult))


class TestReturnsChoice(unittest.TestCase):
    """_returns_choice annotation inspection."""

    def test_bare_tool_choice(self):
        from voxpipe.llm.tools import _returns_choice, ToolChoice
        self.assertTrue(_returns_choice(ToolChoice))

    def test_optional_tool_choice(self):
        from typing import Optional
        from voxpipe.llm.tools import _returns_choice, ToolChoice
        self.assertTrue(_returns_choice(Optional[ToolChoice]))

    def test_union_tool_choice(self):
        from typing import Union
        from voxpipe.llm.tools import _returns_choice, ToolChoice, ToolResult
        self.assertTrue(_returns_choice(Union[ToolChoice, ToolResult]))

    def test_bare_str(self):
        from voxpipe.llm.tools import _returns_choice
        self.assertFalse(_returns_choice(str))

    def test_none_type(self):
        from voxpipe.llm.tools import _returns_choice
        self.assertFalse(_returns_choice(type(None)))

    def test_none_value(self):
        from voxpipe.llm.tools import _returns_choice
        self.assertFalse(_returns_choice(None))

    def test_list_of_str(self):
        from voxpipe.llm.tools import _returns_choice
        self.assertFalse(_returns_choice(list[str]))


class TestMayReturnChoice(unittest.TestCase):
    """Tool.may_return_choice from_callable behavior."""

    def test_inferred_from_annotation(self):
        from voxpipe.llm.tools import Tool, ToolChoice, ToolResult
        def fn(x: int) -> ToolChoice:
            return ToolChoice(result=str(x))
        t = Tool.from_callable("test", fn)
        self.assertTrue(t.may_return_choice)

    def test_inferred_from_optional_annotation(self):
        from typing import Optional
        from voxpipe.llm.tools import Tool, ToolChoice, ToolResult
        def fn(x: int) -> Optional[ToolChoice]:
            return ToolChoice(result=str(x))
        t = Tool.from_callable("test", fn)
        self.assertTrue(t.may_return_choice)

    def test_not_inferred_when_tool_result(self):
        from voxpipe.llm.tools import Tool, ToolResult
        def fn(x: int) -> ToolResult:
            return ToolResult(result=str(x))
        t = Tool.from_callable("test", fn)
        self.assertIsNone(t.may_return_choice)

    def test_not_inferred_when_str(self):
        from voxpipe.llm.tools import Tool
        def fn(x: int) -> str:
            return str(x)
        t = Tool.from_callable("test", fn)
        self.assertIsNone(t.may_return_choice)

    def test_manual_true(self):
        from voxpipe.llm.tools import Tool, ToolResult
        def fn(x: int) -> ToolResult:
            return ToolResult(result=str(x))
        t = Tool.from_callable("test", fn, may_return_choice=True)
        self.assertTrue(t.may_return_choice)

    def test_manual_false(self):
        from voxpipe.llm.tools import Tool, ToolChoice
        def fn(x: int) -> ToolChoice:
            return ToolChoice(result=str(x))
        t = Tool.from_callable("test", fn, may_return_choice=False)
        self.assertFalse(t.may_return_choice)


class TestToolSerialization(unittest.TestCase):
    """Tool schema generation, from_callable, to_dict."""

    def test_from_callable_no_params(self):
        from voxpipe.llm.tools import Tool
        def fn():
            return 42
        t = Tool.from_callable("my_tool", fn)
        self.assertEqual(t.name, "my_tool")
        self.assertIsNotNone(t.description)

    def test_from_callable_with_params(self):
        from voxpipe.llm.tools import Tool
        def greet(name: str, age: int = 0):
            """Say hello.

            Args:
                name: The person's name.
                age: Their age.
            """
            return f"Hello {name}"
        t = Tool.from_callable("greet", greet)
        d = t.to_dict()
        self.assertEqual(d["function"]["name"], "greet")
        props = d["function"]["parameters"]["properties"]
        self.assertIn("name", props)
        self.assertIn("age", props)
        self.assertEqual(props["name"]["type"], "string")
        self.assertEqual(props["age"]["type"], "integer")

    def test_to_dict_round_trip(self):
        from voxpipe.llm.tools import Tool
        t1 = Tool(name="test", description="desc",
                   parameters=Tool.Parameter(type="object"))
        d = t1.to_dict()
        t2 = Tool.from_dict(d)
        self.assertEqual(t2.name, "test")
        self.assertEqual(t2.description, "desc")

    def test_call_calls_backend(self):
        from voxpipe.llm.tools import Tool, ToolResult
        captured = {}
        def fn(x: int) -> dict:
            captured["x"] = x
            return {"doubled": x * 2}
        t = Tool.from_callable("double", fn)
        result = t(x=5)
        self.assertEqual(captured["x"], 5)
        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.speech, None)
        self.assertIn("doubled", result.result)
        self.assertIn("10", result.result)

    def test_parameter_from_dict_to_dict(self):
        from voxpipe.llm.tools import Tool
        p = Tool.Parameter(type="object", properties={
            "name": Tool.Parameter(type="string", description="The name"),
        }, required=["name"])
        d = p.to_dict()
        p2 = Tool.Parameter.from_dict(d)
        self.assertEqual(p2.type, "object")
        self.assertIn("name", p2.properties)

    def test_parameter_no_description(self):
        from voxpipe.llm.tools import Tool
        p = Tool.Parameter(type="string")
        d = p.to_dict()
        self.assertEqual(d["type"], "string")
        self.assertNotIn("description", d)


class TestPermissionAndMeta(unittest.TestCase):
    """Tool._meta, UID auto-generation, and permission interception behavior."""

    def setUp(self):
        from voxpipe.llm.tools import Tool
        Tool.clear_meta()

    def test_tool_choice_auto_uid(self):
        from voxpipe.llm.tools import ToolChoice
        tc = ToolChoice(result={"choice": ["A", "B"]})
        self.assertTrue(tc.uid.startswith("tc_"))
        self.assertIn("uid", tc.choices_dict)

    def test_tool_meta_get_set_clear(self):
        from voxpipe.llm.tools import Tool
        Tool.set_meta("my_tool", "custom_key", 123)
        self.assertEqual(Tool.get_meta("my_tool").get("custom_key"), 123)
        Tool.clear_meta()
        self.assertEqual(Tool.get_meta("my_tool"), {})

    def test_permission_interception_none(self):
        from voxpipe.llm.tools import Tool, ToolChoice
        def sensitive_action(target: str) -> dict:
            return {"deleted": target}
        t = Tool.from_callable("delete", sensitive_action)
        t.requires_permission = True
        res = t(target="file.txt")
        self.assertIsInstance(res, ToolChoice)
        self.assertIn("allow", res.choices_dict)
        self.assertIn("remember", res.choices_dict)
        self.assertTrue(res.uid.startswith("tc_"))

    def test_permission_interception_remember_true(self):
        from voxpipe.llm.tools import Tool, ToolResult
        def sensitive_action(target: str) -> dict:
            return {"deleted": target}
        t = Tool.from_callable("delete", sensitive_action)
        t.requires_permission = True
        Tool.set_meta("delete", "_permission", True)
        res = t(target="file.txt")
        self.assertIsInstance(res, ToolResult)
        self.assertIn("file.txt", res.result)

    def test_permission_interception_remember_false(self):
        from voxpipe.llm.tools import Tool, ToolResult
        def sensitive_action(target: str) -> dict:
            return {"deleted": target}
        t = Tool.from_callable("delete", sensitive_action)
        t.requires_permission = True
        Tool.set_meta("delete", "_permission", False)
        res = t(target="file.txt")
        self.assertIsInstance(res, ToolResult)
        self.assertIn("permanently denied", res.result)

