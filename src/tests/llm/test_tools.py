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
        from voxpipe.llm.tools import Tool, ToolChoice
        def fn(x: int) -> ToolChoice:
            return ToolChoice(result=str(x))
        t = Tool.from_callable("test", fn)
        self.assertTrue(t.may_return_choice)

    def test_inferred_from_optional_annotation(self):
        from typing import Optional
        from voxpipe.llm.tools import Tool, ToolChoice
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


class TestPermission(unittest.TestCase):
    """UID auto-generation and the permission gate."""

    @staticmethod
    def _delete_tool():
        from voxpipe.llm.tools import Tool

        def sensitive_action(target: str) -> dict:
            return {"deleted": target}

        tool = Tool.from_callable("delete", sensitive_action)
        tool.requires_permission = True
        return tool

    def test_tool_choice_auto_uid(self):
        from voxpipe.llm.tools import ToolChoice
        tc = ToolChoice(result={"choice": ["A", "B"]})
        self.assertTrue(tc.uid.startswith("tc_"))
        self.assertEqual(tc.choices_dict["uid"], tc.uid)

    def test_tool_choice_keeps_a_given_uid_and_leaves_the_input_alone(self):
        from voxpipe.llm.tools import ToolChoice
        options = {"choice": ["A", "B"]}
        self.assertEqual(ToolChoice(result=options, uid="tc_fixed").uid, "tc_fixed")
        self.assertEqual(ToolChoice(result={"uid": "tc_payload"}).uid, "tc_payload")
        self.assertEqual(options, {"choice": ["A", "B"]})

    def test_gate_asks_when_the_user_has_not_decided(self):
        from voxpipe.llm.tools import PERMISSION_KEYS, ToolChoice
        gate = self._delete_tool().permission_gate(None)
        self.assertIsInstance(gate, ToolChoice)
        self.assertEqual(set(gate.choices_dict) - {"uid"}, PERMISSION_KEYS)
        self.assertTrue(gate.uid.startswith("tc_"))

    def test_gate_opens_when_granted(self):
        self.assertIsNone(self._delete_tool().permission_gate(True))

    def test_gate_refuses_when_denied(self):
        from voxpipe.llm.tools import ToolChoice
        gate = self._delete_tool().permission_gate(False)
        self.assertNotIsInstance(gate, ToolChoice)
        self.assertIn("permanently denied", gate.result)

    def test_gate_is_open_for_ordinary_tools(self):
        from voxpipe.llm.tools import Tool
        tool = Tool.from_callable("plain", lambda: None)
        for granted in (None, True, False):
            self.assertIsNone(tool.permission_gate(granted))

    def test_permission_decorator_marks_the_tool(self):
        from voxpipe.llm.tools import Tool, permission

        @permission
        def bare(target: str):
            return {"deleted": target}

        @permission()
        def called(target: str):
            return {"deleted": target}

        for fn in (bare, called):
            with self.subTest(fn=fn.__name__):
                tool = Tool.from_callable(fn.__name__, fn)
                self.assertTrue(tool.requires_permission)
                self.assertEqual(fn(target="x"), {"deleted": "x"}, "the function itself is unchanged")
                self.assertIn("x", tool(target="x").result, "consent is the dispatcher's job, not execute()'s")


class TestSchemaInference(unittest.TestCase):
    """from_callable: type hints and docstrings to JSON schema."""

    def _schema(self, fn):
        from voxpipe.llm.tools import Tool
        return Tool.from_callable(fn.__name__, fn).to_dict()["function"]

    def test_union_syntaxes(self):
        from typing import Optional, Union

        def fn(a: Optional[int] = None, b: int | None = None, c: Union[float, None] = None, d: bool | None = None):
            pass

        props = self._schema(fn)["parameters"]["properties"]
        self.assertEqual({k: v["type"] for k, v in props.items()},
                         {"a": "integer", "b": "integer", "c": "number", "d": "boolean"})

    def test_containers_and_literals(self):
        from typing import Literal

        def fn(names: list[str], bare: list, pairs: tuple[int, ...], opts: dict, unit: Literal["c", "f"] = "c"):
            pass

        params = self._schema(fn)["parameters"]
        props = params["properties"]
        self.assertEqual(props["names"], {"type": "array", "items": {"type": "string"}})
        self.assertEqual(props["bare"], {"type": "array"})
        self.assertEqual(props["pairs"], {"type": "array", "items": {"type": "integer"}})
        self.assertEqual(props["opts"], {"type": "object"})
        self.assertEqual(props["unit"], {"type": "string", "enum": ["c", "f"]})
        self.assertEqual(params["required"], ["names", "bare", "pairs", "opts"])

    def test_var_keyword_is_not_a_parameter(self):
        from voxpipe.llm.tools import Tool

        def fn(count: int | None = None, **extra):
            return {"count": count, "extra": extra}

        tool = Tool.from_callable("fn", fn)
        self.assertEqual(tool.to_dict()["function"]["parameters"], {"type": "object", "properties": {"count": {"type": "integer"}}})
        self.assertEqual(tool(count="5", color="red").raw_result, {"count": 5, "extra": {"color": "red"}})

    def test_only_var_keyword_accepts_anything(self):
        from voxpipe.llm.tools import Tool
        tool = Tool.from_callable("fn", lambda **kwargs: kwargs)
        self.assertIsNone(tool.parameters)
        self.assertEqual(tool(x=1).raw_result, {"x": 1})

    def test_union_return_may_be_a_choice(self):
        from voxpipe.llm.tools import Tool, ToolChoice, ToolResult

        def fn() -> ToolResult | ToolChoice:
            return ToolResult({})

        self.assertTrue(Tool.from_callable("fn", fn).may_return_choice)

    def test_google_docstring(self):
        def weather(city: str, unit: str = "c"):
            """Get the weather.

            Longer explanation of what it does.

            Args:
                city: The city to look up,
                    including country.
                unit (str): Temperature unit.

            Returns:
                A dict with the temperature.

            Raises:
                ValueError: If the city is unknown.

            Instruction:
                Call when the user asks about weather
                or what to wear.
            """

        from voxpipe.llm.tools import Tool
        tool = Tool.from_callable("weather", weather)
        self.assertEqual(tool.description, "Get the weather.\n\nLonger explanation of what it does.")
        props = tool.parameters.properties
        self.assertEqual(props["city"].description, "The city to look up, including country.")
        self.assertEqual(props["unit"].description, "Temperature unit.")
        self.assertEqual(tool.instruction, "Call when the user asks about weather or what to wear.")

    def test_rest_docstring(self):
        def weather(city: str):
            """Get the weather.

            :param city: The city to look up,
                including country.
            :returns: A dict with the temperature.
            :instruction: Call when the user asks about weather.
            """

        from voxpipe.llm.tools import Tool
        tool = Tool.from_callable("weather", weather)
        self.assertEqual(tool.description, "Get the weather.")
        self.assertEqual(tool.parameters.properties["city"].description, "The city to look up, including country.")
        self.assertEqual(tool.instruction, "Call when the user asks about weather.")

    def test_no_docstring(self):
        from voxpipe.llm.tools import Tool
        tool = Tool.from_callable("fn", lambda: None)
        self.assertEqual(tool.description, "No description provided.")
        self.assertIsNone(tool.instruction)


class TestExecution(unittest.TestCase):
    """Argument validation, casting, result normalisation and async callbacks."""

    def _tool(self):
        from voxpipe.llm.tools import Tool

        def fn(count: int, ratio: float = 1.0, flag: bool = False, label: str = ""):
            return {"count": count, "ratio": ratio, "flag": flag, "label": label}

        return Tool.from_callable("fn", fn)

    def test_scalars_are_cast_to_declared_types(self):
        result = self._tool()(count="3", ratio="0.5", flag="TRUE", label=7).raw_result
        self.assertEqual(result, {"count": 3, "ratio": 0.5, "flag": True, "label": "7"})
        self.assertEqual(self._tool()(count=4.0).raw_result["count"], 4)

    def test_invalid_arguments_raise_tool_error(self):
        from voxpipe.core.exceptions import ToolError
        cases = [
            {},  # missing required
            {"count": 1, "bogus": 2},  # unexpected
            {"count": "three"},
            {"count": 3.7},
            {"count": True},
            {"count": 1, "ratio": "fast"},
            {"count": 1, "flag": "maybe"},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ToolError):
                self._tool()(**arguments)

    def test_missing_callback_raises_tool_error(self):
        from voxpipe.core.exceptions import ToolError
        from voxpipe.llm.tools import Tool
        with self.assertRaises(ToolError):
            Tool(name="t", description="")()

    def test_result_normalisation(self):
        from voxpipe.llm.tools import Tool, ToolResult
        ready = ToolResult({"a": 1}, speech="said")
        cases = [
            (None, {"status": "success"}, None),
            ({"a": 1}, {"a": 1}, None),
            ("text", {"result": "text"}, None),
            (100, {"result": 100}, None),
            (({"a": 1}, "spoken"), {"a": 1}, "spoken"),
            (("plain", "spoken"), {"result": "plain"}, "spoken"),
            (ready, {"a": 1}, "said"),
        ]
        for returned, payload, speech in cases:
            with self.subTest(returned=returned):
                result = Tool.from_callable("t", lambda: returned)()
                self.assertEqual((result.raw_result, result.speech), (payload, speech))
        with self.assertRaises(TypeError):
            Tool.from_callable("t", lambda: object())()

    def test_async_callback(self):
        import asyncio
        from voxpipe.llm.tools import Tool

        async def fetch(q: str):
            await asyncio.sleep(0)
            return {"q": q}

        tool = Tool.from_callable("fetch", fetch)
        self.assertTrue(tool.is_async)
        self.assertEqual(tool(q="x").raw_result, {"q": "x"})
        self.assertEqual(asyncio.run(tool.aexecute(q="y")).raw_result, {"q": "y"})

    def test_aexecute_runs_blocking_callbacks_off_the_loop(self):
        import asyncio
        import threading
        from voxpipe.llm.tools import Tool

        tool = Tool.from_callable("where", lambda: {"thread": threading.get_ident()})

        async def run():
            return (await tool.aexecute()).raw_result["thread"], threading.get_ident()

        worker, loop_thread = asyncio.run(run())
        self.assertNotEqual(worker, loop_thread)


class TestFromClassAndRegistry(unittest.TestCase):
    def setUp(self):
        from voxpipe.llm.tools import Tool
        Tool.clear_registry()
        self.addCleanup(Tool.clear_registry)

    def test_from_class(self):
        from voxpipe.llm.tools import Tool

        @Tool.from_class
        class Lights:
            """Controls the lights."""

            def __init__(self, room: str):
                self.room = room

            def turn_on(self) -> dict:
                """Turn the lights on."""
                return {"room": self.room, "on": True}

            def _internal(self):
                pass

        tools = Lights("kitchen")
        self.assertEqual([t.name for t in tools], ["turn_on"])
        self.assertEqual(tools[0]().raw_result, {"room": "kitchen", "on": True})
        self.assertIsNone(tools[0].parameters, "self is not a parameter")

    def test_register_forms(self):
        from voxpipe.llm.tools import Tool

        @Tool.register
        def alpha():
            """A."""

        @Tool.register(name="renamed")
        def beta():
            """B."""

        gamma = Tool.register(Tool(name="gamma", description="G"))

        self.assertEqual(set(Tool.registry()), {"alpha", "renamed", "gamma"})
        self.assertIs(Tool.registry()["gamma"], gamma)
        self.assertIsInstance(alpha, Tool)

    def test_registry_returns_a_copy(self):
        from voxpipe.llm.tools import Tool
        Tool.register(Tool(name="t", description=""))
        Tool.registry().clear()
        self.assertIn("t", Tool.registry())

    def test_register_rejects_non_callables(self):
        from voxpipe.core.exceptions import ToolError
        from voxpipe.llm.tools import Tool
        with self.assertRaises(ToolError):
            Tool.register(42)

