"""Tool definitions: schema, argument validation, execution, and the result types tools return."""

import asyncio
import functools
import inspect
import json
import re
import threading
import types
import uuid
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
)

from ..core.exceptions import ToolError

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_JSON_TYPES = frozenset({"array", "boolean", "integer", "number", "object", "string"})
_PERMISSION_MARKER = "__voxpipe_requires_permission__"
_UNION_ORIGINS = (Union, types.UnionType)
_ARRAY_TYPES = (list, tuple, set, frozenset)
_CONTAINER_TYPES: Dict[str, tuple] = {"array": (list, tuple), "object": (dict,)}

PERMISSION_KEYS = frozenset({"allow", "remember"})


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}", compare=False)


@dataclass(frozen=True)
class MalformedToolCall:
    """A tool invocation the model attempted but that could not be parsed."""

    raw: str
    name: Optional[str] = None


@dataclass(init=False)
class ToolResult:
    """What a tool produced: a JSON-able payload for the model and optional text to speak."""

    _result: dict
    speech: str | None = None
    uid: str = field(default="", compare=False)  # random per result; two results are equal by content

    def __init__(self, result: dict, speech: str | None = None, uid: str | None = None):
        if not isinstance(result, dict):
            raise TypeError(f"{type(self).__name__} result must be a dict, got {type(result).__name__}")
        self.uid = uid or f"tr_{uuid.uuid4().hex[:8]}"
        self._result = result
        self.speech = speech

    @property
    def raw_result(self) -> dict:
        return self._result

    @property
    def result_json(self) -> str:
        """The payload as JSON; values JSON has no type for are written as strings."""
        return json.dumps(self._result, default=str)

    @property
    def result(self) -> str:
        """The payload as the model is shown it."""
        return self.result_json


@dataclass(init=False)
class ToolChoice(ToolResult):
    """A question for the user: the tool needs one of these options before it can finish.

    The payload maps each key the user must settle to its allowed values (or to
    a free-form prompt), and always carries the ``uid`` that ``_confirm`` answers.
    """

    def __init__(self, result: dict, speech: str | None = None, uid: str | None = None):
        if not isinstance(result, dict):
            raise TypeError(f"ToolChoice result must be a dict, got {type(result).__name__}")
        options = dict(result)
        options.setdefault("uid", uid or f"tc_{uuid.uuid4().hex[:8]}")
        super().__init__(result=options, speech=speech, uid=options["uid"])

    @property
    def choices_dict(self) -> dict:
        return self._result

    @property
    def result(self) -> str:
        return f"Available options: {self.result_json}"


def permission(fn: Callable | None = None):
    """Mark a function as needing the user's permission before it runs as a tool.

    Usable bare (``@permission``) or called (``@permission()``), on a function
    or on a Tool already built from one.  On a function the mark is picked up
    by :meth:`Tool.from_callable`; whoever dispatches the tool enforces it via
    :meth:`Tool.permission_gate`.
    """

    def mark(f: Any) -> Any:
        if isinstance(f, Tool):
            f.requires_permission = True
        else:
            setattr(f, _PERMISSION_MARKER, True)
        return f

    return mark if fn is None else mark(fn)


@dataclass
class Tool:
    """A function the model may call, described by OpenAI's function-calling schema."""

    @dataclass
    class Parameter:
        """A JSON Schema node describing one parameter (or the parameter object itself)."""

        type: str
        description: Optional[str] = None
        properties: Optional[Dict[str, "Tool.Parameter"]] = None
        enum: Optional[List[Any]] = None
        required: Optional[List[str]] = None
        items: Optional["Tool.Parameter"] = None

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "Tool.Parameter":
            if not isinstance(data, dict):
                raise ToolError("Parameter definition must be an object.")
            json_type = data.get("type")
            if json_type not in _JSON_TYPES:
                raise ToolError("Parameter definition has an unsupported type.")
            properties = data.get("properties", {})
            if not isinstance(properties, dict):
                raise ToolError("Parameter properties must be an object.")
            required = data.get("required")
            if required is not None and not (isinstance(required, list) and all(isinstance(r, str) for r in required)):
                raise ToolError("Required parameters must be a string array.")
            if required and not set(required).issubset(properties):
                raise ToolError("Required parameters must exist in properties.")
            is_object = json_type == "object"
            items = data.get("items")
            return cls(
                type=json_type,
                description=data.get("description"),
                properties={k: cls.from_dict(v) for k, v in properties.items()} if is_object and properties else None,
                enum=data.get("enum"),
                required=required if is_object else None,
                items=cls.from_dict(items) if json_type == "array" and items is not None else None,
            )

        def to_dict(self) -> Dict[str, Any]:
            d: Dict[str, Any] = {"type": self.type}
            if self.description:
                d["description"] = self.description
            if self.type == "object":
                if self.properties:
                    d["properties"] = {k: v.to_dict() for k, v in self.properties.items()}
                if self.required:
                    d["required"] = self.required
            if self.type == "array" and self.items is not None:
                d["items"] = self.items.to_dict()
            if self.enum:
                d["enum"] = self.enum
            return d

        def __str__(self) -> str:
            return str(self.to_dict())

    _registry: ClassVar[Dict[str, "Tool"]] = {}

    name: str
    description: str
    parameters: Optional[Parameter] = None
    callback: Optional[Callable] = None
    instruction: Optional[str] = None
    may_return_choice: Optional[bool] = None
    requires_permission: bool = False
    allow_extra_arguments: bool = False

    # --- execution -------------------------------------------------------

    @property
    def is_async(self) -> bool:
        """Whether the callback is a coroutine function, or an object whose ``__call__`` is one."""
        callback = self.callback
        return inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(getattr(callback, "__call__", None))

    # ``self`` is positional-only below: the keywords are the model's, and it may name one "self".

    def __call__(self, /, **kwargs) -> ToolResult:
        return self.execute(**kwargs)

    def execute(self, /, **kwargs) -> ToolResult:
        """Validate the arguments, run the callback, and normalise what it returns.

        This does not consult :attr:`requires_permission`; consent belongs to
        the caller, which checks :meth:`permission_gate` first.
        """
        if self.is_async:
            return asyncio.run(self.aexecute(**kwargs))
        callback = self._require_callback()
        return self._normalize(callback(**self._coerce_arguments(kwargs)))

    async def aexecute(self, /, **kwargs) -> ToolResult:
        """Like :meth:`execute`, awaiting async callbacks and running blocking ones in a thread.

        An async callback runs on the caller's event loop: one that blocks
        instead of awaiting holds up everything else on that loop.
        """
        callback = self._require_callback()
        arguments = self._coerce_arguments(kwargs)
        if self.is_async:
            return self._normalize(await callback(**arguments))
        return self._normalize(await _call_in_thread(callback, arguments))

    def permission_gate(self, granted: Optional[bool]) -> Optional[ToolResult]:
        """Return what to answer instead of running the tool, or None when it may run.

        ``granted`` is the remembered decision for this tool: True, False, or
        None when the user has not been asked.
        """
        if not self.requires_permission or granted is True:
            return None
        if granted is False:
            return ToolResult(
                {"error": f"Permission permanently denied for tool '{self.name}'"},
                speech=f"Permission for {self.name} was denied.",
            )
        return ToolChoice(
            {"allow": [True, False], "remember": [True, False]},
            speech=f"Permission requested to run {self.name}.",
        )

    def _require_callback(self) -> Callable:
        if self.callback is None:
            raise ToolError(f"Tool '{self.name}' has no callback registered.")
        return self.callback

    def _normalize(self, result: Any) -> ToolResult:
        if result is None:
            return ToolResult(result={"status": "success"})
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, dict):
            return ToolResult(result=result)
        if isinstance(result, (str, int, float, bool)):
            return ToolResult(result={"result": result})
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], str):
            payload = result[0] if isinstance(result[0], dict) else {"result": result[0]}
            return ToolResult(result=payload, speech=result[1])
        raise TypeError(
            f"Tool '{self.name}' callback returned {type(result).__name__}; "
            "expected a dict, scalar, ToolResult, or (dict, speech) tuple"
        )

    def _coerce_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Check model-supplied arguments against the schema and cast scalars to their declared types."""
        if self.parameters is None:
            return arguments

        properties = self.parameters.properties or {}
        if not self.allow_extra_arguments:
            unexpected = set(arguments) - set(properties)
            if unexpected:
                raise ToolError(f"Unexpected tool arguments: {', '.join(sorted(unexpected))}")
        required = set(self.parameters.required or ())
        missing = required - set(arguments)
        if missing:
            raise ToolError(f"Missing required tool arguments: {', '.join(sorted(missing))}")

        coerced: Dict[str, Any] = {}
        for name, value in arguments.items():
            parameter = properties.get(name)
            if parameter is None:
                coerced[name] = value
            elif value is None:
                # JSON null for an optional parameter means "not given": the callback's default applies.
                if name in required:
                    raise ToolError(f"Tool argument '{name}' is required and must not be null.")
            else:
                value = _cast(name, value, parameter.type)
                if parameter.enum and value not in parameter.enum:
                    allowed = ", ".join(repr(option) for option in parameter.enum)
                    raise ToolError(f"Tool argument '{name}' must be one of: {allowed}.")
                coerced[name] = value
        return coerced

    # --- construction ----------------------------------------------------

    @staticmethod
    def from_callable(name: str, fn: Callable, *, may_return_choice: Optional[bool] = None) -> "Tool":
        """Build a Tool from a function: schema from its signature, descriptions from its docstring.

        A parameter without a type hint is declared, and passed on, as a
        string.  Arguments a ``functools.partial`` has already bound by keyword
        are left out of the schema, so the model can neither see nor override them.
        """
        if not isinstance(name, str) or not _IDENTIFIER_RE.fullmatch(name):
            raise ToolError("Tool name must be a valid 1-64 character identifier.")
        bound = set(fn.keywords) if isinstance(fn, functools.partial) else set()
        doc = _parse_docstring(inspect.getdoc(fn.func if isinstance(fn, functools.partial) else fn))
        signature = inspect.signature(fn)
        hints = _resolve_hints(fn, signature)

        properties: Dict[str, Tool.Parameter] = {}
        required: List[str] = []
        allow_extra = False
        for param_name, param in signature.parameters.items():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                allow_extra = True
                continue
            if param_name == "self" or param_name in bound or param.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            properties[param_name] = _parameter_for(hints[param_name], doc.params.get(param_name))
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        if may_return_choice is None and _returns_choice(hints["return"]):
            may_return_choice = True

        return Tool(
            name=name,
            description=doc.description or "No description provided.",
            parameters=Tool.Parameter(type="object", properties=properties, required=required or None)
            if properties
            else None,
            callback=fn,
            instruction=doc.instruction or None,
            may_return_choice=may_return_choice,
            requires_permission=bool(getattr(fn, _PERMISSION_MARKER, False)),
            allow_extra_arguments=allow_extra,
        )

    @staticmethod
    def from_class(klass: Type) -> Callable[..., List["Tool"]]:
        """Class decorator turning a class into a factory of tools.

        Calling the decorated name builds an instance (forwarding the
        arguments) and returns one Tool per public method (instance, class or
        static), bound to it.  Properties are not touched.
        """
        methods = [
            name
            for name, member in inspect.getmembers_static(klass)
            if not name.startswith("_") and isinstance(member, (types.FunctionType, classmethod, staticmethod))
        ]

        def factory(*args, **kwargs) -> List["Tool"]:
            instance = klass(*args, **kwargs)
            return [Tool.from_callable(name, getattr(instance, name)) for name in methods]

        factory.__name__ = klass.__name__
        factory.__qualname__ = klass.__qualname__
        factory.__doc__ = klass.__doc__
        return factory

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the OpenAI tool schema (also what llama-cpp-python expects)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.to_dict() if self.parameters else {"type": "object", "properties": {}},
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tool":
        """Build a Tool (without a callback) from an OpenAI tool schema or the legacy flat schema."""
        if not isinstance(d, dict):
            raise ToolError("Tool definition must be an object.")

        # ASVS 1.5.2 / 2.2.1: deserialize only the expected structure and
        # validate identifiers before they become dispatch keys.
        if "type" in d and d["type"] != "function":
            raise ToolError("Tool type must be 'function'.")
        function_data = d.get("function") or d
        if not isinstance(function_data, dict):
            raise ToolError("Tool function definition must be an object.")
        name = function_data.get("name")
        if not isinstance(name, str) or not _IDENTIFIER_RE.fullmatch(name):
            raise ToolError("Tool name must be a valid 1-64 character identifier.")

        parameters_data = function_data.get("parameters")
        legacy_params = function_data.get("params")
        if parameters_data is None and legacy_params is not None:
            if not isinstance(legacy_params, list):
                raise ToolError("Legacy tool params must be an array.")
            if not all(isinstance(p, str) and _IDENTIFIER_RE.fullmatch(p) for p in legacy_params):
                raise ToolError("Tool parameter names must be valid identifiers.")
            parameters_data = {
                "type": "object",
                "properties": {p: {"type": "string"} for p in legacy_params},
                "required": list(legacy_params),
            }

        parameters = cls.Parameter.from_dict(parameters_data) if parameters_data else None
        if parameters is not None and parameters.type != "object":
            raise ToolError("Tool parameters must be described by an object schema.")
        return cls(
            name=name,
            description=function_data.get("description") or "No description provided.",
            parameters=parameters,
            instruction=function_data.get("instruction"),
        )

    # --- registry --------------------------------------------------------

    @classmethod
    def register(cls, fn: Union[Callable, "Tool", None] = None, *, name: str | None = None):
        """Register a tool process-wide; usable as ``@Tool.register`` or ``@Tool.register(name=...)``."""

        def add(target: Union[Callable, "Tool"]) -> "Tool":
            if isinstance(target, Tool):
                tool = target
            elif callable(target):
                tool = cls.from_callable(name or target.__name__, target)
            else:
                raise ToolError("Tool.register expects a callable or Tool instance.")
            cls._registry[tool.name] = tool
            return tool

        return add if fn is None else add(fn)

    @classmethod
    def registry(cls) -> Dict[str, "Tool"]:
        return dict(cls._registry)

    @classmethod
    def clear_registry(cls) -> None:
        cls._registry.clear()


class ToolSet(Dict[str, Tool]):
    """Tools keyed by name."""

    def __init__(self, tools: Union[Mapping[str, Tool], Iterable[Tool]] = ()):
        super().__init__(tools if isinstance(tools, Mapping) else {tool.name: tool for tool in tools})

    def add(self, *tools: Tool) -> None:
        """Add tools under their own names, replacing any that share one."""
        for tool in tools:
            self[tool.name] = tool


async def _call_in_thread(callback: Callable, arguments: Dict[str, Any]) -> Any:
    """Run a blocking callback off the event loop and await what it returns.

    Each call gets its own daemon thread rather than a slot in the loop's
    executor: a callback that never returns then neither uses up a shared
    worker pool nor keeps the interpreter from exiting.
    """
    loop = asyncio.get_running_loop()
    future: "asyncio.Future[Any]" = loop.create_future()

    def deliver(error: Optional[BaseException], result: Any) -> None:
        if future.done():  # the caller stopped waiting
            return
        if error is None:
            future.set_result(result)
        else:
            future.set_exception(error)

    def run() -> None:
        error, result = None, None
        try:
            result = callback(**arguments)
        except Exception as e:  # noqa: BLE001 - whatever the callback raises belongs to the awaiting caller
            error = e
        except BaseException as e:  # noqa: BLE001 - SystemExit and the like must not take the loop down
            error = RuntimeError(f"tool callback exited with {type(e).__name__}")
        try:
            loop.call_soon_threadsafe(deliver, error, result)
        except RuntimeError:
            pass  # the loop has closed; nobody is waiting any more

    threading.Thread(target=run, name="voxpipe-tool-call", daemon=True).start()
    return await future


# --- argument casting -------------------------------------------------------


def _cast(name: str, value: Any, json_type: str) -> Any:
    if json_type == "integer":
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ToolError(f"Tool argument '{name}' must be an integer.")
        try:
            return int(value)
        except (ValueError, TypeError):
            raise ToolError(f"Tool argument '{name}' must be an integer.") from None
    if json_type == "number":
        if isinstance(value, bool):
            raise ToolError(f"Tool argument '{name}' must be a number.")
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ToolError(f"Tool argument '{name}' must be a number.") from None
    if json_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ToolError(f"Tool argument '{name}' must be a boolean.")
    if json_type == "string":
        if isinstance(value, (dict, list)):
            raise ToolError(f"Tool argument '{name}' must be a string.")
        return str(value)
    if json_type in _CONTAINER_TYPES:
        if isinstance(value, str):  # models often send nested JSON as a string
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        if not isinstance(value, _CONTAINER_TYPES[json_type]):
            raise ToolError(f"Tool argument '{name}' must be an {json_type}.")
    return value


# --- type hints to JSON schema ----------------------------------------------


def _resolve_hints(fn: Callable, signature: inspect.Signature) -> Dict[str, Any]:
    """The type hint of each parameter and of the return value (under "return").

    Hints written as strings (``from __future__ import annotations``) are
    evaluated one at a time, so one that cannot be resolved, such as a name
    imported only for type checking, does not cost the others their types.
    """
    namespace = getattr(inspect.unwrap(fn.func if isinstance(fn, functools.partial) else fn), "__globals__", {})

    def resolve(annotation: Any) -> Any:
        if not isinstance(annotation, str):
            return annotation
        try:
            return eval(annotation, namespace)  # noqa: S307 - the function's own annotation, as inspect does
        except Exception:  # noqa: BLE001 - an unresolvable hint is treated as no hint
            return inspect.Parameter.empty

    hints = {name: resolve(param.annotation) for name, param in signature.parameters.items()}
    hints["return"] = resolve(signature.return_annotation)
    return hints


def _strip_optional(annotation: Any) -> Any:
    """``Optional[X]`` / ``X | None`` / ``Annotated[X, ...]`` → ``X``; other unions resolve to their first member."""
    if get_origin(annotation) is Annotated:
        return _strip_optional(get_args(annotation)[0])
    if get_origin(annotation) in _UNION_ORIGINS:
        members = [a for a in get_args(annotation) if a is not type(None)]
        return _strip_optional(members[0]) if members else str
    return annotation


def _get_json_type(annotation: Any) -> str:
    """Map a Python type hint to a JSON Schema type name, defaulting to "string"."""
    annotation = _strip_optional(annotation)
    origin = get_origin(annotation)

    if origin is Literal:
        values = get_args(annotation)
        return _get_json_type(type(values[0])) if values else "string"
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation in _ARRAY_TYPES or origin in _ARRAY_TYPES:
        return "array"
    if annotation is dict or origin is dict:
        return "object"
    return "string"


def _parameter_for(annotation: Any, description: Optional[str]) -> Tool.Parameter:
    inner = _strip_optional(annotation)
    json_type = _get_json_type(inner)
    items = None
    if json_type == "array" and get_args(inner):
        items = _parameter_for(get_args(inner)[0], None)
    return Tool.Parameter(
        type=json_type,
        description=description,
        enum=list(get_args(inner)) if get_origin(inner) is Literal else None,
        items=items,
    )


def _returns_choice(annotation: Any) -> bool:
    """Whether a return annotation admits a ToolChoice."""
    if get_origin(annotation) in _UNION_ORIGINS:
        return any(_returns_choice(a) for a in get_args(annotation))
    return isinstance(annotation, type) and issubclass(annotation, ToolChoice)


# --- docstring parsing ------------------------------------------------------


@dataclass
class _DocInfo:
    description: str = ""
    params: Dict[str, str] = field(default_factory=dict)
    instruction: str = ""


_SECTIONS = {
    "args": "params",
    "arguments": "params",
    "parameters": "params",
    "returns": "skip",
    "return": "skip",
    "yields": "skip",
    "raises": "skip",
    "example": "skip",
    "examples": "skip",
    "instruction": "instruction",
}
_SECTION_RE = re.compile(r"^(\w+):\s*$")
_RST_FIELD_RE = re.compile(
    r"^:(param\s+(?:[\w\[\], |.]+\s+)?\w+|returns?|instruction|raises\s+\w+|rtype|type\s+\w+):\s*(.*)$"
)
_GOOGLE_PARAM_RE = re.compile(r"^\*{0,2}(\w+)\s*(?:\([^)]*\))?:\s*(.*)$")


def _parse_docstring(docstring: Optional[str]) -> _DocInfo:
    """Extract a tool's description, per-parameter descriptions and usage instruction.

    Understands Google-style sections (``Args:``, ``Returns:``, ``Instruction:``)
    and reST fields (``:param x:``, ``:returns:``, ``:instruction:``).  Whatever
    documents return values or exceptions is left out of the description.
    """
    info = _DocInfo()
    if not docstring:
        return info

    description: List[str] = []
    instruction: List[str] = []
    section = "description"
    item_indent: Optional[int] = None  # indentation of entries in the current Google section
    target: Optional[str] = None  # what continuation lines extend: a param name, "instruction", or None

    for raw in inspect.cleandoc(docstring).splitlines():
        line = raw.strip()
        indent = len(raw) - len(raw.lstrip())

        header = _SECTION_RE.match(line)
        if indent == 0 and header and header.group(1).lower() in _SECTIONS:
            section, item_indent, target = _SECTIONS[header.group(1).lower()], None, None
            continue

        rst = _RST_FIELD_RE.match(line)
        if rst:
            kind, text = rst.group(1).split()[0], rst.group(2).strip()
            section, item_indent = "rst", indent
            if kind == "param":
                target = rst.group(1).split()[-1]  # ":param city:" or ":param str city:"
                info.params[target] = text
            elif kind == "instruction":
                target = "instruction"
                instruction.append(text)
            else:
                target = None
            continue

        if not line:
            if section in ("description", "rst"):
                description.append("")
                section = "description"
            continue

        if section == "rst":
            if indent > (item_indent or 0):
                _extend(info, instruction, target, line)
                continue
            section = "description"
        elif section != "description":
            if indent == 0:
                section = "description"
            elif section == "params":
                if item_indent is None:
                    item_indent = indent
                entry = _GOOGLE_PARAM_RE.match(line) if indent == item_indent else None
                if entry:
                    target = entry.group(1)
                    info.params[target] = entry.group(2).strip()
                else:
                    _extend(info, instruction, target, line)
                continue
            elif section == "instruction":
                instruction.append(line)
                continue
            else:
                continue

        description.append(line)

    info.description = "\n".join(description).strip()
    info.instruction = " ".join(part for part in instruction if part)
    return info


def _extend(info: _DocInfo, instruction: List[str], target: Optional[str], line: str) -> None:
    """Append a continuation line to whichever entry is being described."""
    if target == "instruction":
        instruction.append(line)
    elif target is not None:
        info.params[target] = f"{info.params[target]} {line}".strip()
