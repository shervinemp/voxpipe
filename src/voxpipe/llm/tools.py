"""Tool definitions: schema, argument validation, execution, and the result types tools return."""

import asyncio
import inspect
import json
import re
import types
import uuid
from dataclasses import dataclass, field
from typing import (
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
    uid: str = ""

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
    def result(self) -> str:
        """The payload serialised for the model."""
        return json.dumps(self._result, default=str)


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
        return f"Available options: {super().result}"


def permission(fn: Callable | None = None):
    """Mark a function as needing the user's permission before it runs as a tool.

    Usable bare (``@permission``) or called (``@permission()``).  The mark is
    picked up by :meth:`Tool.from_callable`; whoever dispatches the tool
    enforces it via :meth:`Tool.permission_gate`.
    """

    def mark(f: Callable) -> Callable:
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
        return inspect.iscoroutinefunction(self.callback)

    def __call__(self, **kwargs) -> ToolResult:
        return self.execute(**kwargs)

    def execute(self, **kwargs) -> ToolResult:
        """Validate the arguments, run the callback, and normalise what it returns.

        This does not consult :attr:`requires_permission`; consent belongs to
        the caller, which checks :meth:`permission_gate` first.
        """
        if self.is_async:
            return asyncio.run(self.aexecute(**kwargs))
        callback = self._require_callback()
        return self._normalize(callback(**self._coerce_arguments(kwargs)))

    async def aexecute(self, **kwargs) -> ToolResult:
        """Like :meth:`execute`, awaiting async callbacks and running blocking ones in a thread."""
        callback = self._require_callback()
        arguments = self._coerce_arguments(kwargs)
        if self.is_async:
            return self._normalize(await callback(**arguments))
        return self._normalize(await asyncio.to_thread(callback, **arguments))

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
            payload = result[0] if isinstance(result[0], dict) else {"result": str(result[0])}
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
        missing = set(self.parameters.required or ()) - set(arguments)
        if missing:
            raise ToolError(f"Missing required tool arguments: {', '.join(sorted(missing))}")

        return {
            name: _cast(name, value, properties[name].type) if name in properties else value
            for name, value in arguments.items()
        }

    # --- construction ----------------------------------------------------

    @staticmethod
    def from_callable(name: str, fn: Callable, *, may_return_choice: Optional[bool] = None) -> "Tool":
        """Build a Tool from a function: schema from its signature, descriptions from its docstring."""
        doc = _parse_docstring(inspect.getdoc(fn))
        try:
            signature = inspect.signature(fn, eval_str=True)
        except (NameError, TypeError):
            signature = inspect.signature(fn)

        properties: Dict[str, Tool.Parameter] = {}
        required: List[str] = []
        allow_extra = False
        for param_name, param in signature.parameters.items():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                allow_extra = True
                continue
            if param_name == "self" or param.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            properties[param_name] = _parameter_for(param.annotation, doc.params.get(param_name))
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        if may_return_choice is None and _returns_choice(signature.return_annotation):
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
    def from_class(cls: Type) -> Callable[..., List["Tool"]]:
        """Class decorator turning a class into a factory of tools.

        Calling the decorated name builds an instance (forwarding the
        arguments) and returns one Tool per public method, bound to it.
        """

        def factory(*args, **kwargs) -> List["Tool"]:
            instance = cls(*args, **kwargs)
            return [
                Tool.from_callable(name, method)
                for name, method in inspect.getmembers(instance, predicate=inspect.ismethod)
                if not name.startswith("_")
            ]

        factory.__name__ = cls.__name__
        factory.__qualname__ = cls.__qualname__
        factory.__doc__ = cls.__doc__
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

        return cls(
            name=name,
            description=function_data.get("description") or "No description provided.",
            parameters=cls.Parameter.from_dict(parameters_data) if parameters_data else None,
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
        return str(value)
    return value


# --- type hints to JSON schema ----------------------------------------------


def _strip_optional(annotation: Any) -> Any:
    """``Optional[X]`` / ``X | None`` → ``X``; other unions resolve to their first member."""
    if get_origin(annotation) in _UNION_ORIGINS:
        members = [a for a in get_args(annotation) if a is not type(None)]
        return members[0] if members else str
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
_RST_FIELD_RE = re.compile(r"^:(param\s+\w+|returns?|instruction|raises\s+\w+|rtype|type\s+\w+):\s*(.*)$")
_GOOGLE_PARAM_RE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?:\s*(.*)$")


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
                target = rst.group(1).split()[1]
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
