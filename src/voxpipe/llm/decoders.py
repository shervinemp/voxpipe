"""Stream decoders: turn a model's raw text stream into text chunks and tool calls."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, Optional, Tuple, Type, Union

from ..core.exceptions import ProviderError
from ..core.utils import get_logger
from .tools import MalformedToolCall, ToolCall

_log = get_logger(__name__)

StreamItem = Union[str, ToolCall, MalformedToolCall]

_GEMMA_STRING_DELIMITER = '<|"|>'
_GEMMA_CALL_RE = re.compile(r"call:([A-Za-z0-9_]+)(.*)", re.DOTALL)
_BARE_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)")


class StreamDecoder(ABC):
    """Strategy interface for interpreting and intercepting LLM streams."""

    @abstractmethod
    def __call__(self, stream: Iterable[StreamItem]) -> Iterator[StreamItem]: ...


class NativeDecoder(StreamDecoder):
    """Passes the stream through untouched, for providers that emit structured tool calls."""

    def __call__(self, stream: Iterable[StreamItem]) -> Iterator[StreamItem]:
        yield from stream


@dataclass(frozen=True)
class TagFormat:
    """A delimited span in model output and how to interpret what lies inside it.

    ``parse`` is "json" for ``{"name": ..., "arguments": {...}}`` bodies,
    "call:" for Gemma's ``call:name{key:<|"|>value<|"|>}`` syntax, or
    "thought" for reasoning that is dropped from the output.
    """

    open: str
    close: str
    parse: Literal["json", "call:", "thought"]


class GeneralDecoder(StreamDecoder):
    """Scans streamed text for tagged tool calls and thoughts in any of several formats.

    Text outside tags is yielded as it arrives, holding back only a trailing
    fragment that could be the start of an opening tag.  Thoughts are dropped.
    The first complete tool call is yielded and ends the stream, so the caller
    can run it; a call that is unparseable, oversized or never closed is
    reported as a MalformedToolCall.
    """

    DEFAULT_FORMATS: Tuple[TagFormat, ...] = (
        TagFormat("<|tool_call>", "<tool_call|>", "call:"),
        TagFormat("<tool_call>", "</tool_call>", "json"),
        TagFormat("<toolcall>", "</toolcall>", "json"),
        TagFormat("&lt;toolcall&gt;", "&lt;/toolcall&gt;", "json"),
        TagFormat("<|channel>thought", "<channel|>", "thought"),
        TagFormat("<think>", "</think>", "thought"),
    )
    MAX_CALL_CHARS = 10_000

    def __init__(self, formats: Optional[Iterable[Union[TagFormat, Mapping[str, Any]]]] = None):
        chosen = self.DEFAULT_FORMATS if formats is None else formats
        self.formats: List[TagFormat] = [f if isinstance(f, TagFormat) else TagFormat(**f) for f in chosen]

    def __call__(self, stream: Iterable[StreamItem]) -> Iterator[StreamItem]:
        buffer = ""
        active: Optional[TagFormat] = None

        for chunk in stream:
            if not isinstance(chunk, str):
                # The provider already produced a structured call; nothing left to decode.
                yield chunk
                return
            buffer += chunk

            while True:
                if active is None:
                    active, start = self._find_opener(buffer)
                    if active is None:
                        held = self._partial_opener_length(buffer)
                        if len(buffer) > held:
                            yield buffer[: len(buffer) - held]
                        buffer = buffer[len(buffer) - held :]
                        break
                    if start:
                        yield buffer[:start]
                    buffer = buffer[start + len(active.open) :]
                    continue

                end = buffer.find(active.close)
                if active.parse == "thought":
                    if end == -1:
                        # Keep only what could still complete the closing tag.
                        buffer = buffer[-(len(active.close) - 1) :] if len(active.close) > 1 else ""
                        break
                    buffer = buffer[end + len(active.close) :]
                    active = None
                    continue
                if end != -1:
                    yield _parse_call(active, buffer[:end].strip())
                    return
                if len(buffer) > self.MAX_CALL_CHARS:
                    _log.warning("Tool call exceeded %d characters; discarding.", self.MAX_CALL_CHARS)
                    yield MalformedToolCall(raw=buffer)
                    return
                break

        if active is None:
            if buffer:
                yield buffer
        elif active.parse != "thought":
            _log.warning("Stream ended inside an unterminated tool call.")
            yield MalformedToolCall(raw=buffer)

    def _find_opener(self, buffer: str) -> Tuple[Optional[TagFormat], int]:
        """Earliest opening tag in the buffer; the longer tag wins when two start together."""
        best: Tuple[Optional[TagFormat], int] = (None, -1)
        for fmt in self.formats:
            index = buffer.find(fmt.open)
            if index == -1:
                continue
            current, at = best
            if current is None or index < at or (index == at and len(fmt.open) > len(current.open)):
                best = (fmt, index)
        return best

    def _partial_opener_length(self, buffer: str) -> int:
        """Length of the longest buffer suffix that is a proper prefix of some opening tag."""
        longest = max((len(f.open) for f in self.formats), default=1) - 1
        for size in range(min(longest, len(buffer)), 0, -1):
            tail = buffer[-size:]
            if any(f.open.startswith(tail) for f in self.formats):
                return size
        return 0


class GemmaE2BDecoder(GeneralDecoder):
    """Gemma 4's ``<|tool_call>call:name{...}<tool_call|>`` syntax and thought channel."""

    DEFAULT_FORMATS = (
        TagFormat("<|tool_call>", "<tool_call|>", "call:"),
        TagFormat("<|channel>thought", "<channel|>", "thought"),
    )


class LegacyXMLDecoder(GeneralDecoder):
    """JSON tool calls in ``<toolcall>`` / ``<tool_call>`` tags, as used by Qwen and Nemotron."""

    DEFAULT_FORMATS = (
        TagFormat("<toolcall>", "</toolcall>", "json"),
        TagFormat("<tool_call>", "</tool_call>", "json"),
        TagFormat("<think>", "</think>", "thought"),
    )


def _parse_call(fmt: TagFormat, body: str) -> Union[ToolCall, MalformedToolCall]:
    try:
        if fmt.parse == "call:":
            match = _GEMMA_CALL_RE.match(body)
            if not match:
                raise ValueError("expected call:<name>{...}")
            name, arguments = match.group(1), _parse_gemma_arguments(match.group(2).strip())
        else:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("tool call must be a JSON object")
            name, arguments = payload.get("name") or payload.get("function"), payload.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise ValueError("tool call has no name")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
    except ValueError as e:  # json.JSONDecodeError is a ValueError
        _log.warning("Failed to parse tool call (%s): %s", e, body[:200])
        return MalformedToolCall(raw=body)
    return ToolCall(name=name, arguments=arguments)


def _parse_gemma_arguments(text: str) -> dict:
    """Parse Gemma's argument object: bare keys, strings delimited by ``<|"|>``."""
    if not text:
        return {}
    segments = text.split(_GEMMA_STRING_DELIMITER)
    if len(segments) % 2 == 0:
        raise ValueError("unbalanced string delimiter")
    # Even segments are structure, whose bare keys need quoting; odd ones are string literals.
    rebuilt = "".join(
        _json_string(segment) if index % 2 else _BARE_KEY_RE.sub(r'\1"\2"\3', segment)
        for index, segment in enumerate(segments)
    )
    return json.loads(rebuilt)


def _json_string(literal: str) -> str:
    """Encode a model-written string literal, honouring JSON escapes when it uses them consistently."""
    try:
        value = json.loads(f'"{literal}"')
    except json.JSONDecodeError:
        value = literal
    return json.dumps(value)


DECODERS: Dict[str, Type[StreamDecoder]] = {
    "native": NativeDecoder,
    "general": GeneralDecoder,
    "gemma_e2b": GemmaE2BDecoder,
    "legacy_xml": LegacyXMLDecoder,
}


def get_decoder(name: str) -> StreamDecoder:
    """Instantiate the decoder registered under ``name``."""
    try:
        return DECODERS[name]()
    except KeyError:
        raise ProviderError(f"Unknown decoder {name!r}. Expected one of: {', '.join(sorted(DECODERS))}.") from None
