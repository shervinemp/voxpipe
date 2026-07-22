from abc import ABC, abstractmethod
import json
import re
from typing import Generator, Iterator

from ..core.utils import get_logger
from .tools import ToolCall

_log = get_logger(__name__)


class StreamDecoder(ABC):
    """Strategy interface for interpreting and intercepting LLM streams."""
    @abstractmethod
    def __call__(self, stream: Iterator[str | dict | ToolCall]) -> Generator[str | dict | ToolCall, None, None]: ...


class NativeDecoder(StreamDecoder):
    """Yields stream exactly as it arrives (used for OpenAI, Gemini, and native tools)."""
    def __call__(self, stream: Iterator[str | dict | ToolCall]) -> Generator[str | dict | ToolCall, None, None]:
        yield from stream


class LegacyXMLDecoder(StreamDecoder):
    """Legacy parser for standard <toolcall>...</toolcall> used by Qwen/Nemotron."""
    def __call__(self, stream: Iterator[str | dict | ToolCall]) -> Generator[str | dict | ToolCall, None, None]:
        buffer, in_tool = "", False
        for chunk in stream:
            if isinstance(chunk, (dict, ToolCall)):
                yield chunk
                return
            buffer += chunk

            while buffer:
                if in_tool:
                    if "</toolcall>" in buffer:
                        try:
                            tool_body = buffer.split("</toolcall>")[0]
                            tool_dict = json.loads(tool_body.strip())
                            yield ToolCall(
                                name=tool_dict.get("name") or tool_dict.get("function"),
                                arguments=tool_dict.get("arguments", {})
                            )
                        except Exception:
                            _log.warning(
                                "Failed to parse toolcall: %s", tool_body[:200]
                            )
                            yield ToolCall(name="_parse_error", arguments={"raw": tool_body})
                        return  # Halt stream to execute tool
                    else:
                        if len(buffer) > 10_000:
                            _log.warning(
                                "Toolcall buffer exceeded 10K chars, discarding"
                            )
                            yield buffer
                            buffer = ""
                        break  # Wait for more chunks
                else:
                    if "<toolcall>" in buffer:
                        pre = buffer.split("<toolcall>")[0]
                        if pre:
                            yield pre
                        in_tool = True
                        buffer = buffer.split("<toolcall>", 1)[1]
                        continue  # Re-evaluate buffer
                    else:
                        match = re.search(r'<t(?:o(?:o(?:l(?:c(?:a(?:l(?:l)?)?)?)?)?)?)?$', buffer)
                        if match:
                            safe_idx = match.start()
                            if safe_idx > 0:
                                yield buffer[:safe_idx]
                                buffer = buffer[safe_idx:]
                            break  # Wait for more chunks to complete the tag
                        else:
                            yield buffer
                            buffer = ""
                            break

        if buffer and not in_tool:
            yield buffer


class GemmaE2BDecoder(StreamDecoder):
    """Clean, chunk-based sliding window parser for Gemma 4's unique custom syntax."""
    def __call__(self, stream: Iterator[str | dict | ToolCall]) -> Generator[str | dict | ToolCall, None, None]:
        buffer = ""
        in_tool, in_thought = False, False

        for chunk in stream:
            if isinstance(chunk, (dict, ToolCall)):
                yield chunk
                return

            buffer += chunk

            while buffer:
                # 1. Handle Tool Execution
                if in_tool:
                    if "<tool_call|>" in buffer:
                        body = buffer.split("<tool_call|>")[0].replace("<|tool_call>", "").strip()
                        if body.startswith("call:"):
                            match = re.match(r"call:([a-zA-Z0-9_]+)(.*)", body, re.DOTALL)
                            if match:
                                name, args = match.group(1).strip(), match.group(2).strip()
                                args = args.replace('<|"|>', '"')
                                args = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', args)
                                try:
                                    yield ToolCall(name=name, arguments=json.loads(args))
                                except json.JSONDecodeError:
                                    _log.warning(
                                        "Failed to parse gemma toolcall args: %s", args[:200]
                                    )
                                    yield ToolCall(name="_parse_error", arguments={"raw": args})
                        return  # CRITICAL: Halt stream to allow RAG pipeline to trigger
                    break  # Wait for more chunks

                # 2. Filter Thoughts
                if in_thought:
                    if "<channel|>" in buffer:
                        in_thought = False
                        buffer = buffer.split("<channel|>", 1)[1]
                        continue  # Re-evaluate buffer
                    break  # Wait for more chunks

                # 3. Detect Openers
                if "<|tool_call>" in buffer:
                    in_tool = True
                    pre = buffer.split("<|tool_call>")[0]
                    if pre:
                        yield pre
                    buffer = buffer.split("<|tool_call>", 1)[1]
                    continue  # Re-evaluate buffer

                if "<|channel>thought" in buffer:
                    in_thought = True
                    pre = buffer.split("<|channel>thought")[0]
                    if pre:
                        yield pre
                    buffer = buffer.split("<|channel>thought", 1)[1]
                    continue  # Re-evaluate buffer

                # 4. Safe Yield (Wait for partial tags to resolve)
                safe_idx = max(buffer.rfind("<"), buffer.rfind("&"))
                if safe_idx != -1 and (len(buffer) - safe_idx) < 15:
                    if safe_idx > 0:
                        yield buffer[:safe_idx]
                        buffer = buffer[safe_idx:]
                    break  # Wait for more chunks
                else:
                    if buffer:
                        yield buffer
                    buffer = ""
                    break  # Finished processing this chunk

        if buffer and not in_tool and not in_thought:
            yield buffer


class GeneralDecoder(StreamDecoder):
    """Matches tag formats by opener priority — auto-sorted by length (longest first).

    Handles Gemma ``<|tool_call|>``, standard ``<toolcall>``, HTML-escaped
    ``&lt;toolcall&gt;``, and thought ``<|channel>thought`` in one decoder.
    Longer openers are checked first so ``<|tool_call|>`` beats ``<toolcall>``.
    """

    def __init__(self, formats: list[dict] | None = None):
        raw = formats or [
            {"open": "<|tool_call>", "close": "<tool_call|>", "parse": "call:"},
            {"open": "<toolcall>", "close": "</toolcall>", "parse": "json"},
            {"open": "&lt;toolcall&gt;", "close": "&lt;/toolcall&gt;", "parse": "json"},
            {"open": "<|channel>thought", "close": "<channel|>", "parse": "thought"},
        ]
        self.formats = sorted(raw, key=lambda f: -len(f["open"]))

    def __call__(self, stream: Iterator[str | dict | ToolCall]) -> Generator[str | dict | ToolCall, None, None]:
        buffer = ""
        active = None
        body = ""

        for chunk in stream:
            if isinstance(chunk, (dict, ToolCall)):
                yield chunk
                return

            buffer += chunk

            while buffer:
                # If we're inside a tag, look for the closing marker
                if active is not None:
                    if active["parse"] == "thought":
                        idx = buffer.find(active["close"])
                        if idx != -1:
                            buffer = buffer[idx + len(active["close"]):]
                            active = None
                            continue
                        break

                    idx = buffer.find(active["close"])
                    if idx != -1:
                        body += buffer[:idx]
                        yield from self._emit(body.strip(), active["parse"])
                        return
                    # Body accumulates until close arrives
                    body += buffer
                    buffer = ""
                    break

                # Not in a tag — search for any opener (priority order)
                matched = None
                for fmt in self.formats:
                    idx = buffer.find(fmt["open"])
                    if idx != -1:
                        matched = fmt
                        pre = buffer[:idx]
                        if pre:
                            yield pre
                        buffer = buffer[idx + len(fmt["open"]):]
                        if fmt["parse"] == "thought":
                            active = fmt  # enter thought, drop body
                            body = ""
                        else:
                            active = fmt
                            body = ""
                        break

                if matched:
                    continue

                # Safety: hold back text ending with partial tag characters
                safe = max(buffer.rfind("<"), buffer.rfind("&"))
                if safe != -1 and len(buffer) - safe < 15:
                    if safe > 0:
                        yield buffer[:safe]
                        buffer = buffer[safe:]
                    break
                yield buffer
                buffer = ""

        if buffer and active is None:
            yield buffer

    def _emit(self, body: str, fmt: str):
        if fmt == "call:":
            if body.startswith("call:"):
                m = re.match(r"call:([a-zA-Z0-9_]+)(.*)", body, re.DOTALL)
                if m:
                    name = m.group(1).strip()
                    args = m.group(2).strip()
                    args = args.replace('<|"|>', '"')
                    args = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', args)
                    try:
                        yield ToolCall(name=name, arguments=json.loads(args))
                        return
                    except json.JSONDecodeError:
                        _log.warning("Failed to parse call-format args: %s", args[:200])
            yield ToolCall(name="_parse_error", arguments={"raw": body})
            return
        if fmt == "json":
            try:
                d = json.loads(body)
                yield ToolCall(
                    name=d.get("name") or d.get("function"),
                    arguments=d.get("arguments", {}),
                )
            except Exception as e:
                _log.warning("Failed to parse tool call: %s — %s", body[:200], e)
                yield ToolCall(name="_parse_error", arguments={"raw": body})
            return
