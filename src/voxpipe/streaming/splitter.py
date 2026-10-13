"""Sentence splitting for streamed text, so speech can start before a reply is complete."""

import re
from typing import Generator, Iterable

_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "capt", "lt", "sgt", "cpl", "gen", "col", "maj",
    "sen", "rep", "gov", "pres", "vp",
    "vs", "etc", "dept", "est", "approx", "min", "max",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
})  # fmt: skip

# Sentence-ending punctuation, any closing quotes or brackets, then whitespace.
_BOUNDARY = re.compile(r"[.!?…]+[\"'”’)\]]*\s+")
# A boundary can straddle two chunks; this many trailing characters are rescanned to catch it.
_RESCAN_CHARS = 8


def _ends_sentence(buffer: str, match: re.Match) -> bool:
    """Whether a boundary match really ends a sentence.

    A period does not when it follows an abbreviation ("Dr."), sits inside an
    initialism ("e.g."), or numbers a list item ("1." opening a line).
    """
    if not match.group().startswith("."):
        return True
    line = buffer[: match.start()].rsplit("\n", 1)[-1].strip()
    token = (line.rsplit(None, 1)[-1:] or [""])[0].strip("\"'“‘([-").lower()
    if token in _ABBREVIATIONS or "." in token:
        return False
    return not (token.isdigit() and line == token)


def stream_splitter(text_stream: Iterable[str], min_len: int = 0) -> Generator[str, None, None]:
    """Regroup a stream of text chunks into sentences, yielding each as soon as it is complete.

    Nothing is yielded until ``min_len`` characters are buffered, which keeps a
    very short opening fragment from being spoken on its own.  Whatever is left
    when the stream ends is yielded as the last sentence.  Closing this
    generator closes ``text_stream`` too, so stopping playback stops generation.
    """
    buffer = ""
    scan_from = 0
    try:
        for chunk in text_stream:
            buffer += chunk
            if len(buffer) < min_len:
                continue
            while True:
                boundary = next(
                    (m for m in _BOUNDARY.finditer(buffer, scan_from) if _ends_sentence(buffer, m)), None
                )
                if boundary is None:
                    scan_from = max(0, len(buffer) - _RESCAN_CHARS)
                    break
                sentence = buffer[: boundary.end()].strip()
                buffer = buffer[boundary.end() :]
                scan_from = 0
                if sentence:
                    yield sentence
        if buffer.strip():
            yield buffer.strip()
    finally:
        close = getattr(text_stream, "close", None)
        if close:
            close()
