from abc import ABC, abstractmethod
import re
from typing import Any, Generator, Iterable


class ConsumerProducer(ABC):
    __allow_consume: bool = True
    __allow_passthrough: bool = False
    __passthrough: Any

    def __call__(self, value):
        if self.__allow_consume:
            self._consume(value)
        elif self.__allow_passthrough:
            self._consume(self.__passthrough)

    def __iter__(self):
        yield from self._produce()

    def enable(self):
        self.__allow_consume = True
        self.__allow_passthrough = False

    def disable(self):
        self.__allow_consume = False
        self.__allow_passthrough = False

    def disable_w_passthrough(self, value: Any = None):
        self.__allow_consume = False
        self.__allow_passthrough = True
        self.__passthrough = value

    @abstractmethod
    def _consume(self, value): ...

    @abstractmethod
    def _produce(self) -> Generator: ...


_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "capt", "lt", "sgt", "cpl", "gen", "col", "maj",
    "sen", "rep", "gov", "pres", "vp",
    "vs", "etc", "dept", "est", "approx", "min", "max",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
})


def stream_splitter(
    text_stream: Iterable[str], min_len: int = 0
) -> Generator[str, None, None]:
    sentences = re.compile(r"[.!?][\"']*\s+")
    buffer = ""
    for chunk in text_stream:
        buffer += chunk
        if len(buffer) >= min_len:
            search_start = max(0, len(buffer) - 100)
            search_zone = buffer[search_start:]
            for match in re.finditer(sentences, search_zone):
                # Check if preceding word is an abbreviation (case-insensitive)
                end = search_start + match.start()
                word_start = buffer.rfind(" ", 0, end)
                if word_start == -1:
                    word_start = 0
                else:
                    word_start += 1
                word = buffer[word_start:end].strip(' "\'"-').lower()
                if word in _ABBREVIATIONS:
                    continue
                abs_end = search_start + match.end()
                sentence = buffer[:end + 1]
                buffer = buffer[abs_end:]
                yield sentence.strip()
                break
    else:
        if s := buffer.strip():
            yield s
