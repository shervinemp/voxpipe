"""The voice pipeline: microphone -> ASR -> LLM session -> sentence splitter -> TTS."""

import queue
import re
import sqlite3
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, Optional, Tuple

from ..core.config import get_config
from ..core.exceptions import ASRError, LLMError, StorageError, VoxpipeError
from ..core.models import AppConfig
from ..core.utils import get_logger
from ..llm.context import CompositeStrategy, MaxTurnsStrategy, MemoryHandler, TokenBudgetStrategy
from ..llm.model import LLM, LLMProviders
from ..llm.session import Session, TextDelta, ToolCallEvent, ToolResultEvent
from ..llm.tools import Tool
from ..storage import SQLiteStorage
from ..streaming.splitter import stream_splitter
from .events import EventEmitter
from .gate import qualify_transcript

if TYPE_CHECKING:
    from .hotkeys import HotkeyDispatcher

_STOP = object()
_POLL_SECONDS = 0.5  # a blocking queue read cannot be interrupted by Ctrl+C on Windows
_APOLOGY = "Sorry, I ran into a problem answering that."
# What the model is told when the session arrives with no instructions or rules of its own.
_INSTRUCTIONS = "You are a voice-controlled assistant. Respond conversationally."
_RULES = (
    "If the user's message seems incomplete or cut off, ask what they meant.",
    "Do not describe your actions or announce tool usage.",
    "Respond naturally, as if you already know the answer.",
)
# How a command's pattern is turned into a test on the (normalised) utterance.
_COMMAND_MATCHERS: Dict[str, Callable[[str], Callable[[str], Any]]] = {
    "exact": lambda pattern: pattern.__eq__,
    "prefix": lambda pattern: lambda text: text.startswith(pattern),
    "regex": lambda pattern: re.compile(pattern).match,
}


class Pipeline:
    """Listens, thinks and speaks.

    Speech recognition runs on its own thread, so the user can talk over a
    reply: generation is cancelled and playback cut off the moment they start.
    The components are passed in ready-made; :meth:`from_config` builds them
    from the configuration.  ``asr`` and ``tts`` may be None, which leaves the
    pipeline usable for text in (:meth:`handle_transcript`) or text out.

    Events (``pipeline.events.on(name, handler)``): ``asr:transcript``,
    ``asr:utterance``, ``pipeline:state`` ("think" / "speak" / "idle"),
    ``tts:start``, ``tts:utterance``, ``tool:call``, ``tool:result``,
    ``session:reset``, ``vad:level`` and ``pipeline:error``.
    """

    def __init__(
        self,
        session: Session,
        *,
        asr: Any = None,
        tts: Any = None,
        hotkeys: Optional["HotkeyDispatcher"] = None,
        push_to_talk: Optional[str] = None,
        press_to_reset: Optional[str] = None,
        memory: Optional[SQLiteStorage] = None,
    ):
        """``memory`` is the conversation-history storage the pipeline should close on shutdown."""
        self.logger = get_logger(__name__)
        self.events = EventEmitter()
        self.session = session
        self.asr = asr
        self.tts = tts
        self.memory = memory

        self._hotkeys = hotkeys
        self._push_to_talk: Optional[str] = None
        self._press_to_reset: Optional[str] = None
        self._commands: Dict[str, Tuple[Callable[[str], Any], Callable[[], None]]] = {}
        self._transcripts: "queue.Queue[Any]" = queue.Queue()
        self._interrupt = threading.Event()
        self._responding = False
        self._response_parts: list[str] = []
        self._interrupted_at: Optional[str] = None
        self._closed = False

        if asr is not None:
            asr.vad.on_speech_onset = self._on_speech_onset
            asr.vad.on_audio_level = lambda rms, probability: self.events.emit("vad:level", rms, probability)
        self.push_to_talk = push_to_talk
        self.press_to_reset = press_to_reset
        if not session.instructions and not session.rules:
            session.instructions = _INSTRUCTIONS
            session.rules = list(_RULES)

    # --- construction from configuration ---------------------------------

    @classmethod
    def from_config(
        cls,
        config: Optional[AppConfig] = None,
        *,
        session: Optional[Session] = None,
        push_to_talk: Optional[str] = None,
        press_to_reset: Optional[str] = None,
    ) -> "Pipeline":
        """Load the configured models and assemble a pipeline around them.

        A recogniser or synthesiser that fails to load is logged and left out;
        a failing LLM is fatal.  Pass ``session`` to bring your own (its context
        handling is then yours too).  Hotkey arguments override the configuration.
        """
        # Imported here: these pull in the audio stack, which a pipeline built by hand may not need.
        from ..asr.model import ASRProviders
        from ..tts.model import TTSProviders

        config = config or get_config()
        logger = get_logger(__name__)

        asr = tts = memory = own_session = None
        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="voxpipe-load") as pool:
                asr_future = pool.submit(ASRProviders.create, config=config)
                tts_future = pool.submit(TTSProviders.create, config=config)
                try:
                    if session is None:
                        # The LLM loads on this thread while the audio models load on the pool.
                        llm = LLMProviders.create(config.llm.backend, config.llm.model, config)
                        memory = cls._open_memory(config)
                        session = own_session = Session(
                            llm=llm,
                            max_tool_iterations=config.llm.max_tool_iterations,
                            tool_timeout=config.llm.tool_timeout,
                            context_handler=cls._context_handler(llm, config, memory),
                        )
                finally:
                    asr = cls._optional("ASR", asr_future, logger)
                    tts = cls._optional("TTS", tts_future, logger)

            hotkeys = config.hotkeys
            return cls(
                session,
                asr=asr,
                tts=tts,
                push_to_talk=(push_to_talk or hotkeys.push_to_talk) if hotkeys.enable else None,
                press_to_reset=(press_to_reset or hotkeys.press_to_reset) if hotkeys.enable else None,
                memory=memory,
            )
        except BaseException:
            # Don't leave a microphone stream or worker threads behind a pipeline that never came to be.
            if asr is not None:
                asr.stop()
            if tts is not None:
                tts.stop()
            if own_session is not None:
                own_session.close()
            if memory is not None:
                memory.close()
            raise

    @staticmethod
    def _optional(name: str, future: Future, logger) -> Any:
        error = future.exception()
        if error is not None:
            logger.error("%s unavailable: %s", name, error)
            return None
        return future.result()

    @staticmethod
    def _open_memory(config: AppConfig) -> Optional[SQLiteStorage]:
        history = config.conversation_history
        if not history.enabled:
            return None
        try:
            return SQLiteStorage(history.db_path, max_entries=history.max_entries, ttl_days=history.ttl_days)
        except (StorageError, OSError, sqlite3.Error) as e:
            get_logger(__name__).warning("Conversation memory unavailable: %s", e)
            return None

    @staticmethod
    def _context_handler(llm: LLM, config: AppConfig, memory: Optional[SQLiteStorage]) -> MemoryHandler:
        """Keep the prompt within the model's context, and wire in long-term memory when enabled."""
        strategy = CompositeStrategy(
            MaxTurnsStrategy(config.llm.max_turns),
            TokenBudgetStrategy(llm.count_tokens, budget=llm.n_ctx - llm.max_tokens),
        )
        return MemoryHandler(strategy, memory, recall_top_k=config.conversation_history.top_k)

    # --- tools and commands ----------------------------------------------

    def register_tools(self, *tools: Tool | list[Tool]) -> None:
        """Make tools available to the model; lists of tools are flattened."""
        for entry in tools:
            for tool in entry if isinstance(entry, list) else [entry]:
                if tool.callback is None:
                    self.logger.warning("Tool '%s' has no callback; calls to it will report an error.", tool.name)
                self.session.tools.add(tool)

    def register_command(self, pattern: str, handler: Callable[[], None], mode: str = "exact") -> None:
        """Run ``handler`` instead of the LLM when an utterance matches.

        Utterances are compared lowercased and without trailing punctuation.
        ``mode`` is "exact", "prefix", or "regex" (matched from the start).
        """
        try:
            matcher = _COMMAND_MATCHERS[mode](pattern)
        except KeyError:
            raise ValueError(f"Unknown command mode {mode!r}; expected one of {sorted(_COMMAND_MATCHERS)}.") from None
        self._commands[pattern] = (matcher, handler)

    def unregister_command(self, pattern: str) -> None:
        self._commands.pop(pattern, None)

    def _run_command(self, text: str) -> bool:
        cleaned = text.strip().lower().rstrip(".!?,")
        for matches, handler in list(self._commands.values()):
            if matches(cleaned):
                handler()
                return True
        return False

    # --- hotkeys ---------------------------------------------------------

    @property
    def hotkeys(self) -> "HotkeyDispatcher":
        """The hotkey dispatcher, created on first use so that hotkey-less setups need no keyboard hook."""
        if self._hotkeys is None:
            try:
                from .hotkeys import HotkeyDispatcher

                self._hotkeys = HotkeyDispatcher()
            except ImportError as e:
                raise VoxpipeError(f"Hotkeys are configured but keyboard access is unavailable: {e}") from e
        return self._hotkeys

    @property
    def push_to_talk(self) -> Optional[str]:
        return self._push_to_talk

    @push_to_talk.setter
    def push_to_talk(self, hotkey: Optional[str]) -> None:
        """Listen only while ``hotkey`` is held; None returns to always-on listening."""
        if self._push_to_talk:
            self.hotkeys.unregister(self._push_to_talk)
        self._push_to_talk = hotkey or None
        if self.asr is None:
            return
        if self._push_to_talk:
            self.asr.mute()
            self.hotkeys.register(self._push_to_talk, self._hold_to_talk)
        else:
            self.asr.unmute()

    @contextmanager
    def _hold_to_talk(self) -> Iterator[None]:
        self.asr.unmute()
        try:
            yield
        finally:
            self.asr.mute()

    @property
    def press_to_reset(self) -> Optional[str]:
        return self._press_to_reset

    @press_to_reset.setter
    def press_to_reset(self, hotkey: Optional[str]) -> None:
        if self._press_to_reset:
            self.hotkeys.unregister(self._press_to_reset)
        self._press_to_reset = hotkey or None
        if self._press_to_reset:
            self.hotkeys.register(self._press_to_reset, self.reset)

    # --- control ---------------------------------------------------------

    @property
    def status(self) -> Dict[str, str]:
        if self.asr is None:
            asr = "unavailable"
        else:
            asr = "muted" if self.asr.is_muted else "listening"
        if self.tts is None:
            tts = "unavailable"
        else:
            tts = "speaking" if self.tts.is_speaking else "idle"
        return {"asr": asr, "llm": "generating" if self._responding else "idle", "tts": tts}

    def interrupt(self) -> None:
        """Stop the reply in progress: cancel generation and cut off speech. Safe from any thread."""
        self._interrupt.set()
        self.session.cancel()
        if self.tts is not None:
            self.tts.interrupt()

    def reset(self) -> None:
        """Forget the conversation so far, stopping any reply in progress. Safe from any thread."""
        self.events.emit("session:reset")
        self.interrupt()
        self.session.reset()

    def stop(self) -> None:
        """Ask :meth:`run` to return. Safe from any thread."""
        self._transcripts.put(_STOP)
        self.interrupt()

    def _on_speech_onset(self) -> None:
        """The user started talking (called on the ASR thread). If we are mid-reply, yield the floor."""
        speaking = self._responding or (self.tts is not None and self.tts.is_speaking)
        if not speaking:
            return
        if self._response_parts:
            self._interrupted_at = self._response_parts[-1]
        self.interrupt()

    # --- the conversation loop -------------------------------------------

    def run(self) -> None:
        """Listen and respond until :meth:`stop` is called or the microphone gives out. Blocks."""
        if self.asr is None:
            raise ASRError("Cannot run the pipeline without a speech recogniser.")
        try:
            self.asr.start()
            if self.tts is not None:
                self.tts.start()
            if self._hotkeys is not None:
                self._hotkeys.start()
            threading.Thread(target=self._listen, name="voxpipe-asr", daemon=True).start()

            while True:
                try:
                    transcript = self._transcripts.get(timeout=_POLL_SECONDS)
                except queue.Empty:
                    continue
                if transcript is _STOP:
                    return
                try:
                    self.handle_transcript(transcript)
                except Exception as e:  # noqa: BLE001 - one bad turn must not end the conversation
                    self.events.emit("pipeline:error", e)
                    self.logger.error("Error handling %r: %s", transcript, e, exc_info=not isinstance(e, VoxpipeError))
        finally:
            self.close()

    def _listen(self) -> None:
        """ASR thread: recognise utterances and queue their transcripts for the conversation loop."""
        try:
            for transcript in self.asr:
                self._transcripts.put(transcript)
        except Exception as e:  # noqa: BLE001 - report and shut the pipeline down cleanly
            self.events.emit("pipeline:error", e)
            self.logger.error("Speech recognition stopped: %s", e, exc_info=True)
        finally:
            self._transcripts.put(_STOP)

    def handle_transcript(self, transcription: str) -> None:
        """Process one recognised utterance: run a matching command, or have the LLM answer aloud."""
        self.events.emit("asr:transcript", transcription)
        text, annotation = qualify_transcript(transcription)
        if text is None:
            return
        self.events.emit("asr:utterance", text, annotation=annotation)

        interrupted_at, self._interrupted_at = self._interrupted_at, None
        if self._run_command(text):
            return

        notes = [annotation] if annotation else []
        if interrupted_at:
            notes.append(f'(User interrupted after: "{interrupted_at}". Continue naturally.)')
        self._respond(text, "\n".join(notes) or None)

    def _respond(self, text: str, context: Optional[str]) -> None:
        self._response_parts = []
        self._interrupt.clear()
        self._responding = True
        self.events.emit("pipeline:state", "think")
        if self.tts is not None:
            self.tts.interrupt()  # a new reply never queues behind the tail of the previous one

        sentences = stream_splitter(self._reply(text, context), min_len=8)
        try:
            for sentence in sentences:
                if self._interrupt.is_set():
                    break
                self._say(sentence)
        except LLMError as e:
            self.events.emit("pipeline:error", e)
            self.logger.error("The LLM failed to answer: %s", e)
            self._say(_APOLOGY)
        finally:
            sentences.close()  # ends the turn in the session as well
            self._responding = False
            self.events.emit("pipeline:state", "idle")

    def _reply(self, text: str, context: Optional[str]) -> Generator[str, None, None]:
        """The reply's text as it is generated; tool activity is re-emitted as pipeline events."""
        events = self.session.events(text, context=context)
        try:
            for event in events:
                if isinstance(event, TextDelta):
                    yield event.text
                elif isinstance(event, ToolCallEvent):
                    self.events.emit("tool:call", event.call)
                elif isinstance(event, ToolResultEvent):
                    self.events.emit("tool:result", event.call, event.result, event.error)
        finally:
            events.close()

    def _say(self, sentence: str) -> None:
        sentence = sentence.strip()
        if not sentence:
            return
        self._response_parts.append(sentence)
        self.events.emit("pipeline:state", "speak")
        self.events.emit("tts:start", sentence)
        if self.tts is not None:
            self.tts(sentence)
        else:
            self.logger.info("[TTS unavailable] %s", sentence)
        self.events.emit("tts:utterance", sentence)

    # --- shutdown --------------------------------------------------------

    def close(self) -> None:
        """Release every component. Called by :meth:`run` on exit; safe to call again."""
        if self._closed:
            return
        self._closed = True
        steps = [("hotkeys", self._hotkeys), ("ASR", self.asr), ("TTS", self.tts)]
        for name, component in steps:
            if component is not None:
                self._release(name, component.stop)
        self._release("session", self.session.close)
        if self.memory is not None:
            self._release("memory", self.memory.close)
        self.events.close()

    def _release(self, name: str, release: Callable[[], None]) -> None:
        try:
            release()
        except Exception:  # noqa: BLE001 - keep shutting the rest down
            self.logger.warning("Error while shutting down %s.", name, exc_info=True)

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
