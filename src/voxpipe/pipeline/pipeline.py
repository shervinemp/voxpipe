"""The voice pipeline: microphone -> ASR -> LLM session -> sentence splitter -> TTS."""

import queue
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple

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
from .transcript import qualify_transcript
from .topics import ConversationGate, TopicTracker, create_addressee_gate, create_tracker

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


def _normalise(text: str) -> str:
    """The form utterances and command patterns are compared in: lower case, no trailing punctuation."""
    return text.strip().lower().rstrip(".!?,")


def _combine_context(*parts: Optional[str]) -> Optional[str]:
    """Join the notes shown to the model ahead of the user's message."""
    present = [part for part in parts if part]
    return "\n".join(present) if present else None


def _exact(pattern: str) -> Callable[[str], Any]:
    return _normalise(pattern).__eq__


def _prefix(pattern: str) -> Callable[[str], Any]:
    prefix = _normalise(pattern)
    return lambda text: text.startswith(prefix)


def _regex(pattern: str) -> Callable[[str], Any]:
    return re.compile(pattern, re.IGNORECASE).match


# How a command's pattern is turned into a test on the normalised utterance.
_COMMAND_MATCHERS: Dict[str, Callable[[str], Callable[[str], Any]]] = {
    "exact": _exact,
    "prefix": _prefix,
    "regex": _regex,
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
    ``session:reset``, ``vad:level``, ``gate:decided``, ``topic:routed`` and
    ``pipeline:error``.  ``gate:decided`` carries the utterance and its
    :class:`~voxpipe.pipeline.topics.GateDecision`; ``topic:routed`` is also
    emitted when the gate is a
    :class:`~voxpipe.pipeline.topics.TopicTracker`, with its
    :class:`~voxpipe.pipeline.topics.TopicDecision`.  The ``tts:`` events mark
    a sentence being handed to the synthesiser and queued, and
    ``pipeline:state`` follows generation: speech may still be playing after
    "idle" (:attr:`status` tells).
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
        topics: Optional[TopicTracker] = None,
        gate: Optional[ConversationGate] = None,
    ):
        """``memory`` is the conversation-history storage the pipeline should close on shutdown.

        ``gate`` is any :class:`~voxpipe.pipeline.topics.ConversationGate` and
        decides which recognised utterances belong to the ongoing conversation.
        ``topics`` is kept as a convenience alias for the built-in
        :class:`~voxpipe.pipeline.topics.TopicTracker`; the pipeline does not
        depend on that implementation.
        """
        self.logger = get_logger(__name__)
        self.events = EventEmitter()
        self.session = session
        self.asr = asr
        self.tts = tts
        self.memory = memory
        self.gate: Optional[ConversationGate] = gate if gate is not None else topics

        self._hotkeys = hotkeys
        self._push_to_talk: Optional[str] = None
        self._press_to_reset: Optional[str] = None
        self._commands: Dict[str, Tuple[Callable[[str], Any], Callable[[], None]]] = {}
        self._transcripts: "queue.Queue[Any]" = queue.Queue()
        self._speech_event = threading.Event()  # set whenever another utterance is queued
        self._interrupt = threading.Event()
        self._stopping = threading.Event()
        self._running = False
        self._listen_error: Optional[BaseException] = None
        self._responding = False
        self._response_parts: list[str] = []
        self._interrupted_at: Optional[str] = None
        self._onsets = 0  # times the user has started speaking; only the ASR thread writes it
        self._baseline = 0  # the onset count the reply in progress belongs to
        self._speech_run = 0  # the TTS generation the reply in progress speaks in
        self._held: List[Tuple[str, Optional[str]]] = []  # utterances (with context) waiting for an answer
        self._closed = False

        if asr is not None:
            asr.vad.on_speech_onset = self._on_speech_onset
            asr.vad.on_audio_level = lambda rms, probability: self.events.emit("vad:level", rms, probability)
        self.push_to_talk = push_to_talk
        self.press_to_reset = press_to_reset
        if not session.instructions and not session.rules:
            session.instructions = _INSTRUCTIONS
            session.rules = list(_RULES)

    @property
    def topics(self) -> Optional[TopicTracker]:
        """The built-in topic tracker when the configured gate is one."""
        return self.gate if isinstance(self.gate, TopicTracker) else None

    @topics.setter
    def topics(self, tracker: Optional[TopicTracker]) -> None:
        self.gate = tracker

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
                # An argument always wins; ``hotkeys.enable: false`` only switches the configured ones off.
                push_to_talk=push_to_talk or (hotkeys.push_to_talk if hotkeys.enable else None),
                press_to_reset=press_to_reset or (hotkeys.press_to_reset if hotkeys.enable else None),
                memory=memory,
                gate=cls._gate(config),
            )
        except BaseException:
            # Don't leave a microphone stream or worker threads behind a pipeline that never came to be.
            for component, release in ((asr, "stop"), (tts, "stop"), (own_session, "close"), (memory, "close")):
                if component is None:
                    continue
                try:
                    getattr(component, release)()
                except Exception:  # noqa: BLE001 - keep releasing the rest; the original error is what matters
                    logger.warning("Error while releasing %s after a failed start.", component, exc_info=True)
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
        except (StorageError, OSError) as e:
            get_logger(__name__).warning("Conversation memory unavailable: %s", e)
            return None

    @staticmethod
    def _gate(config: AppConfig) -> Optional[ConversationGate]:
        """Build the configured gate; the addressee gate wins over topic tracking."""
        if config.attention.busy_mode or config.attention.enabled:
            return create_addressee_gate(config.attention)
        if config.topic_tracking.enabled:
            tracker = create_tracker(config.topic_tracking)
            get_logger(__name__).info(
                "Topic tracking enabled (backend=%s, threshold=%.2f).",
                config.topic_tracking.backend,
                tracker.threshold,
            )
            return tracker
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

        Case and trailing punctuation are ignored, in the utterance and in the
        pattern alike.  ``mode`` is "exact", "prefix", or "regex" (matched from
        the start of the utterance).
        """
        try:
            matcher = _COMMAND_MATCHERS[mode](pattern)
        except KeyError:
            raise ValueError(f"Unknown command mode {mode!r}; expected one of {sorted(_COMMAND_MATCHERS)}.") from None
        self._commands[pattern] = (matcher, handler)

    def unregister_command(self, pattern: str) -> None:
        self._commands.pop(pattern, None)

    def _run_command(self, text: str) -> bool:
        cleaned = _normalise(text)
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
            if self._running:
                self._hotkeys.start()  # run() could only start a dispatcher that existed when it began
        return self._hotkeys

    def _rebind(self, old: Optional[str], new: Optional[str], action: Callable[[], Any]) -> None:
        """Move ``action`` from the combination ``old`` to ``new``, changing nothing unless ``new`` is usable."""
        if new:
            self.hotkeys.is_registered(new)  # raises here, before anything changes, if it cannot be parsed
        if old:
            self.hotkeys.unregister(old)
        if new:
            if self.hotkeys.is_registered(new):  # still bound with ours gone: it belongs to something else
                if old:
                    self.hotkeys.register(old, action)
                raise VoxpipeError(f"Hotkey {new!r} is already in use.")
            self.hotkeys.register(new, action)

    @property
    def push_to_talk(self) -> Optional[str]:
        return self._push_to_talk

    @push_to_talk.setter
    def push_to_talk(self, hotkey: Optional[str]) -> None:
        """Listen only while ``hotkey`` is held; None returns to always-on listening."""
        hotkey = hotkey or None
        if self.asr is not None:
            self._rebind(self._push_to_talk, hotkey, self._hold_to_talk)
            if hotkey:
                self.asr.mute()
            else:
                self.asr.unmute()
        self._push_to_talk = hotkey

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
        hotkey = hotkey or None
        self._rebind(self._press_to_reset, hotkey, self.reset)
        self._press_to_reset = hotkey

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
        # Nothing of the forgotten conversation may leak into the next one.
        self._interrupted_at = None
        self._response_parts = []
        self._held.clear()
        if self.gate is not None:
            self.gate.reset()
        self.session.reset()

    def stop(self) -> None:
        """Ask :meth:`run` to return, leaving unanswered whatever was still waiting. Safe from any thread."""
        self._stopping.set()
        self._transcripts.put(_STOP)
        self._speech_event.set()
        self.interrupt()

    def submit(self, text: str) -> None:
        """Queue text to be handled as if it had been spoken, in turn with what the recogniser hears.

        Safe from any thread while :meth:`run` is active.  Without ``run``,
        call :meth:`handle_transcript` instead.
        """
        self._transcripts.put((text, None))
        self._speech_event.set()

    def _on_speech_onset(self) -> None:
        """The user started talking (called on the ASR thread). If we are mid-reply, yield the floor."""
        self._speech_event.set()  # also cancels a reply waiting in its deferral window
        self._onsets += 1  # a reply that has not started yet learns from this that it must wait
        speaking = self._responding or (self.tts is not None and self.tts.is_speaking)
        if not speaking:
            return
        parts = self._response_parts  # read once: the conversation loop replaces this list
        if parts:
            self._interrupted_at = parts[-1]
        self.interrupt()

    # --- the conversation loop -------------------------------------------

    def run(self) -> None:
        """Listen and respond until :meth:`stop` is called or the recogniser ends. Blocks.

        Raises ASRError if speech recognition failed rather than being stopped.
        """
        if self.asr is None:
            raise ASRError("Cannot run the pipeline without a speech recogniser.")
        try:
            self.asr.start()
            if self.tts is not None:
                self.tts.start()
            if self._hotkeys is not None:
                self._hotkeys.start()
            self._running = True
            threading.Thread(target=self._listen, name="voxpipe-asr", daemon=True).start()

            while not self._stopping.is_set():
                try:
                    item = self._transcripts.get(timeout=_POLL_SECONDS)
                except queue.Empty:
                    continue
                if item is _STOP:
                    break
                transcript, heard_at = item
                try:
                    self._handle(transcript, heard_at)
                except Exception as e:  # noqa: BLE001 - one bad turn must not end the conversation
                    self.events.emit("pipeline:error", e)
                    self.logger.error("Error handling %r: %s", transcript, e, exc_info=not isinstance(e, VoxpipeError))
            if self._listen_error is not None:
                raise ASRError(f"Speech recognition stopped: {self._listen_error}") from self._listen_error
        finally:
            self._running = False
            self.close()

    def _listen(self) -> None:
        """ASR thread: recognise utterances and queue their transcripts for the conversation loop."""
        try:
            for transcript in self.asr:
                # Stamped with the number of speech onsets so far: if the user has started speaking
                # again by the time this is answered, the count will have moved on.
                self._transcripts.put((transcript, self._onsets))
                self._speech_event.set()
        except Exception as e:  # noqa: BLE001 - report and shut the pipeline down cleanly
            self._listen_error = e
            self.events.emit("pipeline:error", e)
            self.logger.error("Speech recognition stopped: %s", e, exc_info=True)
        finally:
            self._transcripts.put(_STOP)

    def handle_transcript(self, transcription: str) -> None:
        """Process one recognised utterance: run a matching command, or have the LLM answer aloud."""
        self._handle(transcription, heard_at=None)

    def _handle(self, transcription: str, heard_at: Optional[int]) -> None:
        """``heard_at`` is the onset count when the utterance was recognised; None for text handed in directly."""
        self.events.emit("asr:transcript", transcription)
        text = qualify_transcript(transcription)
        if text is None:
            # Noise.  If it is what kept an earlier utterance waiting, that one can be answered now.
            if self._held:
                self._respond(None, None, baseline=self._onsets)
            return
        self.events.emit("asr:utterance", text)

        explicit = not self._should_gate(heard_at)
        if explicit and self._run_command(text):
            # An explicit command is not conversation, and should not be recorded as one.
            self._interrupted_at = None
            return

        topic_context = None
        defer_ms = 0
        if not explicit:
            # Any speech queued from here until the deferral ends means someone
            # else kept talking, so the reply should stay silent.
            self._speech_event.clear()
        gate = self.gate
        if gate is not None:
            decision = gate.evaluate(text, explicit=explicit)
            self.events.emit("gate:decided", text, decision)
            if decision.route is not None:
                self.events.emit("topic:routed", text, decision.route)
            if not decision.answer:
                if decision.route is not None:
                    self.logger.info(
                        "Buffered unrelated utterance in topic %d (score %.2f, %s).",
                        decision.route.buffer_id,
                        decision.route.score,
                        decision.route.reason,
                    )
                else:
                    self.logger.info(
                        "Gate declined the utterance (%s, confidence %.2f).", decision.reason, decision.confidence
                    )
                return
            topic_context = decision.context
            defer_ms = decision.defer_ms

        if not explicit and self._run_command(text):
            # A command heard through the gate is only run after the gate accepted it.
            self._interrupted_at = None
            return

        if not explicit and defer_ms > 0 and self._wait_for_speech(defer_ms):
            self.logger.info("Deferred reply cancelled: another utterance arrived.")
            return

        interrupted_at, self._interrupted_at = self._interrupted_at, None
        context = _combine_context(
            f'(User interrupted after: "{interrupted_at}". Continue naturally.)' if interrupted_at else None,
            topic_context,
        )
        self._respond(text, context, baseline=self._onsets if heard_at is None else heard_at)

    def _should_gate(self, heard_at: Optional[int]) -> bool:
        """Only audio from an always-on microphone is gated; typed input and push-to-talk are explicit."""
        return self.gate is not None and heard_at is not None and not self._push_to_talk

    def _wait_for_speech(self, milliseconds: int) -> bool:
        """Wait for another utterance; True means the reply should stay silent."""
        if self._transcripts.qsize() > 0:
            return True
        return self._speech_event.wait(milliseconds / 1000.0)

    def _respond(self, text: Optional[str], context: Optional[str], baseline: int) -> None:
        """Answer ``text`` (and anything held back before it) aloud, unless the user is speaking again.

        ``baseline`` is the onset count the reply belongs to: any onset beyond
        it, before or during the reply, means the user has taken the floor.
        """
        if text is not None:
            self._held.append((text, context))
        self._interrupt.clear()
        self._baseline = baseline
        self._responding = True  # from here on a speech onset interrupts; before, it only moved the count
        try:
            if self._onsets != baseline:
                # The user was already talking again before this reply could start.  Answering now would
                # talk over them, so what they said waits and is answered together with what follows.
                return
            texts, contexts = zip(*self._held)
            self._held.clear()
            self._speak_reply("\n".join(texts), "\n".join(c for c in contexts if c) or None)
        finally:
            self._responding = False

    def _speak_reply(self, text: str, context: Optional[str]) -> None:
        self._response_parts = []
        self.events.emit("pipeline:state", "think")
        if self.tts is not None:
            self.tts.interrupt()  # a new reply never queues behind the tail of the previous one
            # Everything this reply says belongs to the run of speech starting here: once the user
            # barges in, the run is over and a sentence that arrives late is refused, not played.
            self._speech_run = self.tts.generation

        sentences = stream_splitter(self._reply(text, context), min_len=8)
        try:
            for sentence in sentences:
                if self._cut_short():
                    break
                self._say(sentence)
        except LLMError as e:
            self.events.emit("pipeline:error", e)
            self.logger.error("The LLM failed to answer: %s", e)
            self._say(_APOLOGY)
        finally:
            sentences.close()  # ends the turn in the session as well
            if self.gate is not None and self._response_parts:
                self.gate.record_reply(" ".join(self._response_parts))
            self.events.emit("pipeline:state", "idle")

    def _cut_short(self) -> bool:
        """Whether the reply in progress has been interrupted, by a call or by the user speaking."""
        return self._interrupt.is_set() or self._onsets != self._baseline

    def _reply(self, text: str, context: Optional[str]) -> Generator[str, None, None]:
        """The reply's text as it is generated; tool activity is re-emitted as pipeline events."""
        if self._interrupt.is_set():
            return  # interrupted or reset before the turn began: the model is not asked at all
        events = self.session.events(text, context=context)
        try:
            for event in events:
                if self._cut_short():
                    return  # checked on every event; the first arrives before any tool is started
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
            self.tts(sentence, generation=self._speech_run)
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
        if self.gate is not None:
            self._release("gate", self.gate.close)
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
