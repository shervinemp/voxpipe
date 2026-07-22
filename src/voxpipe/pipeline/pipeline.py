from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import Callable, Optional

from ..llm.conversation import Conversation
from ..asr.model import ASRProviders
from ..tts.model import TTSProviders
from .hotkeys import HotkeyDispatcher
from ..llm import Session, LLMProviders
from ..llm.tools import Tool
from ..streaming.splitter import stream_splitter
from ..core.utils import setup_logging, get_logger
from ..core.config import config
from .events import EventEmitter
from ..core.exceptions import VoiceControlError, ASRError, LLMError, TTSError, ConfigError
from .gate import qualify_transcript


class Pipeline:
    def __init__(
        self,
        session: Optional[Session] = None,
        push_to_talk: str | None = None,
        press_to_reset: str | None = None,
    ):
        self.logger = get_logger(__name__)
        self.events = EventEmitter()
        self._running = False
        self._interrupt_event = threading.Event()
        self._response_parts: list[str] = []
        self._interrupted_at: str | None = None
        self._llm_busy = False
        self._commands: dict[str, tuple[Callable, str]] = {}

        asr_cls = getattr(ASRProviders, config.get("asr.provider"))
        tts_cls = getattr(TTSProviders, config.get("tts.provider"))

        if session is not None:
            self.asr = asr_cls()
            self.tts = tts_cls()
            self.session = session
        else:
            llm_backend = config.get("llm.backend")
            llm_model = config.get("llm.model")
            with ThreadPoolExecutor(max_workers=3) as pool:
                asr_future = pool.submit(asr_cls)
                tts_future = pool.submit(tts_cls)
                llm_future = pool.submit(LLMProviders.create, llm_backend, llm_model)
                for name, future in [("ASR", asr_future), ("TTS", tts_future), ("LLM", llm_future)]:
                    exc = future.exception()
                    if exc:
                        self.logger.error("%s initialization failed: %s", name, exc)
                self.asr = asr_future.result() if not asr_future.exception() else None
                self.tts = tts_future.result() if not tts_future.exception() else None
                llm = llm_future.result() if not llm_future.exception() else None
                if llm is None:
                    raise LLMError("LLM initialization failed")
                max_tool_iterations = config.get("llm.max_tool_iterations", 1)
                self.session = Session(llm=llm, max_tool_iterations=max_tool_iterations)

        if self.asr is None:
            self.logger.warning("ASR unavailable")
        if self.tts is None:
            self.logger.warning("TTS unavailable")

        if self.asr and hasattr(self.asr, "_vad"):
            self.asr._vad.on_speech_onset = self._on_user_interrupt
            self.asr._vad.on_audio_level = lambda rms, prob: self.events.emit("vad:level", rms, prob)

        if push_to_talk is None:
            push_to_talk = config.get("hotkeys.push_to_talk")
        if press_to_reset is None:
            press_to_reset = config.get("hotkeys.press_to_reset")
        if push_to_talk is not None and not config.get("hotkeys.enable", True):
            push_to_talk = None
        if press_to_reset is not None and not config.get("hotkeys.enable", True):
            press_to_reset = None
        self.push_to_talk = push_to_talk
        self.press_to_reset = press_to_reset

        self.memory = None
        self._init_memory()
        self._watchdog = None

    def _start_watchdog(self):
        def _watch():
            while self._running:
                time.sleep(30)
                if self._llm_busy:
                    self.logger.warning("Pipeline watchdog: LLM busy > 30s")
        self._watchdog = threading.Thread(target=_watch, daemon=True)
        self._watchdog.start()

    def _init_memory(self):
        cfg = config.get("conversation_history")
        if not cfg or not getattr(cfg, "enabled", False):
            self.memory = None
            return
        from voxpipe.storage.memory import Memory
        db_path = getattr(cfg, "db_path", "data/conversations.db")
        max_entries = getattr(cfg, "max_entries", 1000)
        ttl_days = getattr(cfg, "ttl_days", 30)
        try:
            self.memory = Memory(db_path=db_path, max_entries=max_entries, ttl_days=ttl_days)
            self._conv_top_k = getattr(cfg, "top_k", 2)
        except Exception as e:
            self.logger.warning("Memory unavailable: %s", e)
            self.memory = None

    def register_command(self, pattern, handler, mode="exact"):
        self._commands[pattern] = (handler, mode)

    def unregister_command(self, pattern):
        self._commands.pop(pattern, None)

    @property
    def status(self):
        asr_muted = getattr(self.asr, "_is_muted", None)
        tts_running = hasattr(self.tts, "audio_player") and self.tts.audio_player._running
        return {
            "asr": "unavailable" if self.asr is None else ("muted" if asr_muted else "listening"),
            "llm": "generating" if self._llm_busy else "idle",
            "tts": "unavailable" if self.tts is None else ("speaking" if tts_running else "idle"),
        }

    def register_tools(self, *tools):
        for tool in tools:
            if isinstance(tool, Tool):
                self.session.conversation.tools[tool.name] = tool
            elif isinstance(tool, list):
                for t in tool:
                    if isinstance(t, Tool):
                        self.session.conversation.tools[t.name] = t
        self._configure_session()

    def _on_user_interrupt(self):
        now = time.monotonic()
        if now - getattr(self, "_last_interrupt", 0) < 0.2:
            return
        self._last_interrupt = now
        if self._response_parts:
            self._interrupted_at = self._response_parts[-1]
        self._interrupt_event.set()
        if self.tts and hasattr(self.tts, "audio_player"):
            self.tts.audio_player.stop_playback()

    def _match_command(self, text):
        cleaned = text.strip().lower()
        for pattern, (handler, mode) in self._commands.items():
            if mode == "exact" and cleaned == pattern:
                handler(); return True
            if mode == "prefix" and cleaned.startswith(pattern):
                handler(); return True
            if mode == "regex" and __import__("re").match(pattern, cleaned):
                handler(); return True
        return False

    def _callback(self, transcription):
        self.events.emit("asr:transcript", transcription)
        text, annotation = qualify_transcript(transcription)
        if text is None:
            return
        self.events.emit("asr:utterance", text, annotation=annotation)
        if annotation:
            text = f"{annotation}\n{text}"
        if self._interrupted_at:
            text = f'(User interrupted after: "{self._interrupted_at}". Continue naturally.)\n{text}'
            self._interrupted_at = None
        if self._match_command(text):
            return

        original_query = text
        if self.memory:
            context = self.memory.retrieve(text, top_k=getattr(self, "_conv_top_k", 2))
            injected = [f"(Earlier: {ctx['content'][:200]})" for ctx in context if ctx.get("content")]
            if injected:
                text = "\n".join(injected) + "\n" + text

        self._response_parts = []
        self._llm_busy = True
        self.events.emit("pipeline:state", "think")
        interrupt = True
        self._interrupt_event.clear()
        out = self.session(text)
        for sentence in stream_splitter(out, min_len=8):
            if self._interrupt_event.is_set():
                self._interrupt_event.clear()
                self._interrupted_at = self._response_parts[-1] if self._response_parts else None
                if hasattr(out, "close"):
                    out.close()
                break
            if s := sentence.strip():
                self._response_parts.append(s)
                self.events.emit("pipeline:state", "speak")
                self.events.emit("tts:start", s)
                if self.tts:
                    self.tts(s, interrupt=interrupt)
                else:
                    self.logger.info("[TTS degraded] %s", s)
                self.events.emit("tts:utterance", s)
                interrupt = False
        self.events.emit("pipeline:state", "idle")
        self._llm_busy = False

        if self.memory and original_query:
            self.memory.store(original_query, role="user")
            full_response = " ".join(self._response_parts)
            if full_response:
                self.memory.store(full_response, role="assistant")

    def run(self):
        if self.asr:
            self.asr.start()
        else:
            self.logger.error("ASR unavailable")
            return
        if self.tts:
            self.tts.start()
        self.hotkey_dispatcher.start()
        self._running = True
        self._start_watchdog()

        try:
            for transcript in self.asr:
                try:
                    self._callback(transcript)
                except VoiceControlError as e:
                    self.events.emit("pipeline:error", e)
                    self.logger.error(f"Pipeline error: {e}")
                except Exception as e:
                    self.events.emit("pipeline:error", e)
                    self.logger.error(f"Unexpected error: {e}", exc_info=True)
        except Exception as e:
            self.events.emit("pipeline:error", e)
            self.logger.error("ASR loop terminated: %s", e, exc_info=True)
        finally:
            self._running = False
            if dispatcher := getattr(self, "_hotkey_dispatcher", None):
                dispatcher.stop()
            if self.asr:
                self.asr.stop()
            if self.tts:
                self.tts.stop()
            self.session.close()
            if self.memory:
                self.memory.close()
            self.events.close()

    def _configure_session(self):
        if not self.session.conversation._system:
            rules = []
            for tool in self.session.conversation.tools.values():
                if tool.instruction:
                    rules.append(f"- {' '.join(tool.instruction.split())}")
            rules.extend([
                "- If the user's message seems incomplete or cut off, ask what they meant.",
                "- Do not describe your actions or announce tool usage.",
                "- Respond naturally, as if you already know the answer.",
            ])
            self.session.conversation.set_system_message(
                "You are a voice-controlled assistant. Respond conversationally.\n\n"
                "Rules:\n" + "\n".join(rules)
            )

    @property
    def push_to_talk(self):
        return self._push_to_talk

    @push_to_talk.setter
    def push_to_talk(self, value):
        if self.asr is None:
            return
        dispatcher = self.hotkey_dispatcher
        if k_ := getattr(self, "_push_to_talk", None):
            dispatcher.unregister(k_)
        self._push_to_talk = value
        if value is None:
            self.asr.enable()
        else:
            @contextmanager
            def cb():
                self.asr.enable()
                yield
                self.asr.disable_w_passthrough()
            self.asr.disable_w_passthrough()
            dispatcher.register(value, cb)

    @property
    def press_to_reset(self):
        return self._press_to_reset

    @press_to_reset.setter
    def press_to_reset(self, value):
        dispatcher = self.hotkey_dispatcher
        if k_ := getattr(self, "_press_to_reset", None):
            dispatcher.unregister(k_)
        self._press_to_reset = value
        if value:
            def cb():
                self.events.emit("session:reset")
                old_system_msg = self.session.conversation._system
                old_tools = self.session.conversation.tools
                self.session.reset(Conversation())
                if old_system_msg:
                    self.session.conversation.set_system_message(old_system_msg)
                if old_tools:
                    self.session.conversation.tools = old_tools
                self._configure_session()
            dispatcher.register(value, cb)

    @property
    def hotkey_dispatcher(self) -> HotkeyDispatcher:
        attr_name = "_hotkey_dispatcher"
        if dispatcher := getattr(self, attr_name, None):
            return dispatcher
        dispatcher = HotkeyDispatcher()
        setattr(self, attr_name, dispatcher)
        return dispatcher


def main():
    setup_logging(log_level="INFO")
    logger = get_logger(__name__)
    while True:
        try:
            pipe = Pipeline()
            ptt = config.get("hotkeys.push_to_talk")
            msg = f"Hold {ptt} to speak." if ptt else "Always-on VAD mode."
            logger.info("Voice pipeline ready. %s", msg)
            pipe.run()
        except Exception as e:
            logger.error(f"Pipeline exited: {e}", exc_info=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
