"""Global hotkeys: hold-to-activate and press-to-trigger key combinations."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from typing import Any, Callable, Dict, FrozenSet, Hashable, Optional, Set, Union

from ..core.utils import get_logger

# Called when a combination is pressed.  Returning a context manager makes it a hold
# action: it is entered on press and exited when the combination is released.
HotkeyAction = Union[Callable[[], AbstractContextManager], Callable[[], None]]


class HotkeyDispatcher:
    """Listens for key combinations system-wide and runs the actions registered for them.

    Combinations use pynput's syntax, e.g. ``"<ctrl>+k"`` or ``"<ctrl_r>+<shift_r>"``.
    A generic modifier (``<ctrl>``) matches either side; a sided one only its own.
    Actions run on a worker thread, in order, never on the OS keyboard hook: a
    slow action cannot stall the keyboard or get the hook removed.
    """

    def __init__(self):
        # Imported here so that merely importing voxpipe needs no display server.
        from pynput import keyboard

        self.logger = get_logger(__name__)
        self._keyboard = keyboard
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._canonical = self.listener.canonical

        self._hotkeys: Dict[FrozenSet[Hashable], HotkeyAction] = {}
        self._down: Dict[Hashable, FrozenSet[Hashable]] = {}  # physical key -> the tokens it satisfies
        self._engaged: Set[FrozenSet[Hashable]] = set()  # combinations currently held
        self._held: Dict[FrozenSet[Hashable], AbstractContextManager] = {}
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hotkey")

    # --- registration ----------------------------------------------------

    def register(self, hotkey_string: str, action: HotkeyAction) -> None:
        """Bind ``action`` to a combination, replacing any action already bound to it."""
        self._hotkeys[self._parse(hotkey_string)] = action

    def unregister(self, hotkey_string: str) -> None:
        combination = self._parse(hotkey_string)
        self._hotkeys.pop(combination, None)
        self._engaged.discard(combination)

    def _parse(self, hotkey_string: str) -> FrozenSet[Hashable]:
        return frozenset(self._token(key) for key in self._keyboard.HotKey.parse(hotkey_string))

    # --- key identity ----------------------------------------------------

    def _token(self, key: Any) -> Hashable:
        """A hashable identity for a key as written in a hotkey string."""
        if isinstance(key, self._keyboard.Key):
            return key
        if key.char is not None:
            return ("char", key.char.lower())
        return ("vk", key.vk)

    def _event_tokens(self, key: Any) -> FrozenSet[Hashable]:
        """Every hotkey token a pressed key satisfies.

        A physical key answers to its own identity, its canonical one and its
        virtual-key code.  So left control satisfies ``<ctrl_l>`` (which pynput
        parses to a key code) as well as ``<ctrl>``, and a letter is recognised
        whatever modifiers are changing the character it types.
        """
        tokens = {self._token(key), self._token(self._canonical(key))}
        code = key.value if isinstance(key, self._keyboard.Key) else key
        if code.vk is not None:
            tokens.add(("vk", code.vk))
        return frozenset(tokens)

    def _physical(self, key: Any) -> Hashable:
        """Identity of the physical key, stable between its press and its release."""
        if isinstance(key, self._keyboard.Key):
            return key
        return ("vk", key.vk) if key.vk is not None else ("char", (key.char or "").lower())

    # --- listener callbacks (OS hook thread: must return immediately) -----

    def _on_press(self, key: Optional[Any]) -> None:
        if key is None:
            return
        self._down[self._physical(key)] = self._event_tokens(key)
        pressed = frozenset().union(*self._down.values())
        for combination, action in list(self._hotkeys.items()):
            # Fire on the transition to "held" only; key auto-repeat must not re-trigger.
            if combination <= pressed and combination not in self._engaged:
                self._engaged.add(combination)
                self._worker.submit(self._engage, combination, action)

    def _on_release(self, key: Optional[Any]) -> None:
        if key is None:
            return
        self._down.pop(self._physical(key), None)
        pressed = frozenset().union(*self._down.values())
        for combination in [c for c in self._engaged if not c <= pressed]:
            self._engaged.discard(combination)
            self._worker.submit(self._disengage, combination)

    # --- actions (worker thread) -----------------------------------------

    def _engage(self, combination: FrozenSet[Hashable], action: HotkeyAction) -> None:
        try:
            result = action()
            if isinstance(result, AbstractContextManager):
                result.__enter__()
                self._held[combination] = result
        except Exception:  # noqa: BLE001 - a failing action must not kill the hotkey worker
            self.logger.error("Hotkey action failed.", exc_info=True)

    def _disengage(self, combination: FrozenSet[Hashable]) -> None:
        held = self._held.pop(combination, None)
        if held is None:
            return
        try:
            held.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 - see _engage
            self.logger.error("Hotkey release action failed.", exc_info=True)

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        self.listener.start()

    def stop(self) -> None:
        self.listener.stop()
        if self.listener.is_alive():
            self.listener.join()
        self._worker.shutdown(wait=False)
