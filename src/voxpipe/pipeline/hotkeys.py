from pynput import keyboard
from typing import Dict, Set, Callable, Union, TypeAlias, ContextManager

HotkeyAction: TypeAlias = Union[
    Callable[[], ContextManager[None]], Callable[[], None]
]


class HotkeyDispatcher:
    def __init__(self):
        """Initializes the hotkey dispatcher."""
        self.hotkeys: Dict[frozenset, HotkeyAction] = {}
        self.active_contexts: Dict[frozenset, ContextManager[None]] = {}
        self.pressed_keys: Set[keyboard.Key | keyboard.KeyCode] = set()
        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )

    def _normalize_key(
        self, key: keyboard.Key | keyboard.KeyCode | None
    ) -> int | None:
        """Forces all pynput keys into native Python integers."""
        if key is None:
            return None
        if hasattr(key, "value"):
            key = key.value
        if hasattr(key, "vk") and key.vk is not None:
            return key.vk
        if hasattr(key, "char") and key.char is not None:
            return ord(key.char.lower())
        return hash(str(key))

    def register(self, hotkey_string: str, action: HotkeyAction):
        """Registers a hotkey with a specific action."""
        parsed_keys = keyboard.HotKey.parse(hotkey_string)
        hotkey = frozenset(self._normalize_key(k) for k in parsed_keys)
        self.hotkeys[hotkey] = action

    def unregister(self, hotkey_string: str):
        """Unregisters a hotkey."""
        parsed_keys = keyboard.HotKey.parse(hotkey_string)
        hotkey = frozenset(self._normalize_key(k) for k in parsed_keys)
        if hotkey in self.hotkeys:
            del self.hotkeys[hotkey]

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None):
        if key is None:
            return

        norm_key = self._normalize_key(key)
        self.pressed_keys.add(norm_key)

        for hotkey, action in self.hotkeys.items():
            if (
                hotkey.issubset(self.pressed_keys)
                and hotkey not in self.active_contexts
            ):
                context_manager = action()
                if hasattr(context_manager, "__enter__") and hasattr(
                    context_manager, "__exit__"
                ):
                    context_manager.__enter__()
                    self.active_contexts[hotkey] = context_manager

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None):
        norm_key = self._normalize_key(key)

        if norm_key in self.pressed_keys:
            self.pressed_keys.remove(norm_key)

        for hotkey, context_manager in list(self.active_contexts.items()):
            if not hotkey.issubset(self.pressed_keys):
                context_manager.__exit__(None, None, None)
                del self.active_contexts[hotkey]

    def start(self):
        """Starts the keyboard listener."""
        self.listener.start()

    def stop(self):
        """Stops the keyboard listener."""
        self.listener.stop()
        if self.listener.is_alive():
            self.listener.join()
