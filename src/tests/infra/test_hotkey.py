"""Hotkey dispatcher tests, driven with synthetic key events (no global hook is installed)."""
import threading
import unittest
from contextlib import contextmanager

import pytest

try:
    from pynput import keyboard
except ImportError as e:  # pynput needs a display server on Linux
    pytest.skip(f"pynput is unavailable: {e}", allow_module_level=True)

from voxpipe.pipeline.hotkeys import HotkeyDispatcher

Key, KeyCode = keyboard.Key, keyboard.KeyCode


def letter(char, vk=None):
    """A letter key event as the OS reports it: virtual-key code plus the typed character."""
    return KeyCode(vk=vk or ord(char.upper()), char=char)


class TestHotkeyDispatcher(unittest.TestCase):
    def setUp(self):
        self.d = HotkeyDispatcher()
        self.addCleanup(self.d._worker.shutdown, wait=True)
        self.fired = []

    def _settle(self):
        """Wait until the worker has run everything submitted so far."""
        self.d._worker.submit(lambda: None).result(timeout=2)

    def _trigger(self, name="hit"):
        return lambda: self.fired.append(name)

    def _press(self, *keys):
        for key in keys:
            self.d._on_press(key)
        self._settle()

    def _release(self, *keys):
        for key in keys:
            self.d._on_release(key)
        self._settle()

    def test_register_unregister(self):
        self.d.register("a", self._trigger())
        self.d.register("<ctrl>+k", self._trigger())
        self.d.register("a", self._trigger("replaced"))
        self.assertEqual(len(self.d._hotkeys), 2)
        self.d.unregister("a")
        self.d.unregister("a")
        self.assertEqual(len(self.d._hotkeys), 1)

    def test_invalid_hotkey_raises(self):
        with self.assertRaises(ValueError):
            self.d.register("totally+invalid!!!", self._trigger())

    def test_character_key(self):
        self.d.register("a", self._trigger())
        self._press(letter("a"))
        self.assertEqual(self.fired, ["hit"])

    def test_character_key_ignores_shift_state(self):
        self.d.register("k", self._trigger())
        self._press(Key.shift_l, letter("K"))
        self.assertEqual(self.fired, ["hit"])

    def test_generic_modifier_matches_either_side(self):
        self.d.register("<ctrl>+k", self._trigger())
        for ctrl in (Key.ctrl_l, Key.ctrl_r):
            with self.subTest(ctrl=ctrl):
                self.fired.clear()
                self._press(ctrl, letter("k"))
                self.assertEqual(self.fired, ["hit"])
                self._release(letter("k"), ctrl)

    def test_sided_modifier_matches_only_its_side(self):
        self.d.register("<ctrl_r>+<shift_r>", self._trigger())
        self._press(Key.ctrl_l, Key.shift_l)
        self.assertEqual(self.fired, [])
        self._release(Key.ctrl_l, Key.shift_l)
        self._press(Key.ctrl_r, Key.shift_r)
        self.assertEqual(self.fired, ["hit"])

    def test_both_sides_required_when_both_are_named(self):
        self.d.register("<ctrl_l>+<ctrl_r>", self._trigger())
        self._press(Key.ctrl_l)
        self.assertEqual(self.fired, [])
        self._press(Key.ctrl_r)
        self.assertEqual(self.fired, ["hit"])

    def test_function_and_virtual_key_hotkeys(self):
        self.d.register("<f5>", self._trigger("f5"))
        self.d.register("<75>", self._trigger("vk"))
        self._press(Key.f5)
        self._press(letter("k"))
        self.assertEqual(self.fired, ["f5", "vk"])

    def test_auto_repeat_fires_once_per_hold(self):
        self.d.register("<ctrl_l>+<ctrl_r>", self._trigger())
        self._press(Key.ctrl_l, *[Key.ctrl_r] * 5)
        self.assertEqual(self.fired, ["hit"])
        self._release(Key.ctrl_r)
        self._press(Key.ctrl_r)
        self.assertEqual(self.fired, ["hit", "hit"], "a fresh press fires again")

    def test_partial_combination_does_not_fire(self):
        self.d.register("<ctrl>+<alt>+s", self._trigger())
        self._press(Key.ctrl_l, letter("s"))
        self.assertEqual(self.fired, [])

    def test_hold_action_is_entered_on_press_and_exited_on_release(self):
        @contextmanager
        def held():
            self.fired.append("down")
            yield
            self.fired.append("up")

        self.d.register("<ctrl_r>+<shift_r>", held)
        self._press(Key.ctrl_r, Key.shift_r, Key.shift_r)
        self.assertEqual(self.fired, ["down"])
        self._release(Key.shift_r)
        self.assertEqual(self.fired, ["down", "up"])
        self._release(Key.ctrl_r)
        self.assertEqual(self.fired, ["down", "up"])

    def test_actions_run_off_the_listener_thread(self):
        gate = threading.Event()
        self.d.register("a", gate.wait)
        self.d._on_press(letter("a"))  # returns although the action is still blocked
        self.d._on_release(letter("a"))
        gate.set()
        self._settle()

    def test_failing_action_does_not_break_later_ones(self):
        def boom():
            raise RuntimeError("nope")

        self.d.register("a", boom)
        self.d.register("b", self._trigger())
        with self.assertLogs("voxpipe.pipeline.hotkeys", level="ERROR"):
            self._press(letter("a"))
        self._press(letter("b"))
        self.assertEqual(self.fired, ["hit"])

    def test_none_key_events_are_ignored(self):
        self.d.register("a", self._trigger())
        self._press(None)
        self._release(None)
        self.assertEqual(self.fired, [])


if __name__ == "__main__":
    unittest.main()
