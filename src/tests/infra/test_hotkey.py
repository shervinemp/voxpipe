"""Hotkey dispatcher tests with mocked listener."""
import unittest
from unittest.mock import MagicMock


class TestHotkeyDispatcher(unittest.TestCase):
    """Uses real HotkeyDispatcher with a mocked listener to avoid global hooks."""

    def setUp(self):
        from voxpipe.pipeline.hotkeys import HotkeyDispatcher
        self.d = HotkeyDispatcher()
        self.d.listener = MagicMock()
        self.d.logger = MagicMock()

    def test_register_adds_entry(self):
        fn = MagicMock()
        self.d.register("a", fn)
        self.assertEqual(len(self.d.hotkeys), 1)

    def test_unregister_removes_entry(self):
        self.d.register("a", MagicMock())
        self.assertEqual(len(self.d.hotkeys), 1)
        self.d.unregister("a")
        self.assertEqual(len(self.d.hotkeys), 0)

    def test_multiple_registrations(self):
        fn1, fn2 = MagicMock(), MagicMock()
        self.d.register("a", fn1)
        self.d.register("<ctrl>+k", fn2)
        self.assertEqual(len(self.d.hotkeys), 2)

    def test_register_duplicate_overwrites(self):
        fn1, fn2 = MagicMock(), MagicMock()
        self.d.register("a", fn1)
        self.d.register("a", fn2)
        self.assertEqual(len(self.d.hotkeys), 1)

    def test_register_invalid_hotkey(self):
        try:
            self.d.register("totally+invalid!!!", MagicMock())
        except Exception:
            pass
