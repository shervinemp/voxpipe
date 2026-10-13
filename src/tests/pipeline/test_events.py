"""Tests for EventEmitter on/emit/off."""
import time
import unittest


class TestEventEmitter(unittest.TestCase):
    def setUp(self):
        from voxpipe.pipeline.events import EventEmitter
        self.em = EventEmitter()

    def test_on_and_emit(self):
        events = []
        self.em.on("test", lambda *a, **kw: events.append((a, kw)))
        self.em.emit("test", 1, key="val")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0][0], 1)

    def test_off_removes_handler(self):
        count = 0
        def cb():
            nonlocal count
            count += 1
        self.em.on("test", cb)
        self.em.off("test", cb)
        self.em.emit("test")
        self.assertEqual(count, 0)

    def test_multiple_handlers(self):
        results = []
        self.em.on("ev", lambda: results.append("a"))
        self.em.on("ev", lambda: results.append("b"))
        self.em.emit("ev")
        self.assertEqual(results, ["a", "b"])

    def test_emit_no_handlers(self):
        self.em.emit("nonexistent")

    def test_async_handler(self):
        results = []
        def slow():
            time.sleep(0.02)
            results.append("done")
        self.em.on("ev", slow, async_=True)
        self.em.emit("ev")
        time.sleep(0.05)
        self.assertIn("done", results)


class TestRemovingHandlers(unittest.TestCase):
    def test_off_removes_a_bound_method(self):
        from voxpipe.pipeline.events import EventEmitter

        class Meter:
            def __init__(self):
                self.levels = []

            def on_level(self, level):
                self.levels.append(level)

        emitter, meter = EventEmitter(), Meter()
        self.addCleanup(emitter.close)
        emitter.on("vad:level", meter.on_level)
        emitter.off("vad:level", meter.on_level)  # a new bound-method object, equal to the registered one
        emitter.emit("vad:level", 0.5)
        self.assertEqual(meter.levels, [])
        self.assertEqual(emitter._handlers, {}, "no empty entry is left behind")

    def test_off_for_an_unknown_event_is_harmless(self):
        from voxpipe.pipeline.events import EventEmitter

        emitter = EventEmitter()
        self.addCleanup(emitter.close)
        emitter.off("never:registered", print)
        self.assertEqual(emitter._handlers, {})
