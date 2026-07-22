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
        import time
        results = []
        def slow():
            time.sleep(0.02)
            results.append("done")
        self.em.on("ev", slow, async_=True)
        self.em.emit("ev")
        time.sleep(0.05)
        self.assertIn("done", results)
