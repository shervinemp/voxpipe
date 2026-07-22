import os
import threading
import time
import tempfile
import unittest


class TestMemoryStoreRetrieve(unittest.TestCase):
    """Basic store/retrieve operations."""

    def _make(self, max_entries=100, ttl_days=30):
        from voxpipe.storage.memory import Memory
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        def cleanup():
            try:
                mem.close()
            except Exception:
                pass
            try:
                os.unlink(path)
            except Exception:
                pass

        mem = Memory(db_path=path, max_entries=max_entries, ttl_days=ttl_days)
        self.addCleanup(cleanup)
        return mem

    def test_store_and_retrieve(self):
        mem = self._make()
        mem.store("What is the meaning of life?", role="user")
        mem.store("The meaning of life is 42.", role="assistant")
        results = mem.retrieve("meaning of life", top_k=5)
        self.assertEqual(len(results), 2)
        contents = [r["content"] for r in results]
        self.assertIn("What is the meaning of life?", contents)
        self.assertTrue(any("42" in c for c in contents))

    def test_retrieve_empty_returns_empty_list(self):
        mem = self._make()
        results = mem.retrieve("nonexistent", top_k=5)
        self.assertEqual(results, [])

    def test_retrieve_no_long_keywords_returns_empty(self):
        mem = self._make()
        mem.store("hello world", role="user")
        results = mem.retrieve("a an the", top_k=5)
        self.assertEqual(results, [])

    def test_role_filtered_retrieve(self):
        mem = self._make()
        mem.store("User query about Elden Ring.", role="user")
        mem.store("Elden Ring is a game by FromSoftware.", role="assistant")
        user_only = mem.retrieve("Elden Ring", top_k=5, role="user")
        self.assertEqual(len(user_only), 1)
        self.assertEqual(user_only[0]["role"], "user")

    def test_top_k_limits_results(self):
        mem = self._make()
        for i in range(10):
            mem.store(f"This is message {i} about testing.", role="user")
        results = mem.retrieve("testing", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_result_dict_keys(self):
        mem = self._make()
        mem.store("test content", role="user")
        results = mem.retrieve("test", top_k=5)
        self.assertGreater(len(results), 0)
        r = results[0]
        self.assertIn("content", r)
        self.assertIn("role", r)
        self.assertIn("created_at", r)
        self.assertIsInstance(r["created_at"], float)

    def test_store_with_meta(self):
        mem = self._make()
        mem.store("game info", role="user", meta={"source": "test", "tags": ["rpg"]})
        results = mem.retrieve("game", top_k=5)
        self.assertTrue(any("game info" in r["content"] for r in results))

    def test_session_and_global_pools(self):
        mem = self._make()
        mem.store("Global system rules for python coding.", session_id=None, role="system")
        mem.store("Alice session preference for dark mode.", session_id="alice", role="user")
        mem.store("Bob session preference for light mode.", session_id="bob", role="user")

        alice_res = mem.retrieve("preference coding mode", session_id="alice", include_global=True, top_k=5)
        alice_contents = [r["content"] for r in alice_res]
        self.assertIn("Alice session preference for dark mode.", alice_contents)
        self.assertIn("Global system rules for python coding.", alice_contents)
        self.assertNotIn("Bob session preference for light mode.", alice_contents)

        alice_only = mem.retrieve("preference coding mode", session_id="alice", include_global=False, top_k=5)
        alice_only_contents = [r["content"] for r in alice_only]
        self.assertIn("Alice session preference for dark mode.", alice_only_contents)
        self.assertNotIn("Global system rules for python coding.", alice_only_contents)


class TestMemoryEviction(unittest.TestCase):
    """Eviction by count and TTL."""

    def _make(self, max_entries=5, ttl_days=30):
        from voxpipe.storage.memory import Memory
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        def cleanup():
            try:
                mem.close()
            except Exception:
                pass
            try:
                os.unlink(path)
            except Exception:
                pass

        mem = Memory(db_path=path, max_entries=max_entries, ttl_days=ttl_days)
        self.addCleanup(cleanup)
        return mem

    def test_evicts_oldest_when_over_max(self):
        mem = self._make(max_entries=3)
        for i in range(5):
            mem.store(f"Message {i} about testing.", role="user")
        results = mem.retrieve("testing", top_k=10)
        self.assertLessEqual(len(results), 3)

    def test_eviction_preserves_newest(self):
        mem = self._make(max_entries=3)
        for i in range(5):
            mem.store(f"Message {i} about testing.", role="user")
        row = mem._conn.execute(
            "SELECT content FROM conversations ORDER BY created_at DESC"
        ).fetchall()
        remaining = [r[0] for r in row]
        self.assertEqual(len(remaining), 3)
        self.assertIn("Message 4", remaining[0],
                      "Newest message should be first in DB")

    def test_no_eviction_at_exact_max(self):
        mem = self._make(max_entries=5)
        for i in range(5):
            mem.store(f"Message {i} about parity.", role="user")
        results = mem.retrieve("parity", top_k=10)
        self.assertEqual(len(results), 5)

    def test_eviction_preserves_exact_max_count(self):
        mem = self._make(max_entries=4)
        for i in range(10):
            mem.store(f"Message {i} about counting.", role="user")
        results = mem.retrieve("counting", top_k=10)
        self.assertEqual(len(results), 4)

    def test_eviction_offset_tracking(self):
        """At each store beyond max, verify which entry was evicted."""
        mem = self._make(max_entries=3)
        stored = []
        for i in range(6):
            mem.store(f"Entry {i} about offset.", role="user")
            stored.append(i)
            row = mem._conn.execute(
                "SELECT content FROM conversations ORDER BY created_at ASC"
            ).fetchall()
            remaining = [r[0] for r in row]
            expected = [f"Entry {j} about offset." for j in stored[-3:]]
            self.assertEqual(
                remaining, expected,
                f"After store {i}: expected [{', '.join(expected)}], "
                f"got [{', '.join(remaining)}]"
            )

    def test_eviction_respects_ttl_before_count(self):
        """TTL eviction runs first; count eviction only on surplus beyond max."""
        mem = self._make(max_entries=10, ttl_days=0)
        mem.store("TTL old entry.", role="user")
        mem._conn.execute(
            "UPDATE conversations SET created_at = 0 WHERE content = ?",
            ("TTL old entry.",),
        )
        mem._conn.commit()
        for i in range(5):
            mem.store(f"Fresh entry {i}.", role="user")
        results = mem.retrieve("entry", top_k=10)
        contents = [r["content"] for r in results]
        self.assertNotIn("TTL old entry.", contents,
                         "TTL-expired entry should be evicted")
        self.assertEqual(len(results), 5,
                         "Fresh entries below max should survive")

    def test_eviction_with_mixed_role_entries(self):
        """Eviction treats all roles equally — oldest first regardless."""
        mem = self._make(max_entries=3)
        mem.store("User greeting message.", role="user")
        mem.store("Assistant response message.", role="assistant")
        mem.store("User followup message.", role="user")
        mem.store("Assistant second message.", role="assistant")
        results = mem.retrieve("message", top_k=10)
        self.assertEqual(len(results), 3)
        contents = [r["content"] for r in results]
        self.assertNotIn("User greeting message.", contents)

    def test_retrieve_after_eviction_still_works(self):
        """Eviction should not corrupt remaining entries or FTS index."""
        mem = self._make(max_entries=3)
        for i in range(4):
            mem.store(f"Unique keyword alpha_{i}.", role="user")
        results = mem.retrieve("alpha_3", top_k=5)
        self.assertEqual(len(results), 1,
                         "Should find the surviving newest entry")
        self.assertIn("alpha_3", results[0]["content"])

    def test_eviction_does_not_affect_unrelated_keywords(self):
        """Entries with different keywords survive eviction correctly."""
        mem = self._make(max_entries=3)
        mem.store("zebra animal.", role="user")
        mem.store("apple fruit.", role="user")
        mem.store("kite toy.", role="user")
        mem.store("zebra again.", role="user")
        results = mem.retrieve("zebra", top_k=5)
        self.assertEqual(len(results), 1,
                         "Only newest zebra entry should survive")
        self.assertIn("again", results[0]["content"])

    def test_does_not_evict_when_under_max(self):
        mem = self._make(max_entries=10)
        for i in range(5):
            mem.store(f"Message {i} about cats.", role="user")
        results = mem.retrieve("cats", top_k=10)
        self.assertEqual(len(results), 5)

    def test_evicts_by_ttl(self):
        mem = self._make(max_entries=100, ttl_days=0)
        mem.store("This is old content.", role="user")
        mem._conn.execute(
            "UPDATE conversations SET created_at = 0 WHERE content = ?",
            ("This is old content.",),
        )
        mem._conn.commit()
        mem.store("This is new content.", role="user")
        results = mem.retrieve("content", top_k=10)
        contents = [r["content"] for r in results]
        self.assertNotIn("This is old content.", contents)
        self.assertIn("This is new content.", contents)


class TestMemoryConcurrency(unittest.TestCase):
    """Thread safety."""

    def _make(self):
        from voxpipe.storage.memory import Memory
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        def cleanup():
            try:
                mem.close()
            except Exception:
                pass
            try:
                os.unlink(path)
            except Exception:
                pass

        mem = Memory(db_path=path, max_entries=1000, ttl_days=30)
        self.addCleanup(cleanup)
        return mem

    def test_concurrent_store(self):
        mem = self._make()
        errors = []
        def worker(n):
            try:
                for i in range(20):
                    mem.store(f"Worker {n} message {i} about concurrency.", role="user")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])
        results = mem.retrieve("concurrency", top_k=100)
        self.assertGreaterEqual(len(results), 1)


class TestMemoryPipelineIntegration(unittest.TestCase):
    """Memory used within the pipeline callback flow."""

    def _make_pipe_with_memory(self, enabled=True):
        from voxpipe.pipeline.pipeline import Pipeline
        from unittest.mock import MagicMock
        import threading as _thr

        p = Pipeline.__new__(Pipeline)
        p.logger = MagicMock()
        p.events = MagicMock()
        p.memory = None
        p._response_parts = []
        p._llm_busy = False
        p._interrupt_event = _thr.Event()
        p._interrupted_at = None
        p._match_command = MagicMock(return_value=False)
        p.tts = None

        if enabled:
            from voxpipe.storage.memory import Memory
            fd, path = tempfile.mkstemp(suffix=".db")
            os.close(fd)

            def cleanup():
                try:
                    p.memory.close()
                except Exception:
                    pass
                try:
                    os.unlink(path)
                except Exception:
                    pass
            self.addCleanup(cleanup)

            p.memory = Memory(db_path=path, max_entries=100, ttl_days=30)
            p._conv_top_k = 3
        return p

    def test_callback_injects_memory_context(self):
        p = self._make_pipe_with_memory(enabled=True)
        p.memory.store("I love playing Elden Ring.", role="user")
        p.memory.store("Elden Ring is an action RPG.", role="assistant")

        captured = []
        def mock_session(text):
            captured.append(text)
            return iter(["OK."])
        p.session = mock_session

        p._callback("What is Elden Ring about?")
        self.assertGreater(len(captured), 0)
        self.assertIn("Elden Ring", captured[0])
        self.assertIn("Earlier:", captured[0])

    def test_callback_no_memory_when_disabled(self):
        p = self._make_pipe_with_memory(enabled=False)
        captured = []
        def mock_session(text):
            captured.append(text)
            return iter(["OK."])
        p.session = mock_session
        p._callback("What is Elden Ring?")
        self.assertIn("Elden Ring", captured[0])

    def test_callback_stores_after_response(self):
        p = self._make_pipe_with_memory(enabled=True)
        p._response_parts = []
        p.session = lambda text: iter(["Elden Ring is a challenging game."])
        p._callback("What is Elden Ring?")
        results = p.memory.retrieve("Elden Ring", top_k=5)
        self.assertGreaterEqual(len(results), 1)
        roles = [r["role"] for r in results]
        self.assertIn("user", roles)

    def test_memory_turns_accumulate(self):
        p = self._make_pipe_with_memory(enabled=True)
        p._response_parts = []
        p.session = lambda text: iter(["Dark Souls is a hard game."])
        p._callback("Tell me about Dark Souls.")

        p._response_parts = []
        p.session = lambda text: iter(["Bloodborne is a faster game."])
        p._callback("Now tell me about Bloodborne.")

        results = p.memory.retrieve("Dark", top_k=5)
        self.assertTrue(any("Dark Souls" in r["content"] for r in results),
                        "Dark Souls should be in memory")
