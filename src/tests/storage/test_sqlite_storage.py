import os
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from voxpipe.core.exceptions import StorageError
from voxpipe.storage import SQLiteStorage


class _SQLiteCase(unittest.TestCase):
    def _path(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(self._unlink, path)
        return path

    @staticmethod
    def _unlink(path):
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(path + suffix)
            except OSError:
                pass

    def _make(self, max_entries=100, ttl_days=30, path=None):
        mem = SQLiteStorage(db_path=path or self._path(), max_entries=max_entries, ttl_days=ttl_days)
        self.addCleanup(self._close, mem)
        return mem

    @staticmethod
    def _close(mem):
        try:
            mem.close()
        except sqlite3.Error:
            pass

    @staticmethod
    def _contents(mem):
        return [row[0] for row in mem._conn.execute("SELECT content FROM records ORDER BY id ASC")]


class TestSQLiteStoreRetrieve(_SQLiteCase):
    """Basic store/retrieve operations."""

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
        self.assertEqual(mem.retrieve("nonexistent", top_k=5), [])

    def test_retrieve_only_stop_words_returns_empty(self):
        mem = self._make()
        mem.store("hello world", role="user")
        self.assertEqual(mem.retrieve("a an the", top_k=5), [])

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
        self.assertEqual(len(mem.retrieve("testing", top_k=3)), 3)

    def test_record_fields(self):
        mem = self._make()
        mem.store("test content", role="user")
        r = mem.retrieve("test", top_k=5)[0]
        self.assertEqual(r["content"], "test content")
        self.assertEqual(r["role"], "user")
        self.assertIsInstance(r["created_at"], float)
        self.assertGreater(r.score, 0)

    def test_store_with_meta(self):
        mem = self._make()
        meta = {"source": "test", "tags": ["rpg"]}
        mem.store("game info", role="user", meta=meta)
        self.assertEqual(meta, {"source": "test", "tags": ["rpg"]}, "caller's meta must not be mutated")
        result = mem.retrieve("game", top_k=5)[0]
        self.assertEqual(result["tags"], ["rpg"])

    def test_dict_content_round_trips_as_json(self):
        mem = self._make()
        mem.store({"fact": "the sky is blue"})
        self.assertIn('"fact"', mem.retrieve("sky")[0].text)

    def test_duplicate_content_returned_once(self):
        mem = self._make()
        for _ in range(3):
            mem.store("repeat after me", role="user")
        self.assertEqual(len(mem.retrieve("repeat", top_k=5)), 1)

    def test_session_and_global_pools(self):
        mem = self._make()
        mem.store("Global system rules for python coding.", session_id=None, role="system")
        mem.store("Alice session preference for dark mode.", session_id="alice", role="user")
        mem.store("Bob session preference for light mode.", session_id="bob", role="user")

        alice = [r["content"] for r in mem.retrieve(
            "preference coding mode", session_id="alice", include_global=True, top_k=5)]
        self.assertIn("Alice session preference for dark mode.", alice)
        self.assertIn("Global system rules for python coding.", alice)
        self.assertNotIn("Bob session preference for light mode.", alice)

        alice_only = [r["content"] for r in mem.retrieve(
            "preference coding mode", session_id="alice", include_global=False, top_k=5)]
        self.assertIn("Alice session preference for dark mode.", alice_only)
        self.assertNotIn("Global system rules for python coding.", alice_only)

    def test_explicit_bank(self):
        mem = self._make()
        mem.store("recipe for pancakes", bank="cooking")
        self.assertEqual(mem.retrieve("pancakes"), [], "default search reads the global bank only")
        self.assertEqual(len(mem.retrieve("pancakes", banks="cooking")), 1)
        self.assertEqual(len(mem.retrieve("pancakes", banks=["cooking", "global"])), 1)


class TestSQLiteRanking(_SQLiteCase):
    """Recall is ranked by relevance, not by which keyword happens to come first."""

    def test_specific_term_beats_recent_filler(self):
        mem = self._make()
        mem.store("my dentist appointment is on friday", role="user")
        for i in range(5):
            mem.store(f"what is the weather like today {i}", role="user")
        results = mem.retrieve("what was my dentist appointment", top_k=2)
        self.assertEqual(results[0]["content"], "my dentist appointment is on friday")

    def test_more_matching_terms_rank_higher(self):
        mem = self._make()
        mem.store("the red bicycle is in the garage", role="user")
        mem.store("the garage door is broken", role="user")
        results = mem.retrieve("red bicycle garage", top_k=2)
        self.assertEqual(results[0]["content"], "the red bicycle is in the garage")
        self.assertGreater(results[0].score, results[1].score)

    def test_query_punctuation_is_not_fts_syntax(self):
        mem = self._make()
        mem.store("my dentist appointment is on friday", role="user")
        for query in ('say "dentist" again', "dentist's (appointment) AND NOT *", "dentist: NEAR friday^2"):
            with self.subTest(query=query):
                self.assertEqual(len(mem.retrieve(query)), 1)

    def test_stemming(self):
        mem = self._make()
        mem.store("we were testing the microphones", role="user")
        self.assertEqual(len(mem.retrieve("microphone test")), 1)


class TestSQLitePersistence(_SQLiteCase):
    def test_rows_survive_close_and_reopen(self):
        path = self._path()
        with SQLiteStorage(path) as mem:
            mem.store("remember the milk", role="user")
        reopened = self._make(path=path)
        self.assertEqual([r["content"] for r in reopened.retrieve("milk")], ["remember the milk"])

    def test_fts_index_survives_reopen_and_eviction(self):
        path = self._path()
        with SQLiteStorage(path, max_entries=2) as mem:
            for word in ("alpha", "bravo", "charlie"):
                mem.store(f"codeword {word}", role="user")
        reopened = self._make(path=path, max_entries=2)
        self.assertEqual(reopened.retrieve("alpha"), [])
        self.assertEqual(len(reopened.retrieve("charlie")), 1)

    def test_legacy_table_is_migrated(self):
        path = self._path()
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY, session_id TEXT, role TEXT NOT NULL,
                content TEXT NOT NULL, meta TEXT DEFAULT '{}', created_at REAL NOT NULL
            );
            INSERT INTO conversations VALUES ('a', NULL, 'user', 'legacy global fact', '{}', strftime('%s','now'));
            INSERT INTO conversations VALUES ('b', 'session:x', 'user', 'legacy session fact', '{}', strftime('%s','now'));
            """
        )
        conn.commit()
        conn.close()

        mem = self._make(path=path)
        self.assertEqual(len(mem.retrieve("legacy")), 1)
        self.assertEqual(len(mem.retrieve("legacy", session_id="x", top_k=5)), 2)
        tables = {row[0] for row in mem._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("conversations", tables)


class TestSQLiteEviction(_SQLiteCase):
    """Eviction by count and TTL."""

    def test_evicts_oldest_when_over_max(self):
        mem = self._make(max_entries=3)
        for i in range(5):
            mem.store(f"Message {i} about testing.", role="user")
        self.assertEqual(self._contents(mem), [f"Message {i} about testing." for i in (2, 3, 4)])

    def test_no_eviction_at_exact_max(self):
        mem = self._make(max_entries=5)
        for i in range(5):
            mem.store(f"Message {i} about parity.", role="user")
        self.assertEqual(len(mem.retrieve("parity", top_k=10)), 5)

    def test_eviction_offset_tracking(self):
        """At each store beyond max, exactly the oldest entry is evicted."""
        mem = self._make(max_entries=3)
        for i in range(6):
            mem.store(f"Entry {i} about offset.", role="user")
            expected = [f"Entry {j} about offset." for j in range(max(0, i - 2), i + 1)]
            self.assertEqual(self._contents(mem), expected)

    def test_eviction_ignores_role(self):
        mem = self._make(max_entries=3)
        mem.store("User greeting message.", role="user")
        mem.store("Assistant response message.", role="assistant")
        mem.store("User followup message.", role="user")
        mem.store("Assistant second message.", role="assistant")
        contents = [r["content"] for r in mem.retrieve("message", top_k=10)]
        self.assertEqual(len(contents), 3)
        self.assertNotIn("User greeting message.", contents)

    def test_retrieve_after_eviction_still_works(self):
        """Eviction must not corrupt remaining entries or the FTS index."""
        mem = self._make(max_entries=3)
        for i in range(4):
            mem.store(f"Unique keyword alpha_{i}.", role="user")
        results = mem.retrieve("alpha_3", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertIn("alpha_3", results[0]["content"])
        self.assertEqual(mem.retrieve("alpha_0"), [])

    def test_evicts_by_ttl_before_count(self):
        mem = self._make(max_entries=10, ttl_days=1)
        mem.store("TTL old entry.", role="user")
        with mem._conn:
            mem._conn.execute("UPDATE records SET created_at = 0 WHERE content = ?", ("TTL old entry.",))
        for i in range(5):
            mem.store(f"Fresh entry {i}.", role="user")
        contents = [r["content"] for r in mem.retrieve("entry", top_k=10)]
        self.assertNotIn("TTL old entry.", contents)
        self.assertEqual(len(contents), 5)

    def test_expired_rows_are_not_recalled_even_before_they_are_deleted(self):
        """Rows are only deleted on the next store(); a recall in between must not see them."""
        mem = self._make(ttl_days=1)
        mem.store("the cat is called Miso")
        self.assertEqual(len(mem.retrieve("cat")), 1)
        with patch("voxpipe.storage.sqlite.time.time", return_value=time.time() + 2 * 86400):
            self.assertEqual(mem.retrieve("cat"), [])
        self.assertEqual(self._contents(mem), ["the cat is called Miso"], "nothing was stored, so nothing was deleted")

    def test_max_entries_must_be_positive(self):
        for max_entries in (0, -1):
            with self.subTest(max_entries=max_entries), self.assertRaises(ValueError):
                SQLiteStorage(self._path(), max_entries=max_entries)

    def test_zero_ttl_keeps_entries(self):
        for ttl in (0, None):
            with self.subTest(ttl=ttl):
                mem = self._make(ttl_days=ttl)
                mem.store("ancient entry", role="user")
                with mem._conn:
                    mem._conn.execute("UPDATE records SET created_at = 0")
                mem.store("recent entry", role="user")
                self.assertEqual(len(mem.retrieve("entry", top_k=5)), 2)


class TestSQLiteErrors(_SQLiteCase):
    def test_a_file_that_is_not_a_database_is_a_storage_error_and_is_not_left_locked(self):
        path = self._path()
        with open(path, "wb") as f:
            f.write(b"definitely not a sqlite database " * 64)
        with self.assertRaisesRegex(StorageError, "not a database"):
            SQLiteStorage(path)
        os.remove(path)  # fails on Windows if the connection was left open

    def test_an_unopenable_path_is_a_storage_error(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(os.rmdir, directory)
        with self.assertRaises(StorageError):
            SQLiteStorage(directory)

    def test_use_after_close_is_a_storage_error(self):
        mem = self._make()
        mem.store("something to remember")
        mem.close()
        with self.assertRaisesRegex(StorageError, "closed"):
            mem.store("too late")
        with self.assertRaises(StorageError):
            mem.retrieve("something")
        mem.close()  # closing twice is harmless


class TestSQLiteConcurrency(_SQLiteCase):
    def test_concurrent_store_and_retrieve(self):
        mem = self._make(max_entries=1000)
        errors = []

        def worker(n):
            try:
                for i in range(20):
                    mem.store(f"Worker {n} message {i} about concurrency.", role="user")
                    mem.retrieve("concurrency")
            except Exception as e:  # noqa: BLE001 - any failure in a worker fails the test
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(self._contents(mem)), 80)


if __name__ == "__main__":
    unittest.main()
