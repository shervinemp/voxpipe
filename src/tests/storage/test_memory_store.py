import os
import tempfile
import unittest

from voxpipe.storage import MemoryStore, RAMStorage, SQLiteStorage


class TestMemoryStore(unittest.TestCase):
    def test_defaults_to_ram_storage(self):
        mem = MemoryStore()
        self.assertIsInstance(mem.storage, RAMStorage)

    def test_ram_storage_backend(self):
        mem = MemoryStore(storage=RAMStorage())
        mem.store("User prefers dark mode UI.")
        mem.store("User likes Python coding.")

        records = mem.retrieve("dark mode")
        self.assertGreater(len(records), 0)
        self.assertIn("dark mode", records[0].text)

    def test_on_the_fly_backend_swapping(self):
        mem = MemoryStore(storage=RAMStorage())
        mem.store("RAM memory entry")

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with SQLiteStorage(path) as sqlite_backend:
            mem.storage = sqlite_backend
            mem.store("SQLite memory entry")

            records = mem.retrieve("memory entry")
            self.assertEqual([r.text for r in records], ["SQLite memory entry"])

    def test_session_scoping(self):
        storage = RAMStorage()
        alice = MemoryStore(storage, session_id="alice")
        bob = MemoryStore(storage, session_id="bob")
        shared = MemoryStore(storage)

        alice.store("alice keeps bees")
        bob.store("bob keeps goats")
        shared.store("everyone keeps chickens")

        self.assertCountEqual(
            [r.text for r in alice.retrieve("keeps", top_k=5)],
            ["alice keeps bees", "everyone keeps chickens"],
        )
        self.assertEqual([r.text for r in shared.retrieve("keeps", top_k=5)], ["everyone keeps chickens"])

    def test_session_scoping_without_global(self):
        storage = RAMStorage()
        MemoryStore(storage).store("everyone keeps chickens")
        private = MemoryStore(storage, session_id="alice", include_global=False)
        private.store("alice keeps bees")
        self.assertEqual([r.text for r in private.retrieve("keeps", top_k=5)], ["alice keeps bees"])


if __name__ == "__main__":
    unittest.main()
