import unittest
from voxpipe.storage import MemoryStore, RAMStorage, SQLiteStorage, Record
import tempfile
import os


class TestMemoryStoreFacade(unittest.TestCase):
    def test_ram_storage_backend(self):
        mem = MemoryStore(backend=RAMStorage(), bank="session:123")
        mem.store("User prefers dark mode UI.")
        mem.store("User likes Python coding.")

        records = mem.retrieve("dark mode")
        self.assertGreater(len(records), 0)
        self.assertIn("dark mode", records[0].text)

    def test_on_the_fly_backend_swapping(self):
        mem = MemoryStore(backend=RAMStorage(), bank="session:abc")
        mem.store("RAM memory entry")

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        sqlite_backend = SQLiteStorage(path)

        try:
            # Swap backend on the fly!
            mem.backend = sqlite_backend
            mem.store("SQLite memory entry")

            records = mem.retrieve("SQLite")
            self.assertEqual(len(records), 1)
            self.assertIn("SQLite memory entry", records[0].text)
        finally:
            sqlite_backend.close()
            try:
                os.unlink(path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
