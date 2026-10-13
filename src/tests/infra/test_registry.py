"""Tests for the model manifest and downloader."""
import hashlib
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from voxpipe.core import downloader
from voxpipe.core.downloader import download_file, ensure_downloaded, is_downloaded, load_manifest, verify_file_sha256
from voxpipe.core.exceptions import ModelError


class TestManifest(unittest.TestCase):
    def test_manifest_lists_the_default_models(self):
        manifest = load_manifest()
        for name in ("Qwen3", "Gemma4E2B", "Gemma4E4B", "Gemma4_12B", "Kokoro"):
            self.assertIn(name, manifest)
        self.assertIn("repo", manifest["Gemma4E4B"])

    def test_every_configured_local_model_can_be_downloaded(self):
        from voxpipe.core.config import load_config

        manifest = load_manifest()
        for name, entry in load_config().llm.local.items():
            if entry.model_path is None:
                self.assertIn(name, manifest, f"llm.local.{name} has no model_path and no manifest entry")

    def test_unknown_model(self):
        with self.assertRaises(ModelError):
            ensure_downloaded("nonexistent")
        self.assertFalse(is_downloaded("nonexistent"))


class TestVerification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "weights.bin")
        with open(self.path, "wb") as f:
            f.write(b"model weights")
        self.digest = hashlib.sha256(b"model weights").hexdigest()

    def test_verify_file_sha256(self):
        verify_file_sha256(self.path, self.digest)
        verify_file_sha256(self.path, self.digest.upper())
        with self.assertRaises(ValueError):
            verify_file_sha256(self.path, "0" * 64)
        with self.assertRaises(ValueError):
            verify_file_sha256(self.path, "not-a-digest")

    def _manifest(self, sha256=None):
        return {
            "Tiny": {
                "allowed_hosts": ["example.com"],
                "files": {
                    "model": {
                        "url": "https://example.com/files/weights.bin",
                        "sha256": sha256 or self.digest,
                        "max_bytes": 1000,
                    }
                },
            }
        }

    def test_present_files_are_verified_not_downloaded(self):
        with patch.object(downloader, "load_manifest", return_value=self._manifest()), patch.object(
            downloader, "download_file"
        ) as download:
            self.assertEqual(ensure_downloaded("Tiny", local_dir=self.tmp.name), {"model": self.path})
            self.assertTrue(is_downloaded("Tiny", local_dir=self.tmp.name))
        download.assert_not_called()

    def test_corrupt_files_are_rejected(self):
        with patch.object(downloader, "load_manifest", return_value=self._manifest(sha256="0" * 64)):
            self.assertFalse(is_downloaded("Tiny", local_dir=self.tmp.name))
            with self.assertRaisesRegex(ModelError, "delete it to download a fresh copy"):
                ensure_downloaded("Tiny", local_dir=self.tmp.name)
        self.assertTrue(os.path.exists(self.path), "a file that fails the check is never removed for the user")

    def test_a_failed_download_is_a_model_error(self):
        os.unlink(self.path)
        for failure in (OSError("network unreachable"), ValueError("Download exceeds the configured size limit.")):
            with self.subTest(failure=failure), patch.object(
                downloader, "load_manifest", return_value=self._manifest()
            ), patch.object(downloader, "download_file", side_effect=failure):
                with self.assertRaisesRegex(ModelError, "Could not download 'Tiny'"):
                    ensure_downloaded("Tiny", local_dir=self.tmp.name)

    def test_missing_files_are_fetched(self):
        os.unlink(self.path)
        with patch.object(downloader, "load_manifest", return_value=self._manifest()), patch.object(
            downloader, "download_file"
        ) as download:
            self.assertFalse(is_downloaded("Tiny", local_dir=self.tmp.name))
            ensure_downloaded("Tiny", local_dir=self.tmp.name)
        download.assert_called_once()
        self.assertEqual(download.call_args.kwargs["destination"], self.path)
        self.assertEqual(download.call_args.kwargs["allowed_hosts"], {"example.com"})


class TestDownloadFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.destination = os.path.join(self.tmp.name, "sub", "weights.bin")
        self.payload = b"x" * 100
        self.digest = hashlib.sha256(self.payload).hexdigest()

    def _response(self, body=None, redirect_to=None, headers=None):
        response = MagicMock()
        response.__enter__.return_value = response
        response.is_redirect = redirect_to is not None
        response.is_permanent_redirect = False
        response.headers = {"Location": redirect_to} if redirect_to else (headers or {})
        response.iter_content.return_value = iter([body or self.payload])
        return response

    def _download(self, responses, url="https://example.com/weights.bin", **overrides):
        options = dict(expected_sha256=self.digest, allowed_hosts={"example.com", "cdn.example.com"}, max_bytes=1000)
        options.update(overrides)
        with patch.object(downloader.requests, "get", side_effect=responses) as get:
            download_file(url, self.destination, **options)
        return get

    def _leftovers(self):
        directory = os.path.dirname(self.destination)
        return os.listdir(directory) if os.path.isdir(directory) else []

    def test_downloads_and_verifies(self):
        self._download([self._response()])
        with open(self.destination, "rb") as f:
            self.assertEqual(f.read(), self.payload)
        self.assertEqual(self._leftovers(), ["weights.bin"])

    def test_follows_allowlisted_redirects_only(self):
        get = self._download([self._response(redirect_to="https://cdn.example.com/w.bin"), self._response()])
        self.assertEqual(get.call_args_list[1].args[0], "https://cdn.example.com/w.bin")

        with self.assertRaises(ValueError), self.assertLogs(level="WARNING"):
            self._download([self._response(redirect_to="https://evil.example.net/w.bin")])
        with self.assertRaises(ValueError), self.assertLogs(level="WARNING"):
            self._download([self._response(redirect_to="https://example.com/again")] * 10)

    def test_rejects_untrusted_origins(self):
        for url in ("http://example.com/weights.bin", "https://other.example.org/weights.bin"):
            with self.subTest(url=url), self.assertRaises(ValueError), self.assertLogs(level="WARNING"):
                self._download([self._response()], url=url)

    def test_rejects_wrong_digest_and_oversized_bodies_without_leaving_files(self):
        cases = [
            dict(expected_sha256="0" * 64),
            dict(max_bytes=10),
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError), self.assertLogs(level="WARNING"):
                self._download([self._response()], **overrides)
            self.assertEqual(self._leftovers(), [])

        with self.assertRaises(ValueError), self.assertLogs(level="WARNING"):
            self._download([self._response(headers={"Content-Length": "5000"})])

    def test_argument_validation(self):
        for overrides in (dict(max_bytes=0), dict(allowed_hosts=set()), dict(expected_sha256="abc")):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self._download([], **overrides)


if __name__ == "__main__":
    unittest.main()
