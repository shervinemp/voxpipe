"""Tests for LLM provider registry and model manager."""
import os
import unittest


class TestLLMProvidersRegistry(unittest.TestCase):
    def test_get_known(self):
        from voxpipe.llm.model import LLMProviders
        self.assertIsNotNone(LLMProviders.get("Qwen3"))

    def test_get_unknown_raises(self):
        from voxpipe.llm.model import LLMProviders, ProviderError
        with self.assertRaises(ProviderError):
            LLMProviders.get("nonexistent")

    def test_get_empty_raises(self):
        from voxpipe.llm.model import LLMProviders, ProviderError
        with self.assertRaises(ProviderError):
            LLMProviders.get("")

    def test_registry_contains_expected(self):
        from voxpipe.llm.model import LLMProviders
        for name in ["Qwen3", "Gemma4E2B", "Gemma4_12B", "Gemma4E4B", "LiteLLM"]:
            self.assertIsNotNone(LLMProviders.get(name), f"Missing: {name}")


class TestModelManager(unittest.TestCase):
    def test_ensure_downloaded_unknown(self):
        from voxpipe.storage.manager import ensure_downloaded
        from voxpipe.core.exceptions import ModelError
        with self.assertRaises(ModelError):
            ensure_downloaded("nonexistent")

    def test_manifest_path_exists(self):
        from voxpipe.storage.manager import _MANIFEST_PATH
        self.assertTrue(os.path.exists(_MANIFEST_PATH))

    def test_manifest_yaml_loads(self):
        import yaml
        from voxpipe.storage.manager import _MANIFEST_PATH
        with open(_MANIFEST_PATH) as f:
            data = yaml.safe_load(f)
        self.assertIn("Gemma4E4B", data)
        self.assertIn("repo", data["Gemma4E4B"])
