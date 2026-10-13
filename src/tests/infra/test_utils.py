"""Tests for the exception hierarchy, configuration loading and small helpers."""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from voxpipe.core import config as config_module
from voxpipe.core.config import get_config, load_config, set_config
from voxpipe.core.exceptions import (
    ASRError,
    ConfigError,
    LLMError,
    ModelError,
    ProviderError,
    StorageError,
    ToolError,
    TTSError,
    VoxpipeError,
)
from voxpipe.core.models import AppConfig
from voxpipe.core.utils import safe_json_loads


class TestSafeJsonLoads(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(safe_json_loads('{"a": 1}', fallback={}), {"a": 1})
        self.assertEqual(safe_json_loads("[1, 2]"), [1, 2])

    def test_json_wrapped_in_prose_or_a_code_fence(self):
        reply = 'Sure! Here it is:\n```json\n{"a": [1, 2]}\n```\nDone.'
        self.assertEqual(safe_json_loads(reply), {"a": [1, 2]})

    def test_unparseable_text_gives_the_fallback(self):
        self.assertEqual(safe_json_loads("{bad}", fallback=[1]), [1])
        self.assertEqual(safe_json_loads("", fallback="x"), "x")
        self.assertEqual(safe_json_loads(None, fallback={}), {})
        self.assertEqual(safe_json_loads("no json here"), [], "the default fallback is an empty list")


class TestExceptions(unittest.TestCase):
    def test_every_error_is_a_voxpipe_error(self):
        for exc in (ASRError, ConfigError, LLMError, ModelError, ProviderError, StorageError, ToolError, TTSError):
            self.assertTrue(issubclass(exc, VoxpipeError), exc.__name__)

    def test_raise_catch(self):
        with self.assertRaises(VoxpipeError) as raised:
            raise LLMError("test")
        self.assertEqual(str(raised.exception), "test")


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Each test sees a clean environment and an empty working directory.
        for patcher in (
            patch.dict(os.environ, {}, clear=True),
            patch.object(config_module.os, "getcwd", return_value=self.tmp.name),
            patch.object(config_module, "load_dotenv"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(set_config, None)

    def _write(self, text, name="config.yaml"):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_packaged_defaults_are_valid(self):
        config = load_config()
        self.assertIsInstance(config, AppConfig)
        self.assertIn(config.llm.model, config.llm.local)
        self.assertNotIn("_gemma4", config.llm.local, "YAML anchors must not show up as models")
        self.assertEqual(config.llm.local["Gemma4E4B"].decoder, "gemma_e2b")
        self.assertEqual(config.llm.local["Gemma4_12B"].decoder, "legacy_xml")
        self.assertEqual(config.llm.local["Gemma4_12B"].n_ctx, 8192)
        self.assertIsNone(config.asr.sound_device)

    def test_user_file_overrides_defaults_key_by_key(self):
        path = self._write("llm:\n  max_turns: 7\n  local:\n    Mine:\n      model_path: m.gguf\nasr:\n  vad_threshold: 0.9\n")
        config = load_config(path)
        self.assertEqual(config.llm.max_turns, 7)
        self.assertEqual(config.llm.local["Mine"].model_path, "m.gguf")
        self.assertIn("Qwen3", config.llm.local, "sibling keys keep their defaults")
        self.assertEqual(config.asr.vad_threshold, 0.9)
        self.assertEqual(config.asr.trailing_silence_ms, 800)

    def test_config_file_resolution_order(self):
        self._write("llm:\n  max_turns: 3\n")  # ./config.yaml
        self.assertEqual(load_config().llm.max_turns, 3)

        from_env = self._write("llm:\n  max_turns: 4\n", "env.yaml")
        with patch.dict(os.environ, {"VOXPIPE_CONFIG": from_env}):
            self.assertEqual(load_config().llm.max_turns, 4)
            explicit = self._write("llm:\n  max_turns: 5\n", "explicit.yaml")
            self.assertEqual(load_config(explicit).llm.max_turns, 5)

    def test_missing_explicit_file_is_an_error(self):
        with self.assertRaises(ConfigError):
            load_config(os.path.join(self.tmp.name, "nope.yaml"))
        with patch.dict(os.environ, {"VOXPIPE_CONFIG": os.path.join(self.tmp.name, "nope.yaml")}):
            with self.assertRaises(ConfigError):
                load_config()

    def test_invalid_files_are_config_errors(self):
        cases = {
            "unknown top-level key": "hotkey:\n  enable: true\n",
            "unknown nested key": "asr:\n  vad_treshold: 0.5\n",
            "out of range": "asr:\n  vad_threshold: 3\n",
            "wrong type": "llm:\n  max_turns: many\n",
            "unknown backend": "llm:\n  backend: cloud\n",
            "reply longer than the context": "llm:\n  local:\n    Mine:\n      n_ctx: 1024\n      max_tokens: 1024\n",
            "not a mapping": "- just\n- a list\n",
            "bad yaml": "llm: [unclosed\n",
        }
        for label, text in cases.items():
            with self.subTest(label), self.assertRaises(ConfigError):
                load_config(self._write(text, "bad.yaml"))

    def test_error_names_the_offending_key(self):
        with self.assertRaisesRegex(ConfigError, "vad_treshold"):
            load_config(self._write("asr:\n  vad_treshold: 0.5\n"))

    def test_empty_user_file_is_fine(self):
        self.assertEqual(load_config(self._write("")).llm.max_turns, 10)

    def test_env_blocks(self):
        path = self._write(
            "llm:\n  litellm:\n    provider: openai\n    model: gpt-test\n    api_base: https://literal.example\n"
            "    env:\n      api_key: MY_KEY\n      api_base: MY_BASE\n"
        )
        with patch.dict(os.environ, {"MY_KEY": "secret"}):
            litellm = load_config(path).llm.litellm
        self.assertEqual(litellm.api_key, "secret")
        self.assertEqual(litellm.api_base, "https://literal.example", "an unset variable leaves the literal value")

        self.assertIsNone(load_config(path).llm.litellm.api_key, "unset and no literal: the field is empty")

    def test_get_config_caches_and_set_config_replaces(self):
        first = get_config()
        self.assertIs(get_config(), first)

        custom = first.model_copy(update={"llm": first.llm.model_copy(update={"max_turns": 99})})
        set_config(custom)
        self.assertIs(get_config(), custom)

        set_config(None)
        self.assertEqual(get_config().llm.max_turns, 10)

class TestImportSideEffects(unittest.TestCase):
    def test_importing_voxpipe_reads_no_configuration(self):
        program = (
            "import voxpipe, voxpipe.llm.model, voxpipe.pipeline.pipeline, voxpipe.core.config as config\n"
            "assert config._config is None, 'configuration was loaded at import time'\n"
        )
        # Run from a directory holding a broken config.yaml: importing must not even look at it.
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.yaml"), "w", encoding="utf-8") as f:
                f.write("this: is: not: valid: yaml: [")
            result = subprocess.run(
                [sys.executable, "-c", program],
                cwd=tmp,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
