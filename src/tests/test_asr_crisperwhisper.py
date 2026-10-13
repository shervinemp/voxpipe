from unittest.mock import MagicMock, patch

import numpy as np

from voxpipe.asr.models import CrisperWhisper
from voxpipe.core.models import CrisperWhisperConfig


def _vad():
    vad = MagicMock()
    vad.sample_rate = 16000
    vad.hop_size = 512
    vad.__iter__.return_value = iter([np.zeros(16000, dtype=np.float32)])
    return vad


@patch("voxpipe.asr.models.base.sd")
def test_crisperwhisper_transcribe(_sound_device):
    model = MagicMock()
    model.transcribe.return_value.text = "this is a test sentence"

    asr = CrisperWhisper(vad=_vad(), model=model)

    assert list(asr) == ["this is a test sentence"]
    model.transcribe.assert_called_once()
    kwargs = model.transcribe.call_args.kwargs
    assert kwargs["sr"] == 16000
    assert kwargs["mode"] == "verbatim"
    assert kwargs["language"] == "en"


@patch("voxpipe.asr.models.base.sd")
def test_crisperwhisper_loads_the_configured_model(_sound_device):
    fake_package = MagicMock()
    settings = CrisperWhisperConfig(model="nyralabs/CrisperWhisper2.0_large", backend="transformers", mode="intended")

    with patch.dict("sys.modules", {"crisperwhisper": fake_package}):
        asr = CrisperWhisper(vad=_vad(), settings=settings)
        list(asr)

    fake_package.CrisperWhisperModel.assert_called_once_with("nyralabs/CrisperWhisper2.0_large", backend="transformers")
    assert fake_package.CrisperWhisperModel.return_value.transcribe.call_args.kwargs["mode"] == "intended"


@patch("voxpipe.asr.models.base.sd")
def test_parakeet_warms_up_and_transcribes(_sound_device):
    from voxpipe.asr.models import ParakeetV2

    model = MagicMock()
    model.recognize.return_value = "hello there"

    asr = ParakeetV2(vad=_vad(), model=model)

    assert model.recognize.call_count == 1, "one warm-up pass at construction"
    assert list(asr) == ["hello there"]
    assert model.recognize.call_args.kwargs == {"sample_rate": 16000}
