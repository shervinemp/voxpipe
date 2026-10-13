import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np

from voxpipe.asr.models import CrisperWhisper

fake_crisperwhisper = types.ModuleType("crisperwhisper")
fake_crisperwhisper.CrisperWhisperModel = MagicMock()
sys.modules["crisperwhisper"] = fake_crisperwhisper


def _segment():
    return np.zeros(16000, dtype=np.float32)


@patch("voxpipe.asr.models.crisperwhisper.CrisperWhisper._inputstream")
@patch("voxpipe.asr.models.crisperwhisper.Silero")
def test_crisperwhisper_transcribe(mock_silero, mock_input_stream):
    mock_vad = MagicMock()
    mock_vad._model.SAMPLE_RATE = 16000
    mock_vad.__iter__.return_value = iter([_segment()])
    mock_silero.return_value = mock_vad
    mock_input_stream.return_value = MagicMock()

    result = MagicMock()
    result.text = "this is a test sentence"
    model_mock = fake_crisperwhisper.CrisperWhisperModel.return_value
    model_mock.transcribe.return_value = result

    asr = CrisperWhisper()
    text = next(asr._produce())

    assert text == "this is a test sentence"
    fake_crisperwhisper.CrisperWhisperModel.assert_called_once()
    assert (
        fake_crisperwhisper.CrisperWhisperModel.call_args.args[0]
        == "nyralabs/CrisperWhisper2.0_small"
    )
    model_mock.transcribe.assert_called_once()
    kwargs = model_mock.transcribe.call_args.kwargs
    assert kwargs["sr"] == 16000
    assert kwargs["mode"] == "verbatim"
    assert kwargs["language"] == "en"


@patch("voxpipe.asr.models.crisperwhisper.CrisperWhisper._inputstream")
@patch("voxpipe.asr.models.crisperwhisper.Silero")
def test_crisperwhisper_registered(mock_silero, mock_input_stream):
    from voxpipe.asr.model import ASRProviders

    assert ASRProviders.CrisperWhisper is CrisperWhisper
