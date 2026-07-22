"""Tests for AudioPlayer initialization and state."""
import unittest
from unittest.mock import MagicMock, PropertyMock, patch


class TestAudioPlayer(unittest.TestCase):
    """AudioPlayer initialization and state."""

    @patch("voxpipe.tts.audio.sd.query_devices")
    @patch("voxpipe.tts.audio.sd.InputStream")
    @patch("voxpipe.tts.audio.sd.OutputStream")
    @patch("voxpipe.tts.audio.sd.default.device", new_callable=PropertyMock)
    def test_audio_player_init(self, mock_dev, mock_out, mock_in, mock_qd):
        from voxpipe.tts.audio import AudioPlayer
        mock_dev.return_value = (0, 1)
        mock_qd.side_effect = lambda d: {"name": "test", "max_output_channels": 2}
        mock_out.return_value = MagicMock()
        ap = AudioPlayer()
        self.assertIsNotNone(ap)
        ap.stop()

    @patch("voxpipe.tts.audio.sd.query_devices")
    @patch("voxpipe.tts.audio.sd.InputStream")
    @patch("voxpipe.tts.audio.sd.OutputStream")
    @patch("voxpipe.tts.audio.sd.default.device", new_callable=PropertyMock)
    def test_stop_playback_clears_queue(self, mock_dev, mock_out, mock_in, mock_qd):
        from voxpipe.tts.audio import AudioPlayer
        mock_dev.return_value = (0, 1)
        mock_qd.side_effect = lambda d: {"name": "test", "max_output_channels": 2}
        mock_out.return_value = MagicMock()
        ap = AudioPlayer()
        old_gen = ap._gen
        ap.stop_playback()
        self.assertGreater(ap._gen, old_gen)
        ap.stop()
