"""Tests for AudioRecorder, including the proportional wait() timeout."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.capture.audio_recorder import AudioRecorder


class TestAudioRecorderTimeout:
    """wait() must use duration + 5 s, not a fixed 30 s."""

    def _recorder_with_mock_process(self, tmp_path, duration=None):
        recorder = AudioRecorder()
        recorder._output_path = tmp_path / "test.wav"
        recorder._output_path.write_bytes(b"RIFF....fake wav")
        recorder._duration = duration
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = None
        recorder._process = mock_proc
        return recorder, mock_proc

    def test_timeout_proportional_to_duration(self, tmp_path):
        """Long recording: timeout = duration + 5."""
        recorder, mock_proc = self._recorder_with_mock_process(tmp_path, duration=60.0)
        recorder.wait()
        mock_proc.wait.assert_called_once_with(timeout=65.0)

    def test_timeout_short_recording(self, tmp_path):
        """Short recording: timeout = duration + 5."""
        recorder, mock_proc = self._recorder_with_mock_process(tmp_path, duration=5.0)
        recorder.wait()
        mock_proc.wait.assert_called_once_with(timeout=10.0)

    def test_timeout_fallback_when_duration_not_set(self, tmp_path):
        """No duration stored: falls back to 30 s."""
        recorder, mock_proc = self._recorder_with_mock_process(tmp_path, duration=None)
        recorder.wait()
        mock_proc.wait.assert_called_once_with(timeout=30)

    def test_start_stores_duration(self, tmp_path):
        """start() must persist duration so wait() can use it."""
        recorder = AudioRecorder()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            recorder.start(tmp_path / "out.wav", duration=45.0)
        assert recorder._duration == 45.0

    def test_duration_cleared_after_wait(self, tmp_path):
        """_duration is reset to None after wait() completes."""
        recorder, _ = self._recorder_with_mock_process(tmp_path, duration=10.0)
        recorder.wait()
        assert recorder._duration is None

    def test_duration_cleared_on_timeout(self, tmp_path):
        """_duration is also cleared when the process times out."""
        import subprocess
        recorder = AudioRecorder()
        recorder._output_path = tmp_path / "test.wav"
        recorder._duration = 5.0
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10)
        recorder._process = mock_proc

        result = recorder.wait()

        assert result is None
        assert recorder._duration is None
