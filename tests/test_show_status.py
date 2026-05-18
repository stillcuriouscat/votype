"""
Unit tests for show_status() — DB-based status display.

Verifies show_status() reads status, daemon state, and post-processor
from SQLite DB instead of is_recording() + IPC calls.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input
import state_db as _state_db


class TestShowStatusDB:
    """Test show_status() reads from DB."""

    def test_shows_recording_from_db(self, isolated_environment, capsys):
        """show_status should show Recording: Yes when DB status is 'recording'."""
        _state_db.update_state(
            isolated_environment["state_db_path"],
            status="recording",
            recording_pid=12345,
        )

        with patch("voice_input.is_daemon_running", return_value=False):
            voice_input.show_status()

        captured = capsys.readouterr()
        assert "Recording: Yes" in captured.out

    def test_shows_not_recording_from_db(self, isolated_environment, capsys):
        """show_status should show Recording: No when DB status is 'idle'."""
        with patch("voice_input.is_daemon_running", return_value=False):
            voice_input.show_status()

        captured = capsys.readouterr()
        assert "Recording: No" in captured.out

    def test_shows_daemon_running(self, isolated_environment, capsys):
        """show_status should show Daemon: Running when daemon is running."""
        with patch("voice_input.is_daemon_running", return_value=True):
            with patch("voice_input.send_to_daemon") as mock_send:
                mock_send.return_value = {
                    "model": "sensevoice",
                    "name": "SenseVoice",
                    "description": "Test model",
                }
                voice_input.show_status()

        captured = capsys.readouterr()
        assert "Daemon: Running" in captured.out
        assert "SenseVoice" in captured.out

    def test_shows_daemon_not_running(self, isolated_environment, capsys):
        """show_status should show Daemon: Not running."""
        with patch("voice_input.is_daemon_running", return_value=False):
            voice_input.show_status()

        captured = capsys.readouterr()
        assert "Daemon: Not running" in captured.out

    def test_shows_post_processor_from_db(self, isolated_environment, capsys):
        """show_status should show post-processor from DB."""
        _state_db.update_state(
            isolated_environment["state_db_path"],
            post_processor="gemini-fix",
        )

        with patch("voice_input.is_daemon_running", return_value=False):
            voice_input.show_status()

        captured = capsys.readouterr()
        assert "Post-processor:" in captured.out
        assert "gemini-fix" in captured.out

    def test_shows_default_post_processor(self, isolated_environment, capsys):
        """show_status should show 'claude-merge' post-processor for fresh DB."""
        with patch("voice_input.is_daemon_running", return_value=False):
            voice_input.show_status()

        captured = capsys.readouterr()
        assert "Post-processor:" in captured.out
        assert "claude-merge" in captured.out

    def test_no_ipc_for_post_processor(self, isolated_environment, capsys):
        """show_status should NOT call send_to_daemon for post-processor info."""
        _state_db.update_state(
            isolated_environment["state_db_path"],
            post_processor="claude-merge",
        )

        with patch("voice_input.is_daemon_running", return_value=True):
            with patch("voice_input.send_to_daemon") as mock_send:
                mock_send.return_value = {
                    "model": "sensevoice",
                    "name": "SenseVoice",
                    "description": "Test",
                }
                voice_input.show_status()

                # Should NOT have called get_post_processor via IPC
                for c in mock_send.call_args_list:
                    assert c[0][0] != "get_post_processor"

        captured = capsys.readouterr()
        assert "claude-merge" in captured.out

    def test_daemon_not_responsive(self, isolated_environment, capsys):
        """show_status should handle unresponsive daemon gracefully."""
        with patch("voice_input.is_daemon_running", return_value=True):
            with patch("voice_input.send_to_daemon", return_value=None):
                voice_input.show_status()

        captured = capsys.readouterr()
        assert "Not responsive" in captured.out

    def test_shows_configured_model_when_daemon_down(self, isolated_environment, capsys):
        """show_status should show configured model when daemon is not running."""
        with patch("voice_input.is_daemon_running", return_value=False):
            voice_input.show_status()

        captured = capsys.readouterr()
        assert "Configured Model:" in captured.out
