"""
Unit tests for legacy file cleanup in ensure_config_dir().

Verifies that PID_FILE, PROCESSING_FILE, and AUDIO_PATH_FILE are
deleted on startup when they exist (CRITIC-R2-M1).
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input
import state_db as _state_db


class TestLegacyFileCleanup:
    """Test ensure_config_dir() cleans up legacy state files."""

    def test_deletes_pid_file(self, tmp_path, monkeypatch):
        """ensure_config_dir should delete legacy PID file."""
        config_dir = tmp_path / "config" / "voice-input"
        config_dir.mkdir(parents=True)
        pid_file = config_dir / "recording.pid"
        pid_file.write_text("12345")
        db_path = config_dir / "state.db"

        monkeypatch.setattr("voice_input.CONFIG_DIR", config_dir)
        monkeypatch.setattr("voice_input.PID_FILE", pid_file)
        monkeypatch.setattr("voice_input.PROCESSING_FILE", config_dir / "processing.flag")
        monkeypatch.setattr("voice_input.AUDIO_PATH_FILE", config_dir / "recording_path.txt")
        monkeypatch.setattr("voice_input.STATE_DB_PATH", db_path)

        voice_input.ensure_config_dir()

        assert not pid_file.exists()

    def test_deletes_processing_file(self, tmp_path, monkeypatch):
        """ensure_config_dir should delete legacy processing flag file."""
        config_dir = tmp_path / "config" / "voice-input"
        config_dir.mkdir(parents=True)
        processing_file = config_dir / "processing.flag"
        processing_file.touch()
        db_path = config_dir / "state.db"

        monkeypatch.setattr("voice_input.CONFIG_DIR", config_dir)
        monkeypatch.setattr("voice_input.PID_FILE", config_dir / "recording.pid")
        monkeypatch.setattr("voice_input.PROCESSING_FILE", processing_file)
        monkeypatch.setattr("voice_input.AUDIO_PATH_FILE", config_dir / "recording_path.txt")
        monkeypatch.setattr("voice_input.STATE_DB_PATH", db_path)

        voice_input.ensure_config_dir()

        assert not processing_file.exists()

    def test_deletes_audio_path_file(self, tmp_path, monkeypatch):
        """ensure_config_dir should delete legacy audio path file."""
        config_dir = tmp_path / "config" / "voice-input"
        config_dir.mkdir(parents=True)
        audio_path_file = config_dir / "recording_path.txt"
        audio_path_file.write_text("/tmp/recording.wav")
        db_path = config_dir / "state.db"

        monkeypatch.setattr("voice_input.CONFIG_DIR", config_dir)
        monkeypatch.setattr("voice_input.PID_FILE", config_dir / "recording.pid")
        monkeypatch.setattr("voice_input.PROCESSING_FILE", config_dir / "processing.flag")
        monkeypatch.setattr("voice_input.AUDIO_PATH_FILE", audio_path_file)
        monkeypatch.setattr("voice_input.STATE_DB_PATH", db_path)

        voice_input.ensure_config_dir()

        assert not audio_path_file.exists()

    def test_deletes_all_legacy_files(self, tmp_path, monkeypatch):
        """ensure_config_dir should delete all three legacy files when present."""
        config_dir = tmp_path / "config" / "voice-input"
        config_dir.mkdir(parents=True)

        pid_file = config_dir / "recording.pid"
        processing_file = config_dir / "processing.flag"
        audio_path_file = config_dir / "recording_path.txt"

        pid_file.write_text("12345")
        processing_file.touch()
        audio_path_file.write_text("/tmp/recording.wav")
        db_path = config_dir / "state.db"

        monkeypatch.setattr("voice_input.CONFIG_DIR", config_dir)
        monkeypatch.setattr("voice_input.PID_FILE", pid_file)
        monkeypatch.setattr("voice_input.PROCESSING_FILE", processing_file)
        monkeypatch.setattr("voice_input.AUDIO_PATH_FILE", audio_path_file)
        monkeypatch.setattr("voice_input.STATE_DB_PATH", db_path)

        voice_input.ensure_config_dir()

        assert not pid_file.exists()
        assert not processing_file.exists()
        assert not audio_path_file.exists()

    def test_no_error_when_files_absent(self, tmp_path, monkeypatch):
        """ensure_config_dir should not fail when legacy files don't exist."""
        config_dir = tmp_path / "config" / "voice-input"
        config_dir.mkdir(parents=True)
        db_path = config_dir / "state.db"

        monkeypatch.setattr("voice_input.CONFIG_DIR", config_dir)
        monkeypatch.setattr("voice_input.PID_FILE", config_dir / "recording.pid")
        monkeypatch.setattr("voice_input.PROCESSING_FILE", config_dir / "processing.flag")
        monkeypatch.setattr("voice_input.AUDIO_PATH_FILE", config_dir / "recording_path.txt")
        monkeypatch.setattr("voice_input.STATE_DB_PATH", db_path)

        # Should not raise
        voice_input.ensure_config_dir()

    def test_initializes_db(self, tmp_path, monkeypatch):
        """ensure_config_dir should initialize the state DB."""
        config_dir = tmp_path / "config" / "voice-input"
        config_dir.mkdir(parents=True)
        db_path = config_dir / "state.db"

        monkeypatch.setattr("voice_input.CONFIG_DIR", config_dir)
        monkeypatch.setattr("voice_input.PID_FILE", config_dir / "recording.pid")
        monkeypatch.setattr("voice_input.PROCESSING_FILE", config_dir / "processing.flag")
        monkeypatch.setattr("voice_input.AUDIO_PATH_FILE", config_dir / "recording_path.txt")
        monkeypatch.setattr("voice_input.STATE_DB_PATH", db_path)

        voice_input.ensure_config_dir()

        # DB should be initialized and readable
        state = _state_db.get_state(db_path)
        assert state["status"] == "idle"
