"""
Unit Tests - Test behavior of individual functions.

Coverage:
- Configuration and path functions
- Process status check functions
- Daemon readiness check (core of Race Condition fix)
- Socket communication functions
- Text input functions
- Notification functions
"""

import os
import sys
import json
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input


# ============ 1. Configuration and Path Functions ============

class TestEnsureConfigDir:
    """Test ensure_config_dir function."""

    def test_creates_directory_when_not_exists(self, tmp_path, monkeypatch):
        """Should create config directory when it does not exist."""
        config_dir = tmp_path / "new_config" / "voice-input"
        monkeypatch.setattr('voice_input.CONFIG_DIR', config_dir)

        assert not config_dir.exists()
        voice_input.ensure_config_dir()
        assert config_dir.exists()

    def test_no_error_when_exists(self, tmp_path, monkeypatch):
        """Should not raise an error when config directory already exists."""
        config_dir = tmp_path / "existing_config" / "voice-input"
        config_dir.mkdir(parents=True)
        monkeypatch.setattr('voice_input.CONFIG_DIR', config_dir)

        # Should not raise an exception
        voice_input.ensure_config_dir()
        assert config_dir.exists()


class TestGetDaemonPaths:
    """Test get_daemon_paths function."""

    def test_returns_venv_paths_when_exists(self, tmp_path, monkeypatch):
        """Should return venv paths when venv exists."""
        venv_python = tmp_path / ".local" / "share" / "voice-input" / "venv" / "bin" / "python"
        script_path = tmp_path / ".local" / "share" / "voice-input" / "voice_input.py"

        venv_python.parent.mkdir(parents=True)
        venv_python.touch()
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.touch()

        # Mock Path.home() to return tmp_path
        monkeypatch.setattr('voice_input.Path.home', lambda: tmp_path)

        python_path, script = voice_input.get_daemon_paths()

        assert python_path == venv_python
        assert script == script_path

    def test_fallback_to_sys_executable(self, tmp_path, monkeypatch):
        """Should use sys.executable when venv does not exist."""
        # Mock Path.home() to return empty directory
        monkeypatch.setattr('voice_input.Path.home', lambda: tmp_path)

        python_path, script = voice_input.get_daemon_paths()

        assert python_path == Path(sys.executable)


class TestGetIconsDir:
    """Test get_icons_dir function."""

    def test_returns_install_location_when_exists(self, tmp_path, monkeypatch):
        """Should return install location when install directory exists."""
        install_icons = tmp_path / ".local" / "share" / "voice-input" / "icons"
        install_icons.mkdir(parents=True)

        monkeypatch.setattr('os.path.expanduser',
                          lambda x: str(tmp_path) + x[1:] if x.startswith("~") else x)
        monkeypatch.setattr('os.path.isdir', lambda x: x == str(install_icons))

        result = voice_input.get_icons_dir()
        assert result == str(install_icons)

    def test_fallback_to_script_location(self, tmp_path, monkeypatch):
        """Should fall back to script directory when install directory does not exist."""
        monkeypatch.setattr('os.path.expanduser',
                          lambda x: str(tmp_path / "nonexistent"))
        monkeypatch.setattr('os.path.isdir', lambda x: False)

        result = voice_input.get_icons_dir()
        # Should return the icons directory alongside the script
        assert result.endswith("icons")


# ============ 2. Process Status Check Functions ============

class TestIsRecording:
    """Test is_recording function — now DB-based."""

    def test_returns_false_when_idle(self, tmp_path, monkeypatch):
        """Should return False when DB status is idle."""
        import state_db as _state_db
        db_path = tmp_path / "state.db"
        monkeypatch.setattr('voice_input.STATE_DB_PATH', db_path)
        monkeypatch.setattr('state_db.DEFAULT_DB_PATH', db_path)
        _state_db.init_db(db_path)

        assert voice_input.is_recording() is False

    def test_returns_true_when_recording_with_live_pid(self, tmp_path, monkeypatch):
        """Should return True when DB status is recording and PID is alive."""
        import state_db as _state_db
        db_path = tmp_path / "state.db"
        monkeypatch.setattr('voice_input.STATE_DB_PATH', db_path)
        monkeypatch.setattr('state_db.DEFAULT_DB_PATH', db_path)
        _state_db.init_db(db_path)
        _state_db.update_state(db_path, status="recording", recording_pid=os.getpid(), recording_path="/tmp/test.wav")

        assert voice_input.is_recording() is True


class TestIsDaemonRunning:
    """Test is_daemon_running function."""

    def test_returns_false_when_no_pid_or_lock(self, tmp_path, monkeypatch):
        """Should return False when no PID file and no lock held."""
        daemon_pid_file = tmp_path / "daemon.pid"
        daemon_lock_file = tmp_path / "daemon.lock"
        monkeypatch.setattr('voice_input.DAEMON_PID_FILE', daemon_pid_file)
        monkeypatch.setattr('voice_input.DAEMON_LOCK_FILE', daemon_lock_file)

        # No PID file, no lock
        assert voice_input.is_daemon_running() is False

    def test_returns_true_when_lock_held(self, tmp_path, monkeypatch):
        """Should return True when flock is held by another process."""
        daemon_lock_file = tmp_path / "daemon.lock"
        monkeypatch.setattr('voice_input.DAEMON_LOCK_FILE', daemon_lock_file)
        monkeypatch.setattr('voice_input._is_daemon_lock_held', lambda: True)

        assert voice_input.is_daemon_running() is True


# ============ 3. Daemon Readiness Check (Race Condition Core) ============

class TestIsDaemonReady:
    """
    Test is_daemon_ready function.
    This is the core function for fixing the race condition.
    """

    def test_returns_false_when_daemon_not_running(self, tmp_path, monkeypatch):
        """Should return False when daemon is not running."""
        daemon_pid_file = tmp_path / "daemon.pid"
        daemon_lock_file = tmp_path / "daemon.lock"
        monkeypatch.setattr('voice_input.DAEMON_PID_FILE', daemon_pid_file)
        monkeypatch.setattr('voice_input.DAEMON_LOCK_FILE', daemon_lock_file)

        # No PID file
        assert voice_input.is_daemon_ready() is False

    def test_returns_false_when_running_but_not_responsive(self, tmp_path, monkeypatch):
        """
        Should return falsy when PID file exists but ping fails.
        This is the race condition scenario: daemon just started, model still loading.
        """
        daemon_pid_file = tmp_path / "daemon.pid"
        daemon_pid_file.write_text(str(os.getpid()))  # Pretend daemon is running
        monkeypatch.setattr('voice_input.DAEMON_PID_FILE', daemon_pid_file)

        # Mock send_to_daemon to return None (no response)
        monkeypatch.setattr('voice_input.send_to_daemon', lambda cmd: None)

        # Should return falsy (None or False)
        assert not voice_input.is_daemon_ready()

    def test_returns_false_when_ping_returns_error(self, tmp_path, monkeypatch):
        """Should return False when ping returns an error."""
        daemon_pid_file = tmp_path / "daemon.pid"
        daemon_pid_file.write_text(str(os.getpid()))
        monkeypatch.setattr('voice_input.DAEMON_PID_FILE', daemon_pid_file)

        # Mock send_to_daemon to return an error
        monkeypatch.setattr('voice_input.send_to_daemon',
                          lambda cmd: {"error": "connection refused"})

        assert voice_input.is_daemon_ready() is False

    def test_returns_true_when_fully_ready(self, tmp_path, monkeypatch):
        """Should return True when daemon responds to ping."""
        daemon_pid_file = tmp_path / "daemon.pid"
        daemon_pid_file.write_text(str(os.getpid()))
        monkeypatch.setattr('voice_input.DAEMON_PID_FILE', daemon_pid_file)

        # Mock send_to_daemon to return success
        monkeypatch.setattr('voice_input.send_to_daemon',
                          lambda cmd: {"status": "ok"})

        assert voice_input.is_daemon_ready() is True


# ============ 4. Socket Communication Functions ============

class TestSendToDaemon:
    """Test send_to_daemon function."""

    def test_returns_none_when_socket_not_exists(self, tmp_path, monkeypatch):
        """Should return None when socket file does not exist."""
        socket_path = tmp_path / "nonexistent.sock"
        monkeypatch.setattr('voice_input.SOCKET_PATH', socket_path)

        result = voice_input.send_to_daemon("ping")
        assert result is None

    def test_returns_error_on_connection_refused(self, tmp_path, monkeypatch):
        """Should return error when connection is refused."""
        # Create a path that exists but has no listener
        socket_path = tmp_path / "refused.sock"
        socket_path.touch()  # File exists but is not a real socket
        monkeypatch.setattr('voice_input.SOCKET_PATH', socket_path)

        result = voice_input.send_to_daemon("ping")
        assert result is not None
        assert "error" in result

    def test_successful_communication(self, mock_socket_server, monkeypatch):
        """Should return response on successful communication."""
        monkeypatch.setattr('voice_input.SOCKET_PATH',
                          mock_socket_server['socket_path'])

        result = voice_input.send_to_daemon("ping")

        assert result == {"status": "ok"}

    def test_transcribe_command(self, mock_socket_server, monkeypatch):
        """Transcribe command should return text."""
        monkeypatch.setattr('voice_input.SOCKET_PATH',
                          mock_socket_server['socket_path'])

        result = voice_input.send_to_daemon("transcribe", "/path/to/audio.wav")

        assert result == {"text": "mock transcription result"}


# ============ 5. Text Input Functions ============

class TestClipboardFunctions:
    """Test clipboard functions."""

    def test_clipboard_get_calls_xclip(self):
        """clipboard_get should call xclip."""
        with patch('voice_input.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stdout="clipboard content")

            result = voice_input.clipboard_get()

            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "xclip" in args
            assert "-selection" in args
            assert "clipboard" in args
            assert result == "clipboard content"

    def test_clipboard_set_calls_xclip(self):
        """clipboard_set should call xclip."""
        with patch('voice_input.subprocess.Popen') as mock_popen:
            mock_proc = MagicMock()
            mock_popen.return_value = mock_proc

            voice_input.clipboard_set("test text")

            mock_popen.assert_called_once()
            mock_proc.communicate.assert_called_once()
            # Check input is UTF-8 encoded
            call_args = mock_proc.communicate.call_args
            assert call_args[1]['input'] == b"test text"


class TestTypeText:
    """Test type_text function."""

    def test_empty_text_does_nothing(self):
        """Empty text should not perform any operation."""
        with patch('voice_input.clipboard_get') as mock_get:
            with patch('voice_input.clipboard_set') as mock_set:
                voice_input.type_text("")

                mock_get.assert_not_called()
                mock_set.assert_not_called()

    def test_restores_clipboard_after_typing(self):
        """Should restore original clipboard content after typing."""
        original_content = "original clipboard"

        with patch('voice_input.clipboard_get', return_value=original_content):
            with patch('voice_input.clipboard_set') as mock_set:
                with patch('voice_input.subprocess.run'):
                    with patch('voice_input.time.sleep'):
                        voice_input.type_text("new text")

                        # Should be called twice: set new text, restore original
                        assert mock_set.call_count == 2
                        calls = mock_set.call_args_list
                        assert calls[0] == call("new text")
                        assert calls[1] == call(original_content)

    def test_handles_xdotool_not_found(self):
        """Should catch exception when xdotool is not installed."""
        with patch('voice_input.clipboard_get', return_value=""):
            with patch('voice_input.clipboard_set'):
                with patch('voice_input.subprocess.run',
                          side_effect=FileNotFoundError("xdotool not found")):
                    with patch('voice_input.time.sleep'):
                        # Should not raise an exception
                        voice_input.type_text("test")


# ============ 6. Notification Functions ============

class TestNotify:
    """Test notify function."""

    def test_calls_notify_send(self):
        """Should call notify-send."""
        with patch('voice_input.subprocess.Popen') as mock_popen:
            voice_input.notify("Title", "Message")

            mock_popen.assert_called_once()
            args = mock_popen.call_args[0][0]
            assert "notify-send" in args
            assert "Title" in args
            assert "Message" in args

    def test_handles_notify_send_not_installed(self):
        """Should fail silently when notify-send is not installed."""
        with patch('voice_input.subprocess.Popen',
                  side_effect=FileNotFoundError("notify-send not found")):
            # Should not raise an exception
            voice_input.notify("Title", "Message")

    def test_respects_urgency_parameter(self):
        """Should pass urgency parameter."""
        with patch('voice_input.subprocess.Popen') as mock_popen:
            voice_input.notify("Title", "Message", urgency="critical")

            args = mock_popen.call_args[0][0]
            assert "-u" in args
            urgency_idx = args.index("-u")
            assert args[urgency_idx + 1] == "critical"


# ============ 7. STATUS_CONFIG Tests ============

class TestStatusConfig:
    """Test status configuration."""

    def test_all_statuses_defined(self):
        """All statuses should be defined."""
        required_statuses = ["idle", "recording", "processing"]
        for status in required_statuses:
            assert status in voice_input.STATUS_CONFIG

    def test_status_config_format(self):
        """Status config format should be correct."""
        for status, config in voice_input.STATUS_CONFIG.items():
            assert len(config) == 3  # (icon_name, tooltip, menu_label)
            icon_name, tooltip, label = config
            assert isinstance(icon_name, str)
            assert isinstance(tooltip, str)
            assert isinstance(label, str)


# ============ 8. Model Configuration Function Tests ============

class TestModelPresets:
    """Test model preset configuration."""

    def test_all_presets_defined(self):
        """All preset models should be defined."""
        # Note: fun-asr-nano temporarily removed due to FunASR bug #2757
        required_models = ["paraformer", "sensevoice"]
        for model_id in required_models:
            assert model_id in voice_input.MODEL_PRESETS

    def test_preset_format(self):
        """Preset format should be correct."""
        for model_id, preset in voice_input.MODEL_PRESETS.items():
            assert "name" in preset
            assert "description" in preset
            assert "framework" in preset  # Added: framework field
            assert "config" in preset
            assert isinstance(preset["name"], str)
            assert isinstance(preset["description"], str)
            assert isinstance(preset["framework"], str)
            assert isinstance(preset["config"], dict)

    def test_default_model_exists(self):
        """Default model should exist in presets."""
        assert voice_input.DEFAULT_MODEL in voice_input.MODEL_PRESETS


class TestGetCurrentModel:
    """Test get_current_model function - single model architecture: always returns default model."""

    def test_returns_default_when_no_state_file(self, tmp_path, monkeypatch):
        """Single model architecture: always returns default model."""
        result = voice_input.get_current_model()
        assert result == voice_input.DEFAULT_MODEL
        assert result == "sensevoice"

    def test_returns_saved_model(self, tmp_path, monkeypatch):
        """Single model architecture: returns default model even when state file exists."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        state_file = config_dir / "current_model.txt"
        state_file.write_text("sensevoice")
        monkeypatch.setattr('voice_input.CONFIG_DIR', config_dir)
        monkeypatch.setattr('voice_input.MODEL_STATE_FILE', state_file)

        result = voice_input.get_current_model()
        # Single model architecture: no longer reads state file, always returns default model
        assert result == voice_input.DEFAULT_MODEL

    def test_returns_default_for_invalid_model(self, tmp_path, monkeypatch):
        """Single model architecture: always returns default model."""
        result = voice_input.get_current_model()
        assert result == voice_input.DEFAULT_MODEL


class TestSetCurrentModel:
    """Test set_current_model function - single model architecture: no longer saves model state."""

    def test_saves_valid_model(self, tmp_path, monkeypatch):
        """Single model architecture: no longer saves model state."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        state_file = config_dir / "current_model.txt"
        monkeypatch.setattr('voice_input.CONFIG_DIR', config_dir)
        monkeypatch.setattr('voice_input.MODEL_STATE_FILE', state_file)

        voice_input.set_current_model("sensevoice")

        # Single model architecture: no longer writes state file
        assert not state_file.exists()

    def test_rejects_invalid_model(self, tmp_path, monkeypatch):
        """Single model architecture: no longer validates models, does nothing."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        state_file = config_dir / "current_model.txt"
        monkeypatch.setattr('voice_input.CONFIG_DIR', config_dir)
        monkeypatch.setattr('voice_input.MODEL_STATE_FILE', state_file)

        voice_input.set_current_model("invalid_model")

        # Single model architecture: does nothing
        assert not state_file.exists()


class TestModelLoader:
    """Test ModelLoader class (new architecture)."""

    def test_device_config_available(self):
        """Device config should be available."""
        assert hasattr(voice_input, 'DEVICE')
        assert voice_input.DEVICE in ["cuda:0", "cpu"]

    def test_model_presets_have_framework(self):
        """All model presets should have a framework field."""
        valid_frameworks = {"funasr", "transformers", "fireredasr", "faster-whisper"}
        for model_id, preset in voice_input.MODEL_PRESETS.items():
            assert "framework" in preset
            assert preset["framework"] in valid_frameworks


class TestHotwords:
    """Test hotword configuration."""

    def test_hotwords_defined(self):
        """Hotwords should be defined."""
        assert hasattr(voice_input, 'HOTWORDS')
        assert isinstance(voice_input.HOTWORDS, str)
        assert len(voice_input.HOTWORDS) > 0

    def test_hotwords_contains_common_terms(self):
        """Hotwords should contain common technical terms."""
        hotwords = voice_input.HOTWORDS.lower()
        assert "python" in hotwords
        assert "machine learning" in hotwords
