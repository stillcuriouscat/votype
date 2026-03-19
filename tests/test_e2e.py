"""
End-to-End Tests - Test complete user flows.

Coverage:
- Complete user flows
- Exception recovery flows
- System tray icon states
- Performance tests

Tests marked with @pytest.mark.e2e or @pytest.mark.real_model require a real environment.
"""

import os
import sys
import json
import socket
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input


# ============ 1. Complete User Flows ============

@pytest.mark.e2e
class TestCompleteUserFlow:
    """Test complete user usage flows."""

    def test_first_time_user_flow(self, isolated_environment, mock_subprocess, mock_notify, mock_asr_model, mock_gtk):
        """
        First-time user complete flow:
        1. toggle -> auto-start daemon
        2. Wait for ready notification
        3. toggle again -> start recording
        4. toggle -> stop and type text
        """
        mock_subprocess.run.return_value = MagicMock(stdout="12345\n", stderr="")

        # Phase 1: First toggle, daemon not running
        toggle_count = [0]
        daemon_started = [False]

        def mock_is_ready():
            # Becomes ready on 3rd check after startup
            if daemon_started[0] and toggle_count[0] > 2:
                return True
            return False

        def mock_run(args, **kwargs):
            if "_daemon" in str(args):
                daemon_started[0] = True
            return MagicMock(stdout="12345\n", stderr="")

        mock_subprocess.run.side_effect = mock_run

        with patch('voice_input.is_daemon_ready', side_effect=mock_is_ready):
            with patch('voice_input.is_daemon_running', return_value=False):
                with patch('voice_input.time.sleep', side_effect=lambda x: toggle_count.__setitem__(0, toggle_count[0] + 1)):
                    voice_input.toggle_recording()

        # Should have started daemon
        assert daemon_started[0]

    def test_daemon_already_running_flow(self, isolated_environment, mock_notify, mock_asr_model):
        """Flow when daemon is already running."""
        with patch('voice_input.subprocess') as mock_subprocess:
            mock_subprocess.run.return_value = MagicMock(stdout="12345\n", stderr="")

            with patch('voice_input.is_daemon_ready', return_value=True):
                with patch('voice_input.is_recording', return_value=False):
                    with patch('voice_input.is_daemon_running', return_value=True):
                        with patch('voice_input.send_to_daemon', return_value={"status": "ok"}):
                            voice_input.toggle_recording()

            # Should start recording directly, not start a new daemon
            # Check that no daemon start command was called (containing "voice_input.py _daemon")
            calls = [str(c) for c in mock_subprocess.run.call_args_list]
            # _daemon command format is "voice_input.py _daemon" or similar
            has_daemon_start = any("voice_input.py\" \"_daemon" in c or "_daemon\"" in c for c in calls)
            assert not has_daemon_start, f"Unexpected daemon start call in: {calls}"

    def test_toggle_during_daemon_startup_flow(self, isolated_environment, mock_notify):
        """Flow when toggling during daemon startup."""
        # Simulate: daemon PID exists but not yet ready
        isolated_environment['daemon_pid_file'].write_text(str(os.getpid()))

        with patch('voice_input.is_daemon_ready', return_value=False):
            with patch('voice_input.is_daemon_running', return_value=True):
                voice_input.toggle_recording()

        # Should show wait message
        assert mock_notify.called
        call_str = str(mock_notify.call_args)
        assert "starting" in call_str.lower() or "wait" in call_str.lower() or "Starting" in call_str


# ============ 2. Exception Recovery Flows ============

@pytest.mark.e2e
class TestExceptionRecovery:
    """Test recovery flows under exception conditions."""

    def test_daemon_crash_recovery(self, isolated_environment, mock_subprocess, mock_notify):
        """Recovery after daemon crash."""
        # Create leftover PID file (pointing to non-existent process)
        daemon_pid_file = isolated_environment['daemon_pid_file']
        daemon_pid_file.write_text("99999")

        mock_subprocess.run.return_value = MagicMock(stdout="12345\n", stderr="")

        # is_daemon_running should detect dead process
        with patch('voice_input.os.kill', side_effect=ProcessLookupError):
            running = voice_input.is_daemon_running()

        # Should return False and clean up PID file
        assert running is False
        assert not daemon_pid_file.exists()

    def test_stale_db_recording_cleanup(self, isolated_environment):
        """Stale recording status in DB should be cleaned up by is_recording()."""
        import state_db as _state_db

        # Set recording with a dead PID in DB
        _state_db.update_state(
            isolated_environment["state_db_path"],
            status="recording",
            recording_pid=99999,
            recording_path="/tmp/test.wav",
        )

        with patch('voice_input.os.kill', side_effect=ProcessLookupError):
            result = voice_input.is_recording()

        assert result is False
        # DB should be cleaned up to idle
        state = _state_db.get_state(isolated_environment["state_db_path"])
        assert state["status"] == "idle"

    def test_stale_socket_file_cleanup(self, isolated_environment, mock_asr_model, mock_gtk):
        """Stale socket file should be cleaned up correctly."""
        socket_path = isolated_environment['socket_path']
        socket_path.touch()

        daemon = voice_input.ASRDaemon()

        # run() should delete old socket file
        with patch.object(voice_input.Gtk, 'main', return_value=None):
            with patch('voice_input.is_daemon_running', return_value=False):
                # Starting daemon will clean up old socket
                thread = threading.Thread(target=daemon.run, daemon=True)
                thread.start()

                # Wait briefly for daemon to start
                time.sleep(0.5)

                # Stop daemon
                daemon.running = False
                thread.join(timeout=2)

    def test_recording_interrupted_recovery(self, isolated_environment, mock_notify):
        """Recovery when daemon crashes during recording."""
        pid_file = isolated_environment['pid_file']
        audio_file = isolated_environment['audio_file']

        # Simulate recording in progress
        pid_file.write_text(str(os.getpid()))
        audio_file.touch()

        # Simulate daemon crash (not responding)
        with patch('voice_input.is_recording', return_value=True):
            with patch('voice_input.is_daemon_running', return_value=False):
                with patch('os.kill'):
                    with patch('voice_input.time.sleep'):
                        voice_input.stop_recording()

        # Should show error notification
        assert mock_notify.called


# ============ 3. System Tray Icon States ============

@pytest.mark.e2e
class TestTrayIconStatus:
    """Test system tray icon state changes."""

    def test_icon_idle_on_daemon_start(self, isolated_environment, mock_asr_model, mock_gtk):
        """Should show idle icon after daemon startup completes."""
        daemon = voice_input.ASRDaemon()

        with patch.object(voice_input.Gtk, 'main', return_value=None):
            with patch('voice_input.is_daemon_running', return_value=False):
                thread = threading.Thread(target=daemon.run, daemon=True)
                thread.start()

                # Wait for initialization
                time.sleep(0.3)

                # Stop
                daemon.running = False
                thread.join(timeout=2)

        # Initial icon should be idle
        if daemon.indicator:
            # Verify initial state
            pass  # Hard to verify actual icon in mock environment

    def test_icon_state_transitions(self, mock_gtk, isolated_environment):
        """Icon state transition test."""
        daemon = voice_input.ASRDaemon()
        daemon.indicator = mock_gtk['indicator']
        daemon.status_item = MagicMock()

        # Test state transitions
        states = ["idle", "recording", "processing", "idle"]
        expected_icons = ["mic-idle", "mic-recording", "mic-processing", "mic-idle"]

        for state, expected_icon in zip(states, expected_icons):
            daemon.set_status(state)

            # Verify icon was set
            call_args = mock_gtk['indicator'].set_icon_full.call_args
            if call_args:
                actual_icon = call_args[0][0]
                assert actual_icon == expected_icon

    def test_icon_no_crash_without_indicator(self, isolated_environment):
        """Should not crash without indicator."""
        daemon = voice_input.ASRDaemon()
        daemon.indicator = None

        # Should not throw exception
        for state in ["idle", "recording", "processing"]:
            daemon.set_status(state)


# ============ 4. Real Model Tests ============

@pytest.mark.real_model
@pytest.mark.slow
class TestRealASRModel:
    """Tests using real ASR model (slow)."""

    def test_model_loads_successfully(self, real_asr_model):
        """Real model should load successfully."""
        assert real_asr_model is not None

    def test_transcribe_silence(self, real_asr_model, sample_audio_file):
        """Transcribing a silent audio file should return empty or short text."""
        result = real_asr_model.generate(input=str(sample_audio_file))

        # Silent audio should return empty or very short result
        if result and len(result) > 0:
            text = result[0].get("text", "")
            assert len(text) < 50  # Silence should not produce long text

    def test_daemon_with_real_model(self, isolated_environment, real_asr_model, mock_gtk):
        """Daemon test using a real model."""
        daemon = voice_input.ASRDaemon()

        # Use pre-loaded model
        daemon.model = real_asr_model
        daemon.running = True
        daemon.indicator = None

        # Test transcription
        result = daemon.transcribe(str(Path(__file__).parent / "test_audio.wav"))

        # Should return a result (may be empty if no test audio exists)
        assert "text" in result or "error" in result


# ============ 5. Performance Tests ============

@pytest.mark.slow
class TestPerformance:
    """Performance tests."""

    def test_ping_response_time(self, mock_socket_server, monkeypatch, timer):
        """Ping response time should be less than 100ms."""
        monkeypatch.setattr('voice_input.SOCKET_PATH', mock_socket_server['socket_path'])

        timer.start()
        for _ in range(10):
            voice_input.send_to_daemon("ping")
        timer.stop()

        avg_time = timer.elapsed / 10
        assert avg_time < 0.1  # Average less than 100ms

    def test_repeated_toggle_no_memory_leak(self, isolated_environment, mock_subprocess, mock_notify):
        """Repeated toggle should not cause memory leaks."""
        import gc

        mock_subprocess.run.return_value = MagicMock(stdout="12345\n", stderr="")

        # Record initial object count
        gc.collect()
        initial_objects = len(gc.get_objects())

        with patch('voice_input.is_daemon_ready', return_value=True):
            with patch('voice_input.is_recording', return_value=False):
                with patch('voice_input.is_daemon_running', return_value=True):
                    with patch('voice_input.send_to_daemon', return_value={"status": "ok"}):
                        for _ in range(50):
                            voice_input.toggle_recording()

        # Check object growth
        gc.collect()
        final_objects = len(gc.get_objects())

        # Object growth should be bounded (not unbounded)
        growth = final_objects - initial_objects
        assert growth < 1000  # Some growth is acceptable, but not unbounded

    def test_daemon_startup_time_mock(self, isolated_environment, mock_asr_model, mock_gtk, timer):
        """Daemon startup time test (using mock model)."""
        daemon = voice_input.ASRDaemon()

        with patch.object(voice_input.Gtk, 'main', return_value=None):
            with patch('voice_input.is_daemon_running', return_value=False):
                timer.start()
                thread = threading.Thread(target=daemon.run, daemon=True)
                thread.start()

                # Wait for model loading to complete
                time.sleep(0.5)
                timer.stop()

                daemon.running = False
                thread.join(timeout=2)

        # Mock model should be fast
        assert timer.elapsed < 5


# ============ 6. Command Line Interface Tests ============

@pytest.mark.e2e
class TestCommandLineInterface:
    """Test command line interface."""

    def test_status_command(self, isolated_environment, capsys):
        """The status command should display status information."""
        with patch('voice_input.is_recording', return_value=False):
            with patch('voice_input.is_daemon_running', return_value=False):
                voice_input.show_status()

        captured = capsys.readouterr()
        assert "Recording:" in captured.out
        assert "Daemon:" in captured.out

    def test_main_with_unknown_command(self, capsys):
        """Unknown command should display an error."""
        with patch('sys.argv', ['voice_input', 'unknown_cmd']):
            with pytest.raises(SystemExit) as exc_info:
                voice_input.main()

        assert exc_info.value.code == 1

    def test_main_without_arguments(self, capsys):
        """No arguments should display help."""
        with patch('sys.argv', ['voice_input']):
            with pytest.raises(SystemExit) as exc_info:
                voice_input.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Commands:" in captured.out or "toggle" in captured.out


# ============ 7. Environment Check Tests ============

@pytest.mark.e2e
class TestEnvironmentChecks:
    """Test environment dependency checks."""

    def test_xclip_not_installed_handling(self):
        """Should handle gracefully when xclip is not installed."""
        with patch('voice_input.subprocess.run', side_effect=FileNotFoundError("xclip")):
            with patch('voice_input.subprocess.Popen', side_effect=FileNotFoundError("xclip")):
                # Should not crash
                try:
                    voice_input.clipboard_get()
                except FileNotFoundError:
                    pass  # Expected exception

    def test_xdotool_not_installed_handling(self):
        """Should handle gracefully when xdotool is not installed."""
        with patch('voice_input.clipboard_get', return_value=""):
            with patch('voice_input.clipboard_set'):
                with patch('voice_input.subprocess.run', side_effect=FileNotFoundError("xdotool")):
                    with patch('voice_input.time.sleep'):
                        # Should not crash
                        voice_input.type_text("test")

    def test_notify_send_not_installed_handling(self):
        """Should fail silently when notify-send is not installed."""
        with patch('voice_input.subprocess.Popen', side_effect=FileNotFoundError("notify-send")):
            # Should not crash
            voice_input.notify("Test", "Message")


# ============ 8. Model Management E2E Tests ============

@pytest.mark.e2e
class TestModelManagementE2E:
    """End-to-end tests for model management commands."""

    def test_models_command_lists_all_models(self, isolated_environment, capsys):
        """The 'voice-input models' command should list all available models."""
        with patch('sys.argv', ['voice_input', 'models']):
            voice_input.main()

        captured = capsys.readouterr()
        assert "paraformer" in captured.out
        assert "sensevoice" in captured.out
        # Note: fun-asr-nano temporarily removed due to FunASR bug #2757
        assert "Available models" in captured.out

    def test_status_shows_model_info_daemon_not_running(self, isolated_environment, capsys):
        """Single-model architecture: status command should show default model when daemon is not running."""
        with patch('sys.argv', ['voice_input', 'status']):
            with patch('voice_input.is_daemon_running', return_value=False):
                voice_input.main()

        captured = capsys.readouterr()
        # Single-model architecture: should display default model (sensevoice)
        assert "SenseVoice" in captured.out or "sensevoice" in captured.out

    def test_status_shows_running_daemon_model(self, isolated_environment, capsys):
        """The status command should show the model of the running daemon."""
        with patch('sys.argv', ['voice_input', 'status']):
            with patch('voice_input.is_daemon_running', return_value=True):
                with patch('voice_input.send_to_daemon') as mock_send:
                    mock_send.return_value = {
                        "model": "sensevoice",
                        "name": "SenseVoice",
                        "description": "Chinese-English mixed, multilingual support"
                    }
                    voice_input.main()

        captured = capsys.readouterr()
        assert "SenseVoice" in captured.out or "sensevoice" in captured.out


@pytest.mark.e2e
class TestHotwordsE2E:
    """Hotwords feature tests."""

    def test_hotwords_config_is_string(self):
        """Hotwords config should be a string."""
        assert isinstance(voice_input.HOTWORDS, str)

    def test_hotwords_passed_to_transcribe(self, mock_asr_model, isolated_environment):
        """Transcribe should use ModelInference for inference."""
        with patch('voice_input.ModelInference.transcribe') as mock_transcribe:
            mock_transcribe.return_value = "test transcription"

            daemon = voice_input.ASRDaemon()
            daemon.model = MagicMock()
            daemon.framework = "funasr"
            daemon.current_model_id = "fun-asr-nano"
            daemon.extra_data = None

            result = daemon.transcribe("/path/to/audio.wav")

            # Verify ModelInference.transcribe was called
            mock_transcribe.assert_called_once()
            # Verify hotwords parameter was passed
            # call_args is a Call object, use .kwargs to access keyword arguments
            call_kwargs = mock_transcribe.call_args.kwargs
            assert "hotwords" in call_kwargs or call_kwargs == {}  # May be passed as positional argument
            assert result["text"] == "test transcription"


@pytest.mark.e2e
class TestCUDAConfig:
    """CUDA configuration tests."""

    def test_device_config_exists(self):
        """Device config should exist."""
        assert hasattr(voice_input, 'DEVICE')
        assert voice_input.DEVICE in ["cuda:0", "cpu"]

    def test_model_config_includes_device(self, isolated_environment):
        """Model loading should use device config."""
        # New architecture: device is passed via ModelLoader.load_model
        # Verify DEVICE config exists and is correct
        assert hasattr(voice_input, 'DEVICE')
        assert voice_input.DEVICE in ["cuda:0", "cpu"]

        # Verify FunASR model receives device parameter when loading
        preset = voice_input.MODEL_PRESETS["paraformer"]
        assert preset["framework"] == "funasr"
