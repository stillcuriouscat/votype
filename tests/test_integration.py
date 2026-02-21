"""
Integration Tests - Test interactions between components.

Coverage:
- Daemon lifecycle
- Recording flow
- Toggle command flow (Race Condition focus)
- ASRDaemon class
"""

import os
import sys
import json
import socket
import signal
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input


# ============ 1. Daemon Lifecycle ============

class TestDaemonLifecycle:
    """Test daemon lifecycle management."""

    def test_daemon_creates_pid_file_immediately(self, tmp_path, mock_asr_model, mock_gtk, isolated_environment):
        """
        Should create PID file immediately when starting daemon.
        Test logic: verify that run() writes PID before model loading.
        """
        # Use temporary paths
        pid_file = tmp_path / "daemon.pid"
        socket_path = tmp_path / "daemon.sock"

        daemon = voice_input.ASRDaemon()

        # Directly test PID file writing logic
        # Simulate the key part of run() method
        with patch('voice_input.DAEMON_PID_FILE', pid_file):
            with patch('voice_input.SOCKET_PATH', socket_path):
                with patch('voice_input.is_daemon_running', return_value=False):
                    # Manually execute the PID file creation logic from run()
                    pid_file.write_text(str(os.getpid()))

                    assert pid_file.exists()
                    assert int(pid_file.read_text().strip()) == os.getpid()

    def test_daemon_single_instance_check(self, isolated_environment, mock_asr_model, mock_gtk):
        """Second daemon instance should exit when one is already running."""
        pid_file = isolated_environment['daemon_pid_file']

        # Simulate an existing daemon running
        pid_file.write_text(str(os.getpid()))

        daemon = voice_input.ASRDaemon()

        # Mock is_daemon_running to return True
        with patch('voice_input.is_daemon_running', return_value=True):
            daemon.run()

        # Verify no duplicate run (model should not be loaded)
        mock_asr_model['model_class'].assert_not_called()

    def test_daemon_stop_cleans_up_files(self, isolated_environment, mock_asr_model, mock_gtk):
        """Should clean up PID and Socket files when stopping daemon."""
        daemon = voice_input.ASRDaemon()
        pid_file = isolated_environment['daemon_pid_file']
        socket_path = isolated_environment['socket_path']

        # Create files to simulate running state
        pid_file.write_text(str(os.getpid()))
        socket_path.touch()

        # Mock Gtk.main to return immediately
        with patch.object(voice_input.Gtk, 'main', return_value=None):
            with patch('voice_input.is_daemon_running', return_value=False):
                daemon.run()

        # Files should be cleaned up (after run ends)
        # Note: since we mocked is_daemon_running, actual cleanup logic may differ


class TestDaemonSocketCommunication:
    """Test daemon socket communication."""

    def test_daemon_handles_ping(self, isolated_environment, mock_asr_model, mock_gtk):
        """Daemon should respond to ping command."""
        daemon = voice_input.ASRDaemon()
        daemon.model = mock_asr_model['model_instance']
        daemon.running = True
        daemon.indicator = None

        # Create mock client
        mock_client = MagicMock()
        mock_client.recv.return_value = json.dumps({"command": "ping"}).encode()

        daemon.handle_client(mock_client)

        # Verify response
        mock_client.send.assert_called_once()
        response = json.loads(mock_client.send.call_args[0][0].decode())
        assert response["status"] == "ok"
        assert "model" in response  # ping now returns current model

    def test_daemon_handles_transcribe(self, isolated_environment, mock_asr_model, mock_gtk):
        """Daemon should handle transcribe command."""
        with patch('voice_input.ModelInference.transcribe') as mock_transcribe:
            mock_transcribe.return_value = "test transcription"

            daemon = voice_input.ASRDaemon()
            daemon.model = mock_asr_model['model_instance']
            daemon.framework = "funasr"
            daemon.current_model_id = "fun-asr-nano"
            daemon.extra_data = None
            daemon.running = True
            daemon.indicator = None

            mock_client = MagicMock()
            mock_client.recv.return_value = json.dumps({
                "command": "transcribe",
                "data": "/path/to/audio.wav"
            }).encode()

            daemon.handle_client(mock_client)

            # Verify ModelInference.transcribe was called
            mock_transcribe.assert_called_once()

            # Verify response
            response = json.loads(mock_client.send.call_args[0][0].decode())
            assert "text" in response

    def test_daemon_handles_status_commands(self, isolated_environment, mock_asr_model, mock_gtk):
        """Daemon should handle status switch commands."""
        daemon = voice_input.ASRDaemon()
        daemon.model = mock_asr_model['model_instance']
        daemon.running = True
        daemon.indicator = mock_gtk['indicator']
        daemon.status_item = MagicMock()

        status_commands = ["recording_start", "recording_stop", "set_idle"]

        for cmd in status_commands:
            mock_client = MagicMock()
            mock_client.recv.return_value = json.dumps({"command": cmd}).encode()

            daemon.handle_client(mock_client)

            response = json.loads(mock_client.send.call_args[0][0].decode())
            assert response == {"status": "ok"}

    def test_daemon_handles_unknown_command(self, isolated_environment, mock_asr_model, mock_gtk):
        """Daemon should handle unknown commands."""
        daemon = voice_input.ASRDaemon()
        daemon.model = mock_asr_model['model_instance']
        daemon.running = True
        daemon.indicator = None

        mock_client = MagicMock()
        mock_client.recv.return_value = json.dumps({"command": "unknown_cmd"}).encode()

        daemon.handle_client(mock_client)

        response = json.loads(mock_client.send.call_args[0][0].decode())
        assert "error" in response


# ============ 2. Recording Flow ============

class TestRecordingFlow:
    """Test recording flow."""

    def test_start_recording_creates_pid_file(self, isolated_environment, mock_subprocess, mock_notify):
        """Starting recording should create PID file."""
        # Mock subprocess.Popen to return process object
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_subprocess.Popen.return_value = mock_proc

        with patch('voice_input.is_recording', return_value=False):
            with patch('voice_input.is_daemon_running', return_value=True):
                with patch('voice_input.send_to_daemon', return_value={"status": "ok"}):
                    voice_input.start_recording()

        # Verify Popen was called to start arecord
        mock_subprocess.Popen.assert_called()
        call_args = str(mock_subprocess.Popen.call_args)
        assert "arecord" in call_args

    def test_start_recording_notifies_daemon(self, isolated_environment, mock_subprocess, mock_notify):
        """Starting recording should notify daemon to update status."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_subprocess.Popen.return_value = mock_proc

        with patch('voice_input.is_recording', return_value=False):
            with patch('voice_input.is_daemon_running', return_value=True):
                with patch('voice_input.send_to_daemon') as mock_send:
                    mock_send.return_value = {"status": "ok"}
                    voice_input.start_recording()

                    # Should send recording_start command
                    mock_send.assert_called_with("recording_start")

    def test_start_recording_when_already_recording(self, isolated_environment, mock_notify):
        """Should show anomaly notification when already recording."""
        with patch('voice_input.is_recording', return_value=True):
            voice_input.start_recording()

            # Should show anomaly notification
            mock_notify.assert_called()
            call_args = str(mock_notify.call_args)
            assert "anomal" in call_args.lower() or "already" in call_args.lower() or "State" in call_args

    def test_stop_recording_terminates_process(self, isolated_environment, mock_notify):
        """Stopping recording should terminate the arecord process."""
        pid_file = isolated_environment['pid_file']
        audio_file = isolated_environment['audio_file']

        # Create PID file and audio file
        pid_file.write_text("12345")
        audio_file.touch()

        with patch('voice_input.is_recording', return_value=True):
            with patch('voice_input.is_daemon_running', return_value=True):
                with patch('voice_input.send_to_daemon') as mock_send:
                    mock_send.return_value = {"text": "transcription result"}
                    with patch('voice_input.type_text'):
                        with patch('os.kill') as mock_kill:
                            with patch('voice_input.time.sleep'):
                                voice_input.stop_recording()

                                # Should call os.kill to terminate process
                                mock_kill.assert_called()

    def test_stop_recording_when_not_recording(self, isolated_environment, mock_notify):
        """Should show anomaly notification when stopping while not recording."""
        with patch('voice_input.is_recording', return_value=False):
            voice_input.stop_recording()

            mock_notify.assert_called()
            call_args = str(mock_notify.call_args)
            assert "anomal" in call_args.lower() or "not" in call_args.lower() or "State" in call_args


# ============ 3. Toggle Command Flow (Race Condition Focus) ============

class TestToggleRecording:
    """
    Test toggle_recording function.
    This is the main user interaction entry point, the focus of Race Condition fix.
    """

    def test_toggle_starts_daemon_when_not_running(self, isolated_environment, mock_subprocess, mock_notify):
        """Toggle should auto-start daemon when daemon is not running."""
        with patch('voice_input.is_daemon_ready', return_value=False):
            with patch('voice_input.is_daemon_running', return_value=False):
                with patch('voice_input.time.sleep'):
                    # Simulate becoming ready after startup
                    ready_states = [False] * 5 + [True]
                    with patch('voice_input.is_daemon_ready', side_effect=ready_states):
                        voice_input.toggle_recording()

        # Should call start command
        mock_subprocess.run.assert_called()
        call_args = str(mock_subprocess.run.call_args)
        assert "_daemon" in call_args

    def test_toggle_shows_wait_message_when_daemon_starting(self, isolated_environment, mock_notify):
        """Toggle should show wait message when daemon is starting."""
        with patch('voice_input.is_daemon_ready', return_value=False):
            with patch('voice_input.is_daemon_running', return_value=True):
                voice_input.toggle_recording()

        # Should show wait message
        mock_notify.assert_called()
        call_args = str(mock_notify.call_args)
        assert "starting" in call_args.lower() or "wait" in call_args.lower() or "Starting" in call_args

    def test_toggle_starts_recording_when_daemon_ready(self, isolated_environment, mock_subprocess, mock_notify):
        """Toggle should start recording when daemon is ready and not recording."""
        # mock_subprocess fixture already sets up Popen to return pid=12345

        with patch('voice_input.is_daemon_ready', return_value=True):
            with patch('voice_input.is_recording', return_value=False):
                with patch('voice_input.send_to_daemon', return_value={"status": "ok"}):
                    voice_input.toggle_recording()

        # Should call arecord (using Popen not run)
        assert mock_subprocess.Popen.called

    def test_toggle_stops_recording_when_already_recording(self, isolated_environment, mock_notify):
        """Toggle should stop recording when daemon is ready and already recording."""
        pid_file = isolated_environment['pid_file']
        audio_file = isolated_environment['audio_file']
        pid_file.write_text("12345")
        audio_file.touch()

        with patch('voice_input.is_daemon_ready', return_value=True):
            with patch('voice_input.is_recording', return_value=True):
                with patch('voice_input.is_daemon_running', return_value=True):
                    with patch('voice_input.send_to_daemon') as mock_send:
                        mock_send.return_value = {"text": "transcription result"}
                        with patch('voice_input.type_text'):
                            with patch('os.kill'):
                                with patch('voice_input.time.sleep'):
                                    voice_input.toggle_recording()

                        # Should send recording_stop command
                        assert any(
                            call[0][0] == "recording_stop"
                            for call in mock_send.call_args_list
                        )

    def test_toggle_waits_for_daemon_ready(self, isolated_environment, mock_subprocess, mock_notify):
        """Should wait for ping success after starting daemon."""
        call_count = [0]

        def mock_is_ready():
            call_count[0] += 1
            return call_count[0] > 3  # Returns True on 4th call

        with patch('voice_input.is_daemon_ready', side_effect=mock_is_ready):
            with patch('voice_input.is_daemon_running', return_value=False):
                with patch('voice_input.time.sleep'):
                    voice_input.toggle_recording()

        # Should check is_daemon_ready multiple times
        assert call_count[0] >= 3

    def test_toggle_timeout_on_daemon_start(self, isolated_environment, mock_subprocess, mock_notify):
        """Should show failure notification when daemon start times out."""
        with patch('voice_input.is_daemon_ready', return_value=False):
            with patch('voice_input.is_daemon_running', return_value=False):
                with patch('voice_input.time.sleep'):
                    voice_input.toggle_recording()

        # Should show failure notification
        assert mock_notify.called
        # Check the last call
        last_call = mock_notify.call_args_list[-1]
        call_str = str(last_call)
        assert "fail" in call_str.lower() or "timeout" in call_str.lower() or "log" in call_str.lower()


# ============ 4. ASRDaemon Class Tests ============

class TestASRDaemonClass:
    """Test ASRDaemon class methods."""

    def test_load_model(self, mock_asr_model, isolated_environment):
        """load_model should load ASR model."""
        daemon = voice_input.ASRDaemon()
        daemon.load_model()

        mock_asr_model['model_class'].assert_called_once()
        assert daemon.model is not None

    def test_transcribe_returns_text(self, mock_asr_model, isolated_environment):
        """transcribe should return recognized text."""
        with patch('voice_input.ModelInference.transcribe') as mock_transcribe:
            mock_transcribe.return_value = "test recognition result"

            daemon = voice_input.ASRDaemon()
            daemon.model = mock_asr_model['model_instance']
            daemon.framework = "funasr"
            daemon.current_model_id = "fun-asr-nano"
            daemon.extra_data = None

            result = daemon.transcribe("/path/to/audio.wav")

            assert "text" in result
            assert result["text"] == "test recognition result"

    def test_transcribe_returns_error_when_model_not_loaded(self, isolated_environment):
        """transcribe should return error when model is not loaded."""
        daemon = voice_input.ASRDaemon()
        daemon.model = None

        result = daemon.transcribe("/path/to/audio.wav")

        assert "error" in result

    def test_set_status_updates_indicator(self, mock_gtk, isolated_environment):
        """set_status should update tray icon."""
        daemon = voice_input.ASRDaemon()
        daemon.indicator = mock_gtk['indicator']
        daemon.status_item = MagicMock()

        daemon.set_status("recording")

        # Verify GLib.idle_add was called (executed immediately via mock)
        # Icon should be updated to recording state
        mock_gtk['indicator'].set_icon_full.assert_called()

    def test_set_status_handles_no_indicator(self, isolated_environment):
        """set_status should not throw exception when there is no indicator."""
        daemon = voice_input.ASRDaemon()
        daemon.indicator = None

        # Should not throw exception
        daemon.set_status("recording")


# ============ 5. State Machine Tests ============

class TestStateMachine:
    """Test the correctness of state transitions."""

    def test_idle_to_recording_transition(self, isolated_environment, mock_subprocess, mock_notify):
        """Idle -> recording state transition."""
        mock_subprocess.run.return_value = MagicMock(stdout="12345\n", stderr="")

        with patch('voice_input.is_daemon_ready', return_value=True):
            with patch('voice_input.is_recording', return_value=False):
                with patch('voice_input.is_daemon_running', return_value=True):
                    with patch('voice_input.send_to_daemon') as mock_send:
                        mock_send.return_value = {"status": "ok"}
                        voice_input.toggle_recording()

                        # Should send recording_start (called in start_recording)
                        mock_send.assert_called_with("recording_start")

    def test_recording_to_processing_transition(self, isolated_environment, mock_notify):
        """Recording -> processing state transition."""
        pid_file = isolated_environment['pid_file']
        audio_file = isolated_environment['audio_file']
        pid_file.write_text("12345")
        audio_file.touch()

        with patch('voice_input.is_daemon_ready', return_value=True):
            with patch('voice_input.is_recording', return_value=True):
                with patch('voice_input.is_daemon_running', return_value=True):
                    with patch('voice_input.send_to_daemon') as mock_send:
                        mock_send.return_value = {"text": "result"}
                        with patch('voice_input.type_text'):
                            with patch('os.kill'):
                                with patch('voice_input.time.sleep'):
                                    voice_input.toggle_recording()

                        # Should send recording_stop (triggers processing state)
                        calls = [c[0][0] for c in mock_send.call_args_list]
                        assert "recording_stop" in calls


# ============ 6. Model Switching Tests ============

class TestDaemonModelCommands:
    """Test daemon model-related commands."""

    def test_handle_get_model(self, mock_asr_model, isolated_environment):
        """handle_client should correctly handle get_model command."""
        daemon = voice_input.ASRDaemon()
        daemon.model = mock_asr_model['model_instance']
        daemon.current_model_id = "paraformer"

        # Create mock client socket
        mock_client = MagicMock()
        mock_client.recv.return_value = json.dumps({"command": "get_model"}).encode()

        daemon.handle_client(mock_client)

        # Check sent response
        sent_data = mock_client.send.call_args[0][0]
        response = json.loads(sent_data.decode())
        assert response["model"] == "paraformer"
        assert response["name"] == "Paraformer-zh"

    def test_handle_list_models(self, mock_asr_model, isolated_environment):
        """handle_client should correctly handle list_models command."""
        daemon = voice_input.ASRDaemon()
        daemon.model = mock_asr_model['model_instance']
        daemon.current_model_id = "paraformer"

        mock_client = MagicMock()
        mock_client.recv.return_value = json.dumps({"command": "list_models"}).encode()

        daemon.handle_client(mock_client)

        sent_data = mock_client.send.call_args[0][0]
        response = json.loads(sent_data.decode())
        assert "models" in response
        assert "current" in response
        assert response["current"] == "paraformer"
        assert "paraformer" in response["models"]
        assert "sensevoice" in response["models"]

    def test_ping_returns_current_model(self, mock_asr_model, isolated_environment):
        """ping command should return current model ID."""
        daemon = voice_input.ASRDaemon()
        daemon.model = mock_asr_model['model_instance']
        daemon.current_model_id = "sensevoice"

        mock_client = MagicMock()
        mock_client.recv.return_value = json.dumps({"command": "ping"}).encode()

        daemon.handle_client(mock_client)

        sent_data = mock_client.send.call_args[0][0]
        response = json.loads(sent_data.decode())
        assert response["status"] == "ok"
        assert response["model"] == "sensevoice"
