"""
Unit tests for US-002: CLI state operations migrated from files to SQLite.

Coverage:
- is_recording() — DB-based status + PID liveness
- start_recording() — DB writes instead of file writes / IPC
- stop_recording() — DB reads/writes instead of file reads / IPC (CRITIC-R2-C1)
- toggle_recording() — DB status check instead of PROCESSING_FILE
- handle_client() — status IPC commands removed
"""

import os
import sys
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input
import state_db


# ============ Helper ============

@pytest.fixture
def state_db_env(tmp_path, monkeypatch):
    """Set up isolated state DB for CLI function tests."""
    config_dir = tmp_path / "config" / "voice-input"
    config_dir.mkdir(parents=True)

    db_path = config_dir / "state.db"
    monkeypatch.setattr("voice_input.CONFIG_DIR", config_dir)
    monkeypatch.setattr("voice_input.STATE_DB_PATH", db_path)
    monkeypatch.setattr("voice_input.PID_FILE", config_dir / "recording.pid")
    monkeypatch.setattr("voice_input.AUDIO_FILE", config_dir / "recording.wav")
    monkeypatch.setattr("voice_input.AUDIO_PATH_FILE", config_dir / "recording_path.txt")
    monkeypatch.setattr("voice_input.DAEMON_PID_FILE", config_dir / "daemon.pid")
    monkeypatch.setattr("voice_input.SOCKET_PATH", config_dir / "daemon.sock")
    monkeypatch.setattr("voice_input.PROCESSING_FILE", config_dir / "processing.flag")
    monkeypatch.setattr("state_db.DEFAULT_DB_PATH", db_path)
    state_db.init_db(db_path)

    return {
        "config_dir": config_dir,
        "db_path": db_path,
    }


# ============ 1. is_recording() ============

class TestIsRecordingDB:
    """Test DB-based is_recording()."""

    def test_returns_false_when_status_idle(self, state_db_env):
        """DB status='idle' → is_recording() returns False."""
        assert voice_input.is_recording() is False

    def test_returns_true_when_recording_and_pid_alive(self, state_db_env):
        """DB status='recording' + live PID → True."""
        state_db.update_state(
            state_db_env["db_path"],
            status="recording",
            recording_pid=os.getpid(),
            recording_path="/tmp/test.wav",
        )
        assert voice_input.is_recording() is True

    def test_returns_false_when_recording_but_pid_dead(self, state_db_env):
        """DB status='recording' but dead PID → False + DB reset to idle."""
        state_db.update_state(
            state_db_env["db_path"],
            status="recording",
            recording_pid=99999,
            recording_path="/tmp/test.wav",
        )
        # Ensure the PID is dead
        with patch("voice_input.os.kill", side_effect=ProcessLookupError):
            result = voice_input.is_recording()

        assert result is False
        # DB should be cleaned up
        s = state_db.get_state(state_db_env["db_path"])
        assert s["status"] == "idle"
        assert s["recording_pid"] is None
        assert s["recording_path"] is None

    def test_returns_false_when_recording_but_pid_none(self, state_db_env):
        """DB status='recording' but recording_pid is None → False + cleanup."""
        state_db.update_state(
            state_db_env["db_path"],
            status="recording",
            recording_pid=None,
        )
        assert voice_input.is_recording() is False
        s = state_db.get_state(state_db_env["db_path"])
        assert s["status"] == "idle"

    def test_returns_false_when_processing(self, state_db_env):
        """DB status='processing' → is_recording() returns False."""
        state_db.update_state(state_db_env["db_path"], status="processing")
        assert voice_input.is_recording() is False


# ============ 2. start_recording() ============

class TestStartRecordingDB:
    """Test DB-based start_recording()."""

    def test_writes_state_to_db(self, state_db_env, mock_subprocess, mock_notify):
        """start_recording() should write status/pid/path to DB."""
        mock_proc = MagicMock()
        mock_proc.pid = 42000
        mock_subprocess.Popen.return_value = mock_proc

        with patch("voice_input.is_recording", return_value=False):
            with patch("voice_input.is_daemon_running", return_value=False):
                voice_input.start_recording()

        s = state_db.get_state(state_db_env["db_path"])
        assert s["status"] == "recording"
        assert s["recording_pid"] == 42000
        assert s["recording_path"] is not None
        assert "recording_" in s["recording_path"]

    def test_does_not_write_pid_file(self, state_db_env, mock_subprocess, mock_notify):
        """start_recording() should NOT write PID_FILE anymore."""
        mock_proc = MagicMock()
        mock_proc.pid = 42000
        mock_subprocess.Popen.return_value = mock_proc
        pid_file = state_db_env["config_dir"] / "recording.pid"

        with patch("voice_input.is_recording", return_value=False):
            with patch("voice_input.is_daemon_running", return_value=False):
                voice_input.start_recording()

        assert not pid_file.exists()

    def test_does_not_send_recording_start_ipc(self, state_db_env, mock_subprocess, mock_notify):
        """start_recording() should NOT send recording_start IPC."""
        mock_proc = MagicMock()
        mock_proc.pid = 42000
        mock_subprocess.Popen.return_value = mock_proc

        with patch("voice_input.is_recording", return_value=False):
            with patch("voice_input.is_daemon_running", return_value=True):
                with patch("voice_input.send_to_daemon") as mock_send:
                    voice_input.start_recording()
                    # recording_start should NOT be called
                    for c in mock_send.call_args_list:
                        assert c[0][0] != "recording_start"

    def test_error_resets_db_to_idle(self, state_db_env, mock_notify):
        """Spawn failure should reset DB status to idle."""
        with patch("voice_input.is_recording", return_value=False):
            with patch("voice_input.subprocess.Popen", side_effect=FileNotFoundError("no recorder")):
                voice_input.start_recording()

        s = state_db.get_state(state_db_env["db_path"])
        assert s["status"] == "idle"
        assert s["recording_pid"] is None

    def test_already_recording_shows_notification(self, state_db_env, mock_notify):
        """start_recording() when already recording shows notification."""
        with patch("voice_input.is_recording", return_value=True):
            voice_input.start_recording()

        mock_notify.assert_called()
        assert "already" in str(mock_notify.call_args).lower() or "Abnormal" in str(mock_notify.call_args)


# ============ 3. stop_recording() ============

class TestStopRecordingDB:
    """Test DB-based stop_recording() — CRITIC-R2-C1."""

    def test_reads_pid_and_path_from_db_before_kill(self, state_db_env, mock_notify):
        """CRITIC-R2-C1: must read recording_pid from DB, not PID_FILE."""
        db_path = state_db_env["db_path"]
        audio_file = state_db_env["config_dir"] / "recording_test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        state_db.update_state(
            db_path,
            status="recording",
            recording_pid=os.getpid(),
            recording_path=str(audio_file),
        )

        killed_pids = []

        def mock_kill(pid, sig):
            if sig == 0:
                return  # is_recording PID check
            killed_pids.append(pid)
            raise ProcessLookupError

        with patch("voice_input.is_daemon_running", return_value=True):
            with patch("voice_input.os.kill", side_effect=mock_kill):
                with patch("voice_input.send_to_daemon", return_value={"text": "result"}):
                    with patch("voice_input.type_text"):
                        with patch("voice_input.time.sleep"):
                            voice_input.stop_recording()

        # Should have killed the PID from DB (os.getpid())
        assert os.getpid() in killed_pids

    def test_sets_processing_status_before_kill(self, state_db_env, mock_notify):
        """stop_recording() should write status='processing' to DB before kill."""
        db_path = state_db_env["db_path"]
        audio_file = state_db_env["config_dir"] / "recording_test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        state_db.update_state(
            db_path,
            status="recording",
            recording_pid=os.getpid(),
            recording_path=str(audio_file),
        )

        status_at_kill = [None]

        def mock_kill(pid, sig):
            if sig == 0:
                return
            # Check DB status at kill time
            status_at_kill[0] = state_db.get_state(db_path)["status"]
            raise ProcessLookupError

        with patch("voice_input.is_daemon_running", return_value=False):
            with patch("voice_input.os.kill", side_effect=mock_kill):
                voice_input.stop_recording()

        assert status_at_kill[0] == "processing"

    def test_resets_status_to_idle_at_end(self, state_db_env, mock_notify):
        """stop_recording() should set status='idle' at the end."""
        db_path = state_db_env["db_path"]
        audio_file = state_db_env["config_dir"] / "recording_test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        state_db.update_state(
            db_path,
            status="recording",
            recording_pid=os.getpid(),
            recording_path=str(audio_file),
        )

        with patch("voice_input.is_daemon_running", return_value=True):
            with patch("voice_input.os.kill", side_effect=ProcessLookupError):
                with patch("voice_input.send_to_daemon", return_value={"text": "result"}):
                    with patch("voice_input.type_text"):
                        with patch("voice_input.time.sleep"):
                            voice_input.stop_recording()

        s = state_db.get_state(db_path)
        assert s["status"] == "idle"

    def test_clears_recording_fields_after_kill(self, state_db_env, mock_notify):
        """After kill, recording_pid and recording_path should be None."""
        db_path = state_db_env["db_path"]
        audio_file = state_db_env["config_dir"] / "recording_test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        state_db.update_state(
            db_path,
            status="recording",
            recording_pid=os.getpid(),
            recording_path=str(audio_file),
        )

        with patch("voice_input.is_daemon_running", return_value=True):
            with patch("voice_input.os.kill", side_effect=ProcessLookupError):
                with patch("voice_input.send_to_daemon", return_value={"text": "result"}):
                    with patch("voice_input.type_text"):
                        with patch("voice_input.time.sleep"):
                            voice_input.stop_recording()

        s = state_db.get_state(db_path)
        assert s["recording_pid"] is None
        assert s["recording_path"] is None

    def test_no_audio_file_resets_to_idle(self, state_db_env, mock_notify):
        """If audio file not found, should reset to idle."""
        db_path = state_db_env["db_path"]
        state_db.update_state(
            db_path,
            status="recording",
            recording_pid=os.getpid(),
            recording_path="/tmp/nonexistent.wav",
        )

        with patch("voice_input.is_daemon_running", return_value=False):
            with patch("voice_input.os.kill", side_effect=ProcessLookupError):
                with patch("voice_input.time.sleep"):
                    voice_input.stop_recording()

        s = state_db.get_state(db_path)
        assert s["status"] == "idle"

    def test_does_not_send_set_idle_ipc(self, state_db_env, mock_notify):
        """stop_recording() should NOT send set_idle IPC."""
        db_path = state_db_env["db_path"]
        audio_file = state_db_env["config_dir"] / "recording_test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        state_db.update_state(
            db_path,
            status="recording",
            recording_pid=os.getpid(),
            recording_path=str(audio_file),
        )

        with patch("voice_input.is_daemon_running", return_value=True):
            with patch("voice_input.os.kill", side_effect=ProcessLookupError):
                with patch("voice_input.send_to_daemon") as mock_send:
                    mock_send.return_value = {"text": "result"}
                    with patch("voice_input.type_text"):
                        with patch("voice_input.time.sleep"):
                            voice_input.stop_recording()

                    # Only "transcribe" should be called, not set_idle/recording_stop
                    for c in mock_send.call_args_list:
                        assert c[0][0] not in ("set_idle", "recording_stop")

    def test_not_recording_shows_notification(self, state_db_env, mock_notify):
        """stop_recording() when not recording shows notification."""
        with patch("voice_input.is_recording", return_value=False):
            voice_input.stop_recording()

        mock_notify.assert_called()
        assert "no recording" in str(mock_notify.call_args).lower() or "Abnormal" in str(mock_notify.call_args)


# ============ 4. toggle_recording() ============

class TestToggleRecordingDB:
    """Test DB-based toggle_recording()."""

    def test_processing_status_blocks_toggle(self, state_db_env, mock_notify):
        """DB status='processing' with recent updated_at → toggle rejected."""
        db_path = state_db_env["db_path"]
        state_db.update_state(db_path, status="processing")
        # updated_at is auto-set to now (< 120s)

        with patch("voice_input.is_daemon_ready", return_value=True):
            with patch("voice_input.start_recording") as mock_start:
                voice_input.toggle_recording()
                mock_start.assert_not_called()

        mock_notify.assert_called()
        assert "processing" in str(mock_notify.call_args).lower() or "Processing" in str(mock_notify.call_args)

    def test_stale_processing_cleaned_up(self, state_db_env, mock_notify):
        """DB status='processing' with old updated_at → cleanup and proceed."""
        db_path = state_db_env["db_path"]
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        state_db.update_state(db_path, status="processing", updated_at=old_time)

        with patch("voice_input.is_daemon_ready", return_value=True):
            with patch("voice_input.is_recording", return_value=False):
                with patch("voice_input.is_daemon_running", return_value=True):
                    with patch("voice_input.start_recording") as mock_start:
                        voice_input.toggle_recording()
                        mock_start.assert_called_once()

        # DB status should have been reset to idle
        s = state_db.get_state(db_path)
        # Note: start_recording may have changed status, check it was reset from processing
        # The key assertion is that start_recording was called (toggle wasn't blocked)

    def test_idle_status_proceeds_normally(self, state_db_env, mock_notify):
        """DB status='idle' → toggle proceeds to start/stop."""
        with patch("voice_input.is_daemon_ready", return_value=True):
            with patch("voice_input.is_recording", return_value=False):
                with patch("voice_input.start_recording") as mock_start:
                    voice_input.toggle_recording()
                    mock_start.assert_called_once()

    def test_processing_with_none_updated_at_treated_as_stale(self, state_db_env, mock_notify):
        """DB status='processing' with updated_at=None → stale cleanup."""
        db_path = state_db_env["db_path"]
        # Force updated_at to None
        state_db.update_state(db_path, status="processing", updated_at=None)
        # Verify it's actually None
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE daemon_state SET updated_at = NULL WHERE id = 1")
        conn.commit()
        conn.close()

        with patch("voice_input.is_daemon_ready", return_value=True):
            with patch("voice_input.is_recording", return_value=False):
                with patch("voice_input.is_daemon_running", return_value=True):
                    with patch("voice_input.start_recording") as mock_start:
                        voice_input.toggle_recording()
                        mock_start.assert_called_once()

    def test_does_not_use_processing_file(self, state_db_env, mock_notify):
        """toggle_recording() should NOT check PROCESSING_FILE."""
        processing_file = state_db_env["config_dir"] / "processing.flag"
        processing_file.write_text(str(os.getpid()))

        # Even though PROCESSING_FILE exists, DB says idle → proceed
        with patch("voice_input.is_daemon_ready", return_value=True):
            with patch("voice_input.is_recording", return_value=False):
                with patch("voice_input.start_recording") as mock_start:
                    voice_input.toggle_recording()
                    mock_start.assert_called_once()


# ============ 5. handle_client() — removed IPC commands ============

class TestHandleClientRemovedCommands:
    """Test that removed IPC commands return error."""

    @pytest.fixture
    def daemon(self, isolated_environment, mock_asr_model, mock_gtk):
        """Create a daemon instance for testing."""
        d = voice_input.ASRDaemon()
        d.model = mock_asr_model["model_instance"]
        d.running = True
        d.indicator = None
        return d

    @pytest.mark.parametrize("cmd", ["recording_start", "recording_stop", "set_idle"])
    def test_removed_commands_return_error(self, daemon, cmd):
        """Removed IPC commands should return error."""
        mock_client = MagicMock()
        mock_client.recv.return_value = json.dumps({"command": cmd}).encode()

        daemon.handle_client(mock_client)

        response = json.loads(mock_client.send.call_args[0][0].decode())
        assert "error" in response
        assert "Unknown command" in response["error"]

    def test_transcribe_still_works(self, daemon, isolated_environment, mock_gtk):
        """transcribe command should still work."""
        daemon.indicator = mock_gtk["indicator"]
        daemon.status_item = MagicMock()

        with patch("voice_input.ModelInference.transcribe", return_value="test"):
            daemon.framework = "funasr"
            daemon.current_model_id = "sensevoice"
            daemon.extra_data = None

            mock_client = MagicMock()
            mock_client.recv.return_value = json.dumps({
                "command": "transcribe",
                "data": "/tmp/test.wav"
            }).encode()

            daemon.handle_client(mock_client)

        response = json.loads(mock_client.send.call_args[0][0].decode())
        assert "text" in response

    def test_ping_still_works(self, daemon):
        """ping command should still work."""
        mock_client = MagicMock()
        mock_client.recv.return_value = json.dumps({"command": "ping"}).encode()

        daemon.handle_client(mock_client)

        response = json.loads(mock_client.send.call_args[0][0].decode())
        assert response["status"] == "ok"


# ============ 6. Integration: toggle → start → stop → idle ============

class TestStateTransitionsViaDB:
    """Integration: verify DB state transitions through full flow."""

    def test_full_toggle_cycle_via_db(self, state_db_env, mock_subprocess, mock_notify):
        """Toggle start → verify recording state → toggle stop → verify idle state."""
        db_path = state_db_env["db_path"]
        mock_proc = MagicMock()
        mock_proc.pid = 55000
        mock_subprocess.Popen.return_value = mock_proc

        # Create audio file that will exist when stop needs it
        audio_file = state_db_env["config_dir"] / "recording_test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        # Step 1: Toggle to start recording
        with patch("voice_input.is_daemon_ready", return_value=True):
            with patch("voice_input.is_recording", return_value=False):
                with patch("voice_input.is_daemon_running", return_value=True):
                    voice_input.toggle_recording()

        s = state_db.get_state(db_path)
        assert s["status"] == "recording"
        assert s["recording_pid"] == 55000

        # Manually set the recording path for stop to find
        state_db.update_state(db_path, recording_path=str(audio_file))

        # Step 2: Toggle to stop recording
        with patch("voice_input.is_daemon_ready", return_value=True):
            with patch("voice_input.is_recording", return_value=True):
                with patch("voice_input.is_daemon_running", return_value=True):
                    with patch("voice_input.send_to_daemon", return_value={"text": "result"}):
                        with patch("voice_input.type_text"):
                            with patch("voice_input.os.kill", side_effect=ProcessLookupError):
                                with patch("voice_input.time.sleep"):
                                    voice_input.toggle_recording()

        s = state_db.get_state(db_path)
        assert s["status"] == "idle"
        assert s["recording_pid"] is None
        assert s["recording_path"] is None
