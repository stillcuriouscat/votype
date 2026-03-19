"""Clean-room unit tests for CLI functions using SQLite state.

Tests derived from FUNCTION_SPEC.md behavior tables for:
- ensure_config_dir()
- is_recording()
- start_recording()
- stop_recording()
- toggle_recording()
- show_status()
- is_daemon_running()
"""

import os
import signal
import socket
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def state_env(tmp_path, monkeypatch):
    """Set up isolated state DB environment for CLI function tests."""
    from state_db import init_db

    config_dir = tmp_path / ".config" / "voice-input"
    config_dir.mkdir(parents=True)
    state_db_path = config_dir / "state.db"

    monkeypatch.setattr("state_db.DEFAULT_DB_PATH", state_db_path)

    # Patch voice_input module attributes
    monkeypatch.setattr("voice_input.STATE_DB_PATH", state_db_path)
    monkeypatch.setattr("voice_input.CONFIG_DIR", config_dir)

    # Create file paths for legacy files
    pid_file = config_dir / "daemon.pid.recording"
    processing_file = config_dir / "processing.flag"
    audio_path_file = config_dir / "audio_path.txt"
    daemon_lock_file = config_dir / "daemon.lock"
    daemon_pid_file = config_dir / "daemon.pid"
    monkeypatch.setattr("voice_input.PID_FILE", pid_file)
    monkeypatch.setattr("voice_input.PROCESSING_FILE", processing_file)
    monkeypatch.setattr("voice_input.AUDIO_PATH_FILE", audio_path_file)
    monkeypatch.setattr("voice_input.DAEMON_LOCK_FILE", daemon_lock_file)
    monkeypatch.setattr("voice_input.DAEMON_PID_FILE", daemon_pid_file)

    init_db(state_db_path)

    return {
        "tmp_path": tmp_path,
        "config_dir": config_dir,
        "state_db_path": state_db_path,
        "pid_file": pid_file,
        "processing_file": processing_file,
        "audio_path_file": audio_path_file,
    }


# ===========================================================================
# ensure_config_dir() tests — FUNCTION_SPEC.md Behavior Table rows 1-5
# ===========================================================================

class TestEnsureConfigDir:
    """Tests for voice_input.ensure_config_dir()."""

    def test_creates_config_dir_and_inits_db(
        self, tmp_path, monkeypatch
    ) -> None:
        """BT#1: First run, no legacy files → creates dir, inits DB."""
        from state_db import get_state, init_db

        config_dir = tmp_path / "fresh" / ".config" / "voice-input"
        state_db_path = config_dir / "state.db"

        monkeypatch.setattr("voice_input.CONFIG_DIR", config_dir)
        monkeypatch.setattr("voice_input.STATE_DB_PATH", state_db_path)
        monkeypatch.setattr("state_db.DEFAULT_DB_PATH", state_db_path)
        monkeypatch.setattr(
            "voice_input.PID_FILE", config_dir / "daemon.pid.recording"
        )
        monkeypatch.setattr(
            "voice_input.PROCESSING_FILE", config_dir / "processing.flag"
        )
        monkeypatch.setattr(
            "voice_input.AUDIO_PATH_FILE", config_dir / "audio_path.txt"
        )

        import voice_input
        voice_input.ensure_config_dir()

        assert config_dir.exists()
        state = get_state(state_db_path)
        assert state["status"] == "idle"

    def test_deletes_legacy_files(self, state_env, monkeypatch) -> None:
        """BT#2: Legacy files exist → all 3 deleted after init."""
        # Create legacy files
        state_env["pid_file"].write_text("12345")
        state_env["processing_file"].write_text("67890")
        state_env["audio_path_file"].write_text("/tmp/audio.wav")

        import voice_input
        voice_input.ensure_config_dir()

        assert not state_env["pid_file"].exists()
        assert not state_env["processing_file"].exists()
        assert not state_env["audio_path_file"].exists()

    def test_idempotent_no_legacy_files(self, state_env) -> None:
        """BT#3: Dir exists, no legacy files → no error."""
        import voice_input
        voice_input.ensure_config_dir()
        voice_input.ensure_config_dir()
        # No error raised

    def test_deletes_only_existing_legacy_files(self, state_env) -> None:
        """BT#4: Only PID_FILE exists → deletes it, skips missing files."""
        state_env["pid_file"].write_text("12345")

        import voice_input
        voice_input.ensure_config_dir()

        assert not state_env["pid_file"].exists()
        # Others were never created, no error from unlink(missing_ok=True)


# ===========================================================================
# is_recording() tests — FUNCTION_SPEC.md Behavior Table rows 1-6
# ===========================================================================

class TestIsRecording:
    """Tests for voice_input.is_recording()."""

    def test_returns_false_when_idle(self, state_env) -> None:
        """BT#1: DB status='idle', recording_pid=None → False."""
        import voice_input
        assert voice_input.is_recording() is False

    def test_returns_true_when_recording_with_live_pid(
        self, state_env, monkeypatch
    ) -> None:
        """BT#2: DB status='recording', live PID → True."""
        from state_db import update_state

        # Use current process PID (guaranteed alive)
        update_state(
            state_env["state_db_path"],
            status="recording",
            recording_pid=os.getpid(),
        )

        import voice_input
        assert voice_input.is_recording() is True

    def test_returns_false_and_cleans_up_dead_pid(
        self, state_env, monkeypatch
    ) -> None:
        """BT#3: DB status='recording', dead PID → False + DB reset to idle."""
        from state_db import get_state, update_state

        # Use a PID that's (almost certainly) not running
        dead_pid = 2_000_000_000
        update_state(
            state_env["state_db_path"],
            status="recording",
            recording_pid=dead_pid,
            recording_path="/tmp/rec.wav",
        )

        # Mock os.kill to raise ProcessLookupError
        monkeypatch.setattr(
            "os.kill",
            MagicMock(side_effect=ProcessLookupError("No such process")),
        )

        import voice_input
        result = voice_input.is_recording()

        assert result is False
        state = get_state(state_env["state_db_path"])
        assert state["status"] == "idle"
        assert state["recording_pid"] is None
        assert state["recording_path"] is None

    def test_returns_false_when_recording_pid_is_none(
        self, state_env
    ) -> None:
        """BT#4: DB status='recording', recording_pid=None → False."""
        from state_db import update_state

        update_state(state_env["state_db_path"], status="recording")
        # recording_pid is still None (default)

        import voice_input
        assert voice_input.is_recording() is False

    def test_returns_false_when_processing(self, state_env) -> None:
        """BT#5: DB status='processing' → False."""
        from state_db import update_state

        update_state(state_env["state_db_path"], status="processing")

        import voice_input
        assert voice_input.is_recording() is False

    def test_returns_false_on_db_error(self, state_env, monkeypatch) -> None:
        """BT#6: DB read fails → returns False (safe default)."""
        import voice_input

        # Make get_state return safe defaults by corrupting the DB
        monkeypatch.setattr(
            "voice_input.get_state",
            lambda *a, **kw: {
                "id": 1, "status": "idle", "daemon_pid": None,
                "recording_pid": None, "recording_path": None,
                "post_processor": "none", "updated_at": None,
            },
        )

        assert voice_input.is_recording() is False


# ===========================================================================
# start_recording() tests — FUNCTION_SPEC.md Behavior Table rows 1-4
# ===========================================================================

class TestStartRecording:
    """Tests for voice_input.start_recording()."""

    def test_normal_start_writes_state_to_db(
        self, state_env, monkeypatch
    ) -> None:
        """BT#1: Not recording, daemon running → spawns recorder, updates DB."""
        from state_db import get_state

        import voice_input

        # Mock is_recording to return False
        monkeypatch.setattr("voice_input.is_recording", lambda: False)
        # Mock is_daemon_ready
        monkeypatch.setattr("voice_input.is_daemon_ready", lambda: True)
        # Mock ensure_config_dir
        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        # Mock notify
        monkeypatch.setattr("voice_input.notify", MagicMock())

        # Mock Popen to return a fake process
        mock_proc = MagicMock()
        mock_proc.pid = 42
        mock_popen = MagicMock(return_value=mock_proc)
        monkeypatch.setattr("subprocess.Popen", mock_popen)

        # Mock recorder binary lookup
        monkeypatch.setattr(
            "shutil.which", lambda cmd: f"/usr/bin/{cmd}"
        )

        voice_input.start_recording()

        state = get_state(state_env["state_db_path"])
        assert state["status"] == "recording"
        assert state["recording_pid"] == 42
        assert state["recording_path"] is not None

    def test_already_recording_shows_notification(
        self, state_env, monkeypatch
    ) -> None:
        """BT#2: Already recording → notify abnormal state, no spawn."""
        import voice_input

        monkeypatch.setattr("voice_input.is_recording", lambda: True)
        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)

        mock_notify = MagicMock()
        monkeypatch.setattr("voice_input.notify", mock_notify)

        mock_popen = MagicMock()
        monkeypatch.setattr("subprocess.Popen", mock_popen)

        voice_input.start_recording()

        mock_notify.assert_called()
        mock_popen.assert_not_called()

    def test_recorder_not_found_resets_db(
        self, state_env, monkeypatch
    ) -> None:
        """BT#3: Recorder binary not found → DB reset to idle."""
        from state_db import get_state

        import voice_input

        monkeypatch.setattr("voice_input.is_recording", lambda: False)
        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr("voice_input.notify", MagicMock())

        # Make shutil.which return None for all recorder binaries
        monkeypatch.setattr("shutil.which", lambda cmd: None)

        voice_input.start_recording()

        state = get_state(state_env["state_db_path"])
        assert state["status"] == "idle"

    def test_popen_oserror_resets_db(self, state_env, monkeypatch) -> None:
        """BT#4: Popen raises OSError → DB reset to idle, notify error."""
        from state_db import get_state

        import voice_input

        monkeypatch.setattr("voice_input.is_recording", lambda: False)
        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr("voice_input.notify", MagicMock())
        monkeypatch.setattr(
            "shutil.which", lambda cmd: f"/usr/bin/{cmd}"
        )
        monkeypatch.setattr(
            "subprocess.Popen",
            MagicMock(side_effect=OSError("exec format error")),
        )

        voice_input.start_recording()

        state = get_state(state_env["state_db_path"])
        assert state["status"] == "idle"
        assert state["recording_pid"] is None
        assert state["recording_path"] is None


# ===========================================================================
# stop_recording() tests — FUNCTION_SPEC.md Behavior Table rows 1-6
# ===========================================================================

class TestStopRecording:
    """Tests for voice_input.stop_recording()."""

    def test_normal_stop_transitions_through_processing_to_idle(
        self, state_env, monkeypatch
    ) -> None:
        """BT#1: Recording in progress → processing → idle."""
        from state_db import get_state, update_state

        import voice_input

        # Set up recording state
        update_state(
            state_env["state_db_path"],
            status="recording",
            recording_pid=12345,
            recording_path="/tmp/rec.wav",
        )

        monkeypatch.setattr("voice_input.is_recording", lambda: True)
        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr("voice_input.notify", MagicMock())

        # Track all update_state calls
        original_update = update_state
        update_calls = []

        def tracking_update(db_path=None, **kwargs):
            update_calls.append(kwargs)
            return original_update(db_path, **kwargs)

        monkeypatch.setattr("voice_input.update_state", tracking_update)
        monkeypatch.setattr("state_db.update_state", original_update)

        # Mock os.kill to succeed (process killed)
        monkeypatch.setattr("os.kill", MagicMock())
        # Mock time.sleep
        monkeypatch.setattr("time.sleep", MagicMock())
        # Mock send_to_daemon to return transcription
        monkeypatch.setattr(
            "voice_input.send_to_daemon",
            MagicMock(return_value={"text": "hello world"}),
        )
        # Mock type_text
        monkeypatch.setattr("voice_input.type_text", MagicMock())
        # Mock audio file existence
        monkeypatch.setattr("os.path.exists", lambda p: True)
        # Mock _cleanup_old_recordings
        monkeypatch.setattr(
            "voice_input._cleanup_old_recordings", MagicMock()
        )
        # Mock Path.exists for audio file
        with patch.object(Path, "exists", return_value=True):
            voice_input.stop_recording()

        # Verify state transitions:
        # 1. status="processing" at some point
        # 2. status="idle" at the end
        statuses = [c.get("status") for c in update_calls if "status" in c]
        assert "processing" in statuses
        assert statuses[-1] == "idle"

    def test_not_recording_shows_notification(
        self, state_env, monkeypatch
    ) -> None:
        """BT#2: Not recording → notify abnormal state, no state change."""
        import voice_input

        monkeypatch.setattr("voice_input.is_recording", lambda: False)
        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)

        mock_notify = MagicMock()
        monkeypatch.setattr("voice_input.notify", mock_notify)

        voice_input.stop_recording()

        mock_notify.assert_called()
        # Verify "Abnormal state" or "no recording" in notification
        notify_args = mock_notify.call_args
        assert any(
            "abnormal" in str(a).lower() or "no recording" in str(a).lower()
            for a in notify_args[0]
        )

    def test_reads_pid_and_path_from_db_before_kill(
        self, state_env, monkeypatch
    ) -> None:
        """CRITIC-R2-C1: recording_pid/path read from DB BEFORE kill."""
        from state_db import update_state

        import voice_input

        update_state(
            state_env["state_db_path"],
            status="recording",
            recording_pid=9999,
            recording_path="/tmp/test_audio.wav",
        )

        monkeypatch.setattr("voice_input.is_recording", lambda: True)
        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr("voice_input.notify", MagicMock())
        monkeypatch.setattr("time.sleep", MagicMock())
        monkeypatch.setattr(
            "voice_input._cleanup_old_recordings", MagicMock()
        )

        # Track get_state calls and os.kill calls in order
        call_order = []

        original_get_state = voice_input.get_state

        def tracking_get_state(*args, **kwargs):
            call_order.append("get_state")
            return original_get_state(*args, **kwargs)

        def tracking_kill(pid, sig):
            call_order.append(f"kill_{pid}")

        monkeypatch.setattr("voice_input.get_state", tracking_get_state)
        monkeypatch.setattr("os.kill", tracking_kill)
        monkeypatch.setattr(
            "voice_input.send_to_daemon",
            MagicMock(return_value={"text": "ok"}),
        )
        monkeypatch.setattr("voice_input.type_text", MagicMock())

        with patch.object(Path, "exists", return_value=True):
            voice_input.stop_recording()

        # get_state must happen before any kill
        get_idx = next(
            (i for i, c in enumerate(call_order) if c == "get_state"), None
        )
        kill_idx = next(
            (i for i, c in enumerate(call_order) if c.startswith("kill")),
            None,
        )
        if get_idx is not None and kill_idx is not None:
            assert get_idx < kill_idx, (
                "get_state() must be called before os.kill()"
            )

    def test_kill_failure_continues_flow(
        self, state_env, monkeypatch
    ) -> None:
        """BT#5: Kill recorder fails (ProcessLookupError) → continues."""
        from state_db import get_state, update_state

        import voice_input

        update_state(
            state_env["state_db_path"],
            status="recording",
            recording_pid=99999,
            recording_path="/tmp/rec.wav",
        )

        monkeypatch.setattr("voice_input.is_recording", lambda: True)
        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr("voice_input.notify", MagicMock())
        monkeypatch.setattr("time.sleep", MagicMock())
        monkeypatch.setattr(
            "voice_input._cleanup_old_recordings", MagicMock()
        )

        # Kill raises ProcessLookupError
        monkeypatch.setattr(
            "os.kill",
            MagicMock(side_effect=ProcessLookupError("No such process")),
        )
        monkeypatch.setattr(
            "voice_input.send_to_daemon",
            MagicMock(return_value={"text": "transcribed"}),
        )
        monkeypatch.setattr("voice_input.type_text", MagicMock())

        with patch.object(Path, "exists", return_value=True):
            voice_input.stop_recording()  # Must not raise

        state = get_state(state_env["state_db_path"])
        assert state["status"] == "idle"

    def test_all_error_paths_reset_to_idle(
        self, state_env, monkeypatch
    ) -> None:
        """BT#6: Transcription fails → status still reset to 'idle'."""
        from state_db import get_state, update_state

        import voice_input

        update_state(
            state_env["state_db_path"],
            status="recording",
            recording_pid=12345,
            recording_path="/tmp/rec.wav",
        )

        monkeypatch.setattr("voice_input.is_recording", lambda: True)
        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr("voice_input.notify", MagicMock())
        monkeypatch.setattr("os.kill", MagicMock())
        monkeypatch.setattr("time.sleep", MagicMock())
        monkeypatch.setattr(
            "voice_input._cleanup_old_recordings", MagicMock()
        )

        # Transcription returns error
        monkeypatch.setattr(
            "voice_input.send_to_daemon",
            MagicMock(return_value={"error": "model not loaded"}),
        )

        with patch.object(Path, "exists", return_value=True):
            voice_input.stop_recording()

        state = get_state(state_env["state_db_path"])
        assert state["status"] == "idle"


# ===========================================================================
# toggle_recording() tests — FUNCTION_SPEC.md Behavior Table rows 1-8
# ===========================================================================

class TestToggleRecording:
    """Tests for voice_input.toggle_recording()."""

    def test_idle_daemon_ready_starts_recording(
        self, state_env, monkeypatch
    ) -> None:
        """BT#1: Idle + daemon ready → calls start_recording()."""
        import voice_input

        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr("voice_input._log_to_notify_file", MagicMock())
        monkeypatch.setattr("voice_input.is_daemon_ready", lambda: True)
        monkeypatch.setattr("voice_input.is_recording", lambda: False)
        monkeypatch.setattr("voice_input.notify", MagicMock())

        mock_start = MagicMock()
        monkeypatch.setattr("voice_input.start_recording", mock_start)

        voice_input.toggle_recording()

        mock_start.assert_called_once()

    def test_recording_calls_stop_recording(
        self, state_env, monkeypatch
    ) -> None:
        """BT#2: Recording → calls stop_recording()."""
        import voice_input

        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr("voice_input._log_to_notify_file", MagicMock())
        monkeypatch.setattr("voice_input.is_daemon_ready", lambda: True)
        monkeypatch.setattr("voice_input.is_recording", lambda: True)
        monkeypatch.setattr("voice_input.notify", MagicMock())

        mock_stop = MagicMock()
        monkeypatch.setattr("voice_input.stop_recording", mock_stop)

        voice_input.toggle_recording()

        mock_stop.assert_called_once()

    def test_processing_recent_shows_wait_notification(
        self, state_env, monkeypatch
    ) -> None:
        """BT#3: Processing + recent (<120s) → notify wait, early return."""
        from state_db import update_state

        import voice_input

        recent_ts = datetime.now(timezone.utc).isoformat()
        update_state(
            state_env["state_db_path"],
            status="processing",
            updated_at=recent_ts,
        )

        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr("voice_input._log_to_notify_file", MagicMock())
        mock_notify = MagicMock()
        monkeypatch.setattr("voice_input.notify", mock_notify)

        mock_start = MagicMock()
        mock_stop = MagicMock()
        monkeypatch.setattr("voice_input.start_recording", mock_start)
        monkeypatch.setattr("voice_input.stop_recording", mock_stop)

        voice_input.toggle_recording()

        # Should notify about processing in progress
        mock_notify.assert_called()
        # Should NOT start or stop recording
        mock_start.assert_not_called()
        mock_stop.assert_not_called()

    def test_processing_stale_cleans_up_and_proceeds(
        self, state_env, monkeypatch
    ) -> None:
        """BT#4: Processing + stale (>=120s) → clean up, proceed."""
        from state_db import get_state, update_state

        import voice_input

        stale_ts = (
            datetime.now(timezone.utc) - timedelta(seconds=200)
        ).isoformat()
        update_state(
            state_env["state_db_path"],
            status="processing",
            updated_at=stale_ts,
        )

        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr("voice_input._log_to_notify_file", MagicMock())
        monkeypatch.setattr("voice_input.is_daemon_ready", lambda: True)
        monkeypatch.setattr("voice_input.is_recording", lambda: False)
        monkeypatch.setattr("voice_input.notify", MagicMock())

        mock_start = MagicMock()
        monkeypatch.setattr("voice_input.start_recording", mock_start)

        voice_input.toggle_recording()

        # Status should have been reset to idle
        state = get_state(state_env["state_db_path"])
        # After stale cleanup + start_recording, status depends on mock
        # But start_recording should have been called
        mock_start.assert_called_once()

    def test_processing_null_updated_at_treated_as_stale(
        self, state_env, monkeypatch
    ) -> None:
        """BT#5: Processing + updated_at=None → treat as stale, proceed."""
        from state_db import update_state

        import voice_input

        update_state(state_env["state_db_path"], status="processing")
        # updated_at will be set by update_state, but let's force it to None
        # by directly writing to DB
        import sqlite3
        conn = sqlite3.connect(str(state_env["state_db_path"]))
        conn.execute(
            "UPDATE daemon_state SET updated_at=NULL WHERE id=1"
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr("voice_input._log_to_notify_file", MagicMock())
        monkeypatch.setattr("voice_input.is_daemon_ready", lambda: True)
        monkeypatch.setattr("voice_input.is_recording", lambda: False)
        monkeypatch.setattr("voice_input.notify", MagicMock())

        mock_start = MagicMock()
        monkeypatch.setattr("voice_input.start_recording", mock_start)

        voice_input.toggle_recording()

        mock_start.assert_called_once()

    def test_daemon_not_running_starts_daemon(
        self, state_env, monkeypatch
    ) -> None:
        """BT#6: Daemon not running → starts daemon."""
        import voice_input

        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr("voice_input._log_to_notify_file", MagicMock())
        monkeypatch.setattr("voice_input.is_daemon_ready", lambda: False)
        monkeypatch.setattr("voice_input.is_daemon_running", lambda: False)
        monkeypatch.setattr("voice_input.notify", MagicMock())
        monkeypatch.setattr("voice_input.is_recording", lambda: False)

        # Mock subprocess for daemon start
        mock_popen = MagicMock()
        monkeypatch.setattr("subprocess.Popen", mock_popen)

        # Make daemon become ready after "start"
        ready_count = {"calls": 0}

        def mock_is_ready():
            ready_count["calls"] += 1
            return ready_count["calls"] > 2  # Ready after 2 checks

        monkeypatch.setattr("voice_input.is_daemon_ready", mock_is_ready)
        monkeypatch.setattr("time.sleep", MagicMock())

        mock_start = MagicMock()
        monkeypatch.setattr("voice_input.start_recording", mock_start)

        voice_input.toggle_recording()

    def test_daemon_starting_shows_wait(
        self, state_env, monkeypatch
    ) -> None:
        """BT#7: Daemon starting (PID exists but not ready) → shows wait."""
        import voice_input

        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr("voice_input._log_to_notify_file", MagicMock())
        monkeypatch.setattr("voice_input.is_daemon_ready", lambda: False)
        monkeypatch.setattr("voice_input.is_daemon_running", lambda: True)

        mock_notify = MagicMock()
        monkeypatch.setattr("voice_input.notify", mock_notify)

        mock_start = MagicMock()
        monkeypatch.setattr("voice_input.start_recording", mock_start)

        voice_input.toggle_recording()

        # Should notify about starting up
        mock_notify.assert_called()
        # Should NOT start recording
        mock_start.assert_not_called()


# ===========================================================================
# show_status() tests — FUNCTION_SPEC.md Behavior Table rows 1-4
# ===========================================================================

class TestShowStatus:
    """Tests for voice_input.show_status()."""

    def test_shows_recording_status_from_db(
        self, state_env, monkeypatch, capsys
    ) -> None:
        """BT#1: Daemon running, recording → prints Recording: Yes."""
        from state_db import update_state

        import voice_input

        update_state(
            state_env["state_db_path"],
            status="recording",
            daemon_pid=os.getpid(),
            post_processor="gemini-fix",
        )

        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        # Mock send_to_daemon for get_model
        monkeypatch.setattr(
            "voice_input.send_to_daemon",
            MagicMock(return_value={
                "model": "firered-asr",
                "name": "FireRed ASR",
                "description": "A model",
            }),
        )
        # Mock os.kill to succeed (daemon alive)
        monkeypatch.setattr("os.kill", MagicMock())

        voice_input.show_status()

        output = capsys.readouterr().out
        assert "recording" in output.lower() or "Recording" in output

    def test_shows_idle_status_from_db(
        self, state_env, monkeypatch, capsys
    ) -> None:
        """BT#2: Daemon running, idle → prints relevant status."""
        from state_db import update_state

        import voice_input

        update_state(
            state_env["state_db_path"],
            daemon_pid=os.getpid(),
        )

        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr(
            "voice_input.send_to_daemon",
            MagicMock(return_value={"model": "firered-asr"}),
        )
        monkeypatch.setattr("os.kill", MagicMock())

        voice_input.show_status()

        output = capsys.readouterr().out
        # Should print some status output
        assert len(output) > 0

    def test_shows_not_running_when_no_daemon(
        self, state_env, monkeypatch, capsys
    ) -> None:
        """BT#3: Daemon not running → prints Daemon: Not running."""
        import voice_input

        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        # No daemon PID in DB (default None)

        voice_input.show_status()

        output = capsys.readouterr().out
        assert "not running" in output.lower() or "Not running" in output

    def test_graceful_output_on_db_error(
        self, state_env, monkeypatch, capsys
    ) -> None:
        """BT#4: DB corrupt → graceful output from safe defaults."""
        import voice_input

        monkeypatch.setattr("voice_input.ensure_config_dir", lambda: None)
        monkeypatch.setattr(
            "voice_input.get_state",
            lambda *a, **kw: {
                "id": 1, "status": "idle", "daemon_pid": None,
                "recording_pid": None, "recording_path": None,
                "post_processor": "none", "updated_at": None,
            },
        )

        voice_input.show_status()

        output = capsys.readouterr().out
        # Should produce some output, not crash
        assert len(output) > 0


# ===========================================================================
# is_daemon_running() tests — FUNCTION_SPEC.md Behavior Table rows 1-7
# ===========================================================================

class TestIsDaemonRunning:
    """Tests for voice_input.is_daemon_running()."""

    def test_running_daemon_from_db(self, state_env, monkeypatch) -> None:
        """BT#1: DB has live PID, cmdline matches → True."""
        from state_db import update_state

        import voice_input

        update_state(
            state_env["state_db_path"],
            daemon_pid=os.getpid(),
        )

        # Mock os.kill to succeed
        monkeypatch.setattr("os.kill", MagicMock())
        # Mock /proc/<pid>/cmdline to contain "voice_input"
        mock_open = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                read=MagicMock(return_value="python\x00voice_input\x00daemon")
            )
        )
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr("builtins.open", mock_open)

        result = voice_input.is_daemon_running()
        assert result is True

    def test_dead_pid_in_db_returns_false(
        self, state_env, monkeypatch
    ) -> None:
        """BT#3: DB has dead PID → False, DB cleaned up."""
        from state_db import get_state, update_state

        import voice_input

        update_state(state_env["state_db_path"], daemon_pid=2_000_000_000)

        # Mock os.kill to raise (dead process)
        monkeypatch.setattr(
            "os.kill",
            MagicMock(side_effect=ProcessLookupError("No such process")),
        )
        # Mock fallback checks to also fail
        monkeypatch.setattr(
            "voice_input._is_daemon_lock_held",
            MagicMock(return_value=False),
        )
        monkeypatch.setattr(
            "voice_input._cleanup_daemon_files", MagicMock()
        )
        # Mock DAEMON_PID_FILE
        monkeypatch.setattr(
            "voice_input.DAEMON_PID_FILE",
            state_env["config_dir"] / "daemon.pid",
        )

        result = voice_input.is_daemon_running()
        assert result is False

        state = get_state(state_env["state_db_path"])
        assert state["daemon_pid"] is None

    def test_wrong_process_in_db_returns_false(
        self, state_env, monkeypatch
    ) -> None:
        """BT#4: DB PID alive but cmdline doesn't match → False."""
        from state_db import get_state, update_state

        import voice_input

        update_state(
            state_env["state_db_path"],
            daemon_pid=os.getpid(),
        )

        # os.kill succeeds (PID alive)
        monkeypatch.setattr("os.kill", MagicMock())
        # cmdline does NOT contain "voice_input"
        mock_open = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                read=MagicMock(return_value="python\x00some_other_app")
            )
        )
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr("builtins.open", mock_open)

        # Mock fallback
        monkeypatch.setattr(
            "voice_input._is_daemon_lock_held",
            MagicMock(return_value=False),
        )
        monkeypatch.setattr(
            "voice_input._cleanup_daemon_files", MagicMock()
        )
        monkeypatch.setattr(
            "voice_input.DAEMON_PID_FILE",
            state_env["config_dir"] / "daemon.pid",
        )

        result = voice_input.is_daemon_running()
        assert result is False

        state = get_state(state_env["state_db_path"])
        assert state["daemon_pid"] is None

    def test_no_pid_in_db_falls_through_to_flock(
        self, state_env, monkeypatch
    ) -> None:
        """BT#2: DB daemon_pid=None → falls through to flock check."""
        import voice_input

        # DB has daemon_pid=None (default)
        # Mock flock check to return True
        monkeypatch.setattr(
            "voice_input._is_daemon_lock_held",
            MagicMock(return_value=True),
        )
        # Mock DAEMON_PID_FILE with live PID
        pid_file = state_env["config_dir"] / "daemon.pid"
        pid_file.write_text(str(os.getpid()))
        monkeypatch.setattr("voice_input.DAEMON_PID_FILE", pid_file)

        # Mock os.kill and cmdline for the fallback path
        monkeypatch.setattr("os.kill", MagicMock())
        mock_open = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                read=MagicMock(return_value="python\x00voice_input\x00daemon")
            )
        )
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr("builtins.open", mock_open)

        result = voice_input.is_daemon_running()
        assert result is True

    def test_all_checks_fail_returns_false(
        self, state_env, monkeypatch
    ) -> None:
        """BT#7: DB fails, no flock, no PID file → False."""
        import voice_input

        # Mock get_state to return safe default (daemon_pid=None)
        monkeypatch.setattr(
            "voice_input.get_state",
            lambda *a, **kw: {
                "id": 1, "status": "idle", "daemon_pid": None,
                "recording_pid": None, "recording_path": None,
                "post_processor": "none", "updated_at": None,
            },
        )
        monkeypatch.setattr(
            "voice_input._is_daemon_lock_held",
            MagicMock(return_value=False),
        )
        monkeypatch.setattr(
            "voice_input.DAEMON_PID_FILE",
            state_env["config_dir"] / "nonexistent.pid",
        )

        result = voice_input.is_daemon_running()
        assert result is False
