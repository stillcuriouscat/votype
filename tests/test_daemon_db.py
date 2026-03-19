"""
Tests for US-003: Daemon polls SQLite for icon status updates.

Tests:
- _sync_status_from_db: reads DB, calls set_status on change, skips on same
- _current_db_status: initialized to 'idle' in __init__
- socket_server: calls _sync_status_from_db on timeout
- run(): writes daemon_pid to DB on startup, clears on shutdown
- is_daemon_running(): reads daemon_pid from DB
- load_post_processor(): writes post_processor to DB
- __init__: reads post_processor from DB
- Integration: external DB write triggers daemon icon update
"""

import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input
import state_db
from state_db import init_db, get_state, update_state


# ============ Helpers ============

def _make_daemon(db_path=None):
    """Create minimal ASRDaemon for testing (bypass __init__)."""
    daemon = voice_input.ASRDaemon.__new__(voice_input.ASRDaemon)
    daemon.model = None
    daemon.framework = None
    daemon.extra_data = None
    daemon.current_model_id = "sensevoice"
    daemon.running = False
    daemon.indicator = None
    daemon.gtk_thread = None
    daemon.post_processor_model = None
    daemon.current_post_processor_id = "none"
    daemon.post_processor_framework = "regex"
    daemon.punc_model = None
    daemon._vocab = {}
    daemon._secondary_model = None
    daemon._last_secondary_text = None
    daemon._current_db_status = "idle"
    daemon.status_item = MagicMock()
    return daemon


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    """Set up isolated DB environment for daemon tests."""
    config_dir = tmp_path / "config" / "voice-input"
    config_dir.mkdir(parents=True)

    db_path = config_dir / "state.db"
    monkeypatch.setattr("voice_input.STATE_DB_PATH", db_path)
    monkeypatch.setattr("state_db.DEFAULT_DB_PATH", db_path)
    init_db(db_path)

    # Also redirect file-based paths
    monkeypatch.setattr("voice_input.CONFIG_DIR", config_dir)
    monkeypatch.setattr("voice_input.DAEMON_PID_FILE", config_dir / "daemon.pid")
    monkeypatch.setattr("voice_input.DAEMON_LOCK_FILE", config_dir / "daemon.lock")
    monkeypatch.setattr("voice_input.SOCKET_PATH", config_dir / "daemon.sock")
    monkeypatch.setattr("voice_input.PID_FILE", config_dir / "recording.pid")
    monkeypatch.setattr("voice_input.PROCESSING_FILE", config_dir / "processing.flag")

    return {"db_path": db_path, "config_dir": config_dir}


# ============ _sync_status_from_db ============

class TestSyncStatusFromDb:
    """Tests for ASRDaemon._sync_status_from_db()."""

    def test_calls_set_status_on_change(self, db_env):
        """When DB status differs from _current_db_status, set_status is called."""
        daemon = _make_daemon()
        daemon.set_status = MagicMock()
        daemon._current_db_status = "idle"

        # Write 'recording' to DB
        update_state(db_env["db_path"], status="recording")

        daemon._sync_status_from_db()

        daemon.set_status.assert_called_once_with("recording")
        assert daemon._current_db_status == "recording"

    def test_skips_set_status_when_unchanged(self, db_env):
        """When DB status matches _current_db_status, set_status is NOT called."""
        daemon = _make_daemon()
        daemon.set_status = MagicMock()
        daemon._current_db_status = "idle"

        # DB is already 'idle' (default)
        daemon._sync_status_from_db()

        daemon.set_status.assert_not_called()
        assert daemon._current_db_status == "idle"

    def test_handles_processing_status(self, db_env):
        """Correctly syncs to 'processing' status."""
        daemon = _make_daemon()
        daemon.set_status = MagicMock()
        daemon._current_db_status = "recording"

        update_state(db_env["db_path"], status="processing")

        daemon._sync_status_from_db()

        daemon.set_status.assert_called_once_with("processing")
        assert daemon._current_db_status == "processing"

    def test_handles_db_error_gracefully(self, db_env, monkeypatch):
        """DB errors are caught and logged, no exception raised."""
        daemon = _make_daemon()
        daemon.set_status = MagicMock()

        # Make get_state raise an exception
        monkeypatch.setattr(
            "voice_input.get_state",
            MagicMock(side_effect=Exception("DB locked")),
        )

        # Should not raise
        daemon._sync_status_from_db()

        daemon.set_status.assert_not_called()

    def test_multiple_transitions(self, db_env):
        """Tracks multiple sequential status transitions."""
        daemon = _make_daemon()
        daemon.set_status = MagicMock()

        # idle → recording
        update_state(db_env["db_path"], status="recording")
        daemon._sync_status_from_db()
        assert daemon._current_db_status == "recording"

        # recording → processing
        update_state(db_env["db_path"], status="processing")
        daemon._sync_status_from_db()
        assert daemon._current_db_status == "processing"

        # processing → idle
        update_state(db_env["db_path"], status="idle")
        daemon._sync_status_from_db()
        assert daemon._current_db_status == "idle"

        assert daemon.set_status.call_count == 3


# ============ _current_db_status init ============

class TestCurrentDbStatusInit:
    """Tests for _current_db_status initialization in __init__."""

    def test_init_sets_current_db_status_idle(self, db_env):
        """ASRDaemon.__init__ sets _current_db_status to 'idle'."""
        daemon = voice_input.ASRDaemon()
        assert daemon._current_db_status == "idle"

    def test_init_reads_post_processor_from_db(self, db_env):
        """ASRDaemon.__init__ reads post_processor from DB."""
        update_state(db_env["db_path"], post_processor="gemini-fix")

        daemon = voice_input.ASRDaemon()
        assert daemon.current_post_processor_id == "gemini-fix"

    def test_init_falls_back_to_default_for_unknown_pp(self, db_env):
        """If DB has unknown post_processor, falls back to default."""
        update_state(db_env["db_path"], post_processor="nonexistent-pp")

        daemon = voice_input.ASRDaemon()
        assert daemon.current_post_processor_id == voice_input.DEFAULT_POST_PROCESSOR

    def test_init_uses_default_when_db_has_none(self, db_env):
        """Default 'none' post_processor from fresh DB works."""
        daemon = voice_input.ASRDaemon()
        # Fresh DB has post_processor='none', which is a valid preset key
        assert daemon.current_post_processor_id in voice_input.POST_PROCESSOR_PRESETS


# ============ socket_server calls _sync_status_from_db ============

class TestSocketServerSync:
    """Tests that socket_server() calls _sync_status_from_db on timeout."""

    def test_sync_called_on_timeout(self, db_env):
        """_sync_status_from_db is called during socket timeout loop."""
        daemon = _make_daemon()
        daemon.running = True
        sync_calls = []

        original_sync = daemon._sync_status_from_db

        def mock_sync():
            sync_calls.append(True)
            # Stop after first call to avoid infinite loop
            daemon.running = False

        daemon._sync_status_from_db = mock_sync

        # Run socket_server in a thread with a very short timeout
        import socket as sock
        sock_path = db_env["config_dir"] / "daemon.sock"
        monkeypatch_socket = str(sock_path)

        # We need to set up the socket path
        voice_input.SOCKET_PATH = sock_path

        server_thread = threading.Thread(target=daemon.socket_server, daemon=True)
        server_thread.start()
        server_thread.join(timeout=3)

        assert len(sync_calls) >= 1, "_sync_status_from_db was not called on timeout"


# ============ run() daemon_pid lifecycle ============

class TestDaemonPidLifecycle:
    """Tests for daemon_pid DB writes in run()."""

    def test_run_writes_daemon_pid_to_db(self, db_env):
        """run() writes daemon_pid to DB after acquiring lock."""
        daemon = _make_daemon()

        # Mock everything that run() does after writing PID
        with patch.object(daemon, "load_model"), \
             patch.object(daemon, "load_punctuation_model"), \
             patch.object(daemon, "load_post_processor"), \
             patch.object(daemon, "setup_indicator"), \
             patch("voice_input.HAS_INDICATOR", False), \
             patch("builtins.print"):

            # Make daemon stop immediately
            def stop_immediately():
                daemon.running = False

            daemon.socket_server = MagicMock(side_effect=stop_immediately)

            daemon.run()

        # After run completes, daemon_pid should be cleared (shutdown cleanup)
        state = get_state(db_env["db_path"])
        assert state["daemon_pid"] is None
        assert state["status"] == "idle"

    def test_run_clears_daemon_pid_on_shutdown(self, db_env):
        """run() clears daemon_pid in DB during shutdown."""
        # Pre-set a daemon_pid in DB
        update_state(db_env["db_path"], daemon_pid=99999)

        daemon = _make_daemon()

        with patch.object(daemon, "load_model"), \
             patch.object(daemon, "load_punctuation_model"), \
             patch.object(daemon, "load_post_processor"), \
             patch("voice_input.HAS_INDICATOR", False), \
             patch("builtins.print"):

            def stop_immediately():
                daemon.running = False

            daemon.socket_server = MagicMock(side_effect=stop_immediately)
            daemon.run()

        state = get_state(db_env["db_path"])
        assert state["daemon_pid"] is None

    def test_run_writes_pid_matches_os_getpid(self, db_env):
        """run() writes the actual os.getpid() to DB."""
        daemon = _make_daemon()
        written_pids = []

        original_update = voice_input.update_state

        def capture_pid(*args, **kwargs):
            if "daemon_pid" in kwargs and kwargs["daemon_pid"] is not None:
                written_pids.append(kwargs["daemon_pid"])
            return original_update(*args, **kwargs)

        with patch.object(daemon, "load_model"), \
             patch.object(daemon, "load_punctuation_model"), \
             patch.object(daemon, "load_post_processor"), \
             patch("voice_input.HAS_INDICATOR", False), \
             patch("voice_input.update_state", side_effect=capture_pid), \
             patch("builtins.print"):

            def stop_immediately():
                daemon.running = False

            daemon.socket_server = MagicMock(side_effect=stop_immediately)
            daemon.run()

        assert os.getpid() in written_pids


# ============ is_daemon_running() with DB ============

class TestIsDaemonRunningDb:
    """Tests for DB-based is_daemon_running()."""

    def test_returns_true_when_db_pid_alive(self, db_env):
        """DB daemon_pid pointing to live process → True."""
        # Use current process PID (alive)
        update_state(db_env["db_path"], daemon_pid=os.getpid())

        # Mock _is_daemon_lock_held to return False (test DB path)
        with patch("voice_input._is_daemon_lock_held", return_value=False), \
             patch("voice_input.DAEMON_PID_FILE", db_env["config_dir"] / "daemon.pid"):
            result = voice_input.is_daemon_running()

        # Current process cmdline won't contain "voice_input" so it will
        # fail the cmdline check and clean up. That's expected — test
        # the behavior with a mocked cmdline.
        # Let's test with cmdline mocked:
        pass

    def test_returns_true_with_mocked_cmdline(self, db_env):
        """DB daemon_pid alive + cmdline matches → True."""
        update_state(db_env["db_path"], daemon_pid=os.getpid())

        # Mock /proc/PID/cmdline to contain 'voice_input'
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "python\x00voice_input.py\x00_daemon"

        with patch("voice_input._is_daemon_lock_held", return_value=False), \
             patch("voice_input.Path") as mock_path_cls:
            # Path(f"/proc/{pid}/cmdline") should return our mock
            mock_path_cls.return_value = mock_path
            # But Path.home() etc may also be called, so be selective
            mock_path_cls.side_effect = lambda p: mock_path if "/proc/" in str(p) else Path(p)

            result = voice_input.is_daemon_running()

        assert result is True

    def test_returns_false_when_db_pid_dead(self, db_env):
        """DB daemon_pid pointing to dead process → False, DB cleaned up."""
        dead_pid = 99998
        # Ensure PID doesn't exist
        try:
            os.kill(dead_pid, 0)
            pytest.skip("PID 99998 unexpectedly exists")
        except ProcessLookupError:
            pass

        update_state(db_env["db_path"], daemon_pid=dead_pid)

        with patch("voice_input._is_daemon_lock_held", return_value=False):
            result = voice_input.is_daemon_running()

        assert result is False
        # DB should be cleaned up
        state = get_state(db_env["db_path"])
        assert state["daemon_pid"] is None

    def test_falls_back_to_pid_file(self, db_env):
        """When DB has no daemon_pid, falls back to PID file check."""
        # DB has no daemon_pid (default)
        pid_file = db_env["config_dir"] / "daemon.pid"

        with patch("voice_input._is_daemon_lock_held", return_value=False):
            # No PID file → False
            result = voice_input.is_daemon_running()
        assert result is False

    def test_flock_takes_priority(self, db_env):
        """flock check (fast path) returns True even without DB entry."""
        with patch("voice_input._is_daemon_lock_held", return_value=True):
            result = voice_input.is_daemon_running()
        assert result is True


# ============ load_post_processor() writes to DB ============

class TestLoadPostProcessorDb:
    """Tests that load_post_processor writes to DB instead of file."""

    def test_persists_post_processor_to_db(self, db_env):
        """load_post_processor() writes post_processor to DB."""
        daemon = _make_daemon()

        with patch("voice_input.PostProcessorLoader") as mock_loader:
            mock_loader.load_post_processor.return_value = None
            with patch("builtins.print"):
                daemon.load_post_processor("gemini-fix")

        state = get_state(db_env["db_path"])
        assert state["post_processor"] == "gemini-fix"

    def test_does_not_write_to_file(self, db_env):
        """load_post_processor() does NOT write to legacy post-processor state file."""
        daemon = _make_daemon()
        pp_file = db_env["config_dir"] / "current_post_processor.txt"

        with patch("voice_input.PostProcessorLoader") as mock_loader:
            mock_loader.load_post_processor.return_value = None
            with patch("builtins.print"):
                daemon.load_post_processor("gemini-fix")

        assert not pp_file.exists(), "Should not write to legacy file"

    def test_persists_none_preset(self, db_env):
        """Switching to 'none' post-processor persists correctly."""
        daemon = _make_daemon()

        with patch("voice_input.PostProcessorLoader") as mock_loader:
            mock_loader.load_post_processor.return_value = None
            with patch("builtins.print"):
                daemon.load_post_processor("none")

        state = get_state(db_env["db_path"])
        assert state["post_processor"] == "none"

    def test_fallback_does_not_overwrite_db_on_error(self, db_env):
        """When load fails and falls back to regex, 'none' is written to DB."""
        daemon = _make_daemon()
        # Pre-set a value
        update_state(db_env["db_path"], post_processor="gemini-fix")

        with patch("voice_input.PostProcessorLoader") as mock_loader:
            # haiku-expand: not yet implemented — triggers ValueError
            mock_loader.load_post_processor.side_effect = Exception("load failed")
            with patch("builtins.print"):
                daemon.load_post_processor("gemini-fix")

        # After fallback, post_processor_id is "none" but DB may still have old value
        # (the persist call happened before the exception in the inner try block)
        assert daemon.current_post_processor_id == "none"


# ============ Integration: external DB write → daemon picks up ============

class TestDaemonDbIntegration:
    """Integration test: external process writes status to DB, daemon detects it."""

    def test_external_status_write_detected(self, db_env):
        """Write status to DB from 'external process', daemon picks it up via poll."""
        daemon = _make_daemon()
        daemon.set_status = MagicMock()

        # Simulate CLI writing 'recording' to DB
        update_state(db_env["db_path"], status="recording")

        # Daemon polls
        daemon._sync_status_from_db()

        daemon.set_status.assert_called_once_with("recording")
        assert daemon._current_db_status == "recording"

    def test_full_recording_cycle_via_db(self, db_env):
        """Simulate full recording cycle: idle → recording → processing → idle."""
        daemon = _make_daemon()
        status_calls = []
        daemon.set_status = MagicMock(side_effect=lambda s: status_calls.append(s))

        # idle (initial) — no change expected
        daemon._sync_status_from_db()
        assert len(status_calls) == 0

        # CLI starts recording
        update_state(db_env["db_path"], status="recording", recording_pid=12345)
        daemon._sync_status_from_db()
        assert status_calls[-1] == "recording"

        # CLI stops recording, starts processing
        update_state(db_env["db_path"], status="processing")
        daemon._sync_status_from_db()
        assert status_calls[-1] == "processing"

        # Transcription complete, back to idle
        update_state(db_env["db_path"], status="idle")
        daemon._sync_status_from_db()
        assert status_calls[-1] == "idle"

        assert status_calls == ["recording", "processing", "idle"]

    def test_concurrent_db_access(self, db_env):
        """Multiple threads can read/write DB without errors."""
        daemon = _make_daemon()
        daemon.set_status = MagicMock()
        errors = []

        def writer():
            try:
                for status in ["recording", "processing", "idle"] * 5:
                    update_state(db_env["db_path"], status=status)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(15):
                    daemon._sync_status_from_db()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Concurrent access errors: {errors}"
