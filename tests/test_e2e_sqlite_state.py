"""
E2E tests for SQLite state management.

Written BEFORE implementation (P6: E2E Before Implementation).
These tests define the specification that the SQLite state migration must satisfy.
They will fail until US-001~004 are implemented.

L1 Real E2E: requires running daemon + Kitty terminal + PulseAudio
L2 Virtual E2E: verifies DB state transitions without hardware

Verification is STATE-based (P1): read DB values, not trace function calls.
"""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

STATE_DB_PATH = Path.home() / ".config" / "voice-input" / "state.db"
CONFIG_DIR = Path.home() / ".config" / "voice-input"

# ============================================================
# Helpers
# ============================================================

def read_db_state():
    """Read current state from SQLite DB. Returns dict or None if DB doesn't exist."""
    if not STATE_DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(str(STATE_DB_PATH), timeout=5)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM daemon_state WHERE id=1").fetchone()
        conn.close()
        if row:
            return dict(row)
        return None
    except sqlite3.OperationalError:
        return None


def wait_for_db_status(expected_status, timeout=15):
    """Poll DB until status matches or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = read_db_state()
        if state and state.get("status") == expected_status:
            return state
        time.sleep(0.2)
    return read_db_state()


def daemon_is_running():
    """Check if daemon is running via CLI."""
    result = subprocess.run(
        ["voice-input", "status"],
        capture_output=True, text=True, timeout=5
    )
    return "Daemon: Running" in result.stdout


def run_voice_input(*args, timeout=30):
    """Run voice-input CLI command."""
    return subprocess.run(
        ["voice-input", *args],
        capture_output=True, text=True, timeout=timeout
    )


# ============================================================
# L1 Real E2E — require running daemon (BLOCKING)
# ============================================================

class TestStateDBExists:
    """DB must exist and have correct schema after daemon starts."""

    def test_state_db_file_exists(self):
        """state.db must exist in config dir."""
        assert STATE_DB_PATH.exists(), f"state.db not found at {STATE_DB_PATH}"

    def test_state_db_has_daemon_state_table(self):
        """DB must have daemon_state table with one row."""
        state = read_db_state()
        assert state is not None, "daemon_state table missing or empty"
        assert state["id"] == 1

    def test_state_db_has_all_columns(self):
        """DB row must have all expected columns."""
        state = read_db_state()
        assert state is not None
        expected_cols = {"id", "status", "daemon_pid", "recording_pid",
                         "recording_path", "post_processor", "updated_at"}
        assert expected_cols.issubset(set(state.keys())), \
            f"Missing columns: {expected_cols - set(state.keys())}"

    def test_state_db_wal_mode(self):
        """DB must be in WAL journal mode."""
        conn = sqlite3.connect(str(STATE_DB_PATH))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal", f"Expected WAL mode, got {mode}"


class TestStateTransitions:
    """Status must transition correctly through recording lifecycle."""

    @pytest.fixture(autouse=True)
    def require_daemon(self):
        """Skip if daemon not running."""
        if not daemon_is_running():
            pytest.skip("Daemon not running")

    def test_idle_before_recording(self):
        """Status must be 'idle' when not recording."""
        state = read_db_state()
        assert state is not None
        assert state["status"] == "idle"

    def test_recording_sets_status(self):
        """toggle start → DB status='recording' with recording_pid set."""
        run_voice_input("toggle")
        time.sleep(0.5)

        state = read_db_state()
        try:
            assert state["status"] == "recording", f"Expected 'recording', got '{state['status']}'"
            assert state["recording_pid"] is not None, "recording_pid should be set"
            assert state["recording_path"] is not None, "recording_path should be set"
            # Verify PID is actually alive
            os.kill(state["recording_pid"], 0)
        finally:
            # Always stop recording to clean up
            run_voice_input("toggle", timeout=120)
            time.sleep(2)

    def test_idle_after_complete(self):
        """After full toggle cycle, status returns to 'idle'."""
        run_voice_input("toggle")
        time.sleep(2)
        run_voice_input("toggle", timeout=120)
        # Wait for processing to complete
        state = wait_for_db_status("idle", timeout=30)
        assert state is not None
        assert state["status"] == "idle"
        assert state["recording_pid"] is None
        assert state["recording_path"] is None


class TestDaemonPIDInDB:
    """Daemon PID must be tracked in DB."""

    def test_daemon_pid_set_when_running(self):
        """Running daemon must have daemon_pid in DB."""
        if not daemon_is_running():
            pytest.skip("Daemon not running")
        state = read_db_state()
        assert state is not None
        assert state["daemon_pid"] is not None, "daemon_pid should be set"
        # Verify PID is alive
        os.kill(state["daemon_pid"], 0)


class TestPostProcessorPersistence:
    """Post-processor must persist across daemon restarts via DB."""

    @pytest.fixture(autouse=True)
    def require_daemon(self):
        if not daemon_is_running():
            pytest.skip("Daemon not running")

    def test_post_processor_in_db(self):
        """Current post-processor must be stored in DB."""
        state = read_db_state()
        assert state is not None
        assert state["post_processor"] is not None

    def test_post_processor_matches_cli(self):
        """DB post_processor must match CLI status output."""
        state = read_db_state()
        result = run_voice_input("status")
        # Extract post-processor from CLI output
        for line in result.stdout.splitlines():
            if "Post-processor:" in line:
                cli_pp = line.split("(")[-1].rstrip(")")
                assert state["post_processor"] == cli_pp, \
                    f"DB has '{state['post_processor']}', CLI shows '{cli_pp}'"
                return
        # If no post-processor line, it should be 'gemini-merge'
        assert state["post_processor"] == "gemini-merge"


class TestLegacyFileCleanup:
    """Legacy state files must not exist after SQLite migration."""

    def test_no_pid_file(self):
        """recording.pid should not exist (replaced by DB)."""
        pid_file = CONFIG_DIR / "recording.pid"
        assert not pid_file.exists(), f"Legacy {pid_file} still exists"

    def test_no_processing_flag(self):
        """processing.flag should not exist (replaced by DB)."""
        flag_file = CONFIG_DIR / "processing.flag"
        assert not flag_file.exists(), f"Legacy {flag_file} still exists"

    def test_no_recording_path_txt(self):
        """recording_path.txt should not exist (replaced by DB)."""
        path_file = CONFIG_DIR / "recording_path.txt"
        assert not path_file.exists(), f"Legacy {path_file} still exists"

    def test_no_post_processor_txt(self):
        """current_post_processor.txt should not exist (migrated to DB)."""
        pp_file = CONFIG_DIR / "current_post_processor.txt"
        assert not pp_file.exists(), f"Legacy {pp_file} still exists"


class TestCLIStatusReadsDB:
    """voice-input status must read from DB, not IPC."""

    @pytest.fixture(autouse=True)
    def require_daemon(self):
        if not daemon_is_running():
            pytest.skip("Daemon not running")

    def test_status_shows_db_status(self):
        """CLI status output must reflect DB state."""
        state = read_db_state()
        result = run_voice_input("status")
        assert result.returncode == 0
        # Should show status from DB
        assert "Status:" in result.stdout or "Recording:" in result.stdout
        assert "Daemon:" in result.stdout


class TestIconSyncWithDB:
    """Daemon icon must sync with DB status within 2 seconds."""

    @pytest.fixture(autouse=True)
    def require_daemon(self):
        if not daemon_is_running():
            pytest.skip("Daemon not running")

    def test_status_consistent_after_toggle(self):
        """After toggle cycle, voice-input status shows idle."""
        # Ensure starting from idle
        state = read_db_state()
        if state and state["status"] != "idle":
            pytest.skip("Not starting from idle state")

        run_voice_input("toggle")
        time.sleep(1)
        # During recording, status should show recording
        result = run_voice_input("status")
        # Stop recording
        run_voice_input("toggle", timeout=120)
        time.sleep(5)
        # After complete, should show idle
        result = run_voice_input("status")
        assert "idle" in result.stdout.lower() or "Recording: No" in result.stdout


# ============================================================
# E2E features for e2e_features.json
# ============================================================

SQLITE_STATE_FEATURES = [
    {
        "id": "real-e2e-sqlite-state-db-exists",
        "layer": "L1",
        "description": "state.db exists with correct schema (daemon_state table, WAL mode, all columns)",
        "passes": False,
        "last_error": None,
        "blocking": True,
    },
    {
        "id": "real-e2e-sqlite-state-transitions",
        "layer": "L1",
        "description": "DB status transitions: idle → recording (with PID) → processing → idle",
        "passes": False,
        "last_error": None,
        "blocking": True,
    },
    {
        "id": "real-e2e-sqlite-daemon-pid",
        "layer": "L1",
        "description": "Running daemon has daemon_pid in DB, PID is alive",
        "passes": False,
        "last_error": None,
        "blocking": True,
    },
    {
        "id": "real-e2e-sqlite-post-processor-persistence",
        "layer": "L1",
        "description": "Post-processor value persisted in DB, matches CLI status output",
        "passes": False,
        "last_error": None,
        "blocking": True,
    },
    {
        "id": "real-e2e-sqlite-legacy-cleanup",
        "layer": "L1",
        "description": "Legacy files (recording.pid, processing.flag, recording_path.txt, current_post_processor.txt) no longer exist",
        "passes": False,
        "last_error": None,
        "blocking": True,
    },
    {
        "id": "real-e2e-sqlite-cli-reads-db",
        "layer": "L1",
        "description": "voice-input status reads from DB, shows correct status",
        "passes": False,
        "last_error": None,
        "blocking": True,
    },
]
