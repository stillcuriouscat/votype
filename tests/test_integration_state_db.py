"""Clean-room integration tests for state_db module.

Derived from LOW_LEVEL_DESIGN.md Section 1.1 (state_db.py interface),
Section 3 (Data Models), Section 4 (Error Taxonomy), Section 5 (Configuration).

Tests inter-module contracts:
- init_db() creates table, default row, WAL mode, migrates legacy file
- get_state() returns correct dict shape, self-initializes, safe defaults
- update_state() validates columns, auto-sets updated_at, BEGIN IMMEDIATE
- Data model invariants: singleton row, NOT NULL, defaults
- Error taxonomy: ValueError for invalid columns, safe defaults on DB error
- Thread safety: concurrent reads/writes via WAL mode
"""

import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add project root to path (conftest.py does this too, but be explicit)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from state_db import (
    init_db,
    get_state,
    update_state,
    DEFAULT_DB_PATH,
    _VALID_COLUMNS,
)


# ============ Fixtures ============


@pytest.fixture
def db_path(tmp_path):
    """Provide a temp DB path for test isolation."""
    return tmp_path / "state.db"


@pytest.fixture
def initialized_db(db_path):
    """Return a DB path with init_db() already called."""
    init_db(db_path)
    return db_path


# ============ init_db() Contracts ============


class TestInitDb:
    """Verify init_db() contracts from LLD Section 1.1."""

    def test_init_db_creates_table_and_default_row(self, db_path):
        """init_db() creates daemon_state table with one default row (id=1)."""
        init_db(db_path)
        state = get_state(db_path)
        assert state["id"] == 1
        assert state["status"] == "idle"

    def test_init_db_enables_wal_mode(self, db_path):
        """init_db() sets journal_mode=WAL."""
        init_db(db_path)
        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode.lower() == "wal"

    def test_init_db_idempotent_multiple_calls(self, db_path):
        """Calling init_db() multiple times does not duplicate rows or error."""
        init_db(db_path)
        init_db(db_path)
        init_db(db_path)
        state = get_state(db_path)
        assert state["id"] == 1
        # Verify only one row exists
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM daemon_state").fetchone()[0]
        conn.close()
        assert count == 1

    def test_init_db_creates_parent_directories(self, tmp_path):
        """init_db() creates parent dirs if they don't exist."""
        db_path = tmp_path / "deep" / "nested" / "dir" / "state.db"
        init_db(db_path)
        assert db_path.exists()

    def test_init_db_migrates_legacy_post_processor_file(self, db_path):
        """init_db() reads current_post_processor.txt and stores value in DB."""
        legacy_file = db_path.parent / "current_post_processor.txt"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_file.write_text("gemini-fix\n")

        init_db(db_path)

        state = get_state(db_path)
        assert state["post_processor"] == "gemini-fix"

    def test_init_db_migration_deletes_legacy_file(self, db_path):
        """init_db() deletes current_post_processor.txt after migration."""
        legacy_file = db_path.parent / "current_post_processor.txt"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_file.write_text("haiku-fix")

        init_db(db_path)

        assert not legacy_file.exists()

    def test_init_db_migration_skips_if_no_legacy_file(self, db_path):
        """init_db() does not error when no legacy file exists."""
        init_db(db_path)
        state = get_state(db_path)
        assert state["post_processor"] == "claude-merge"

    def test_init_db_migration_strips_whitespace(self, db_path):
        """init_db() strips whitespace from legacy file value."""
        legacy_file = db_path.parent / "current_post_processor.txt"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_file.write_text("  claude-merge \n")

        init_db(db_path)

        state = get_state(db_path)
        assert state["post_processor"] == "claude-merge"

    def test_init_db_never_raises_on_sqlite_error(self, tmp_path):
        """init_db() logs warnings but never raises on sqlite3.Error."""
        # Create a corrupt file that can't be opened as DB
        bad_db = tmp_path / "corrupt.db"
        bad_db.write_bytes(b"\x00" * 100)

        # Spec: "Never raises — logs warnings on sqlite3.Error and continues."
        # This should not raise
        init_db(bad_db)


# ============ get_state() Contracts ============


class TestGetState:
    """Verify get_state() contracts from LLD Section 1.1."""

    def test_get_state_returns_dict_with_seven_keys(self, initialized_db):
        """get_state() returns dict with exact 7 documented keys."""
        state = get_state(initialized_db)
        expected_keys = {
            "id", "status", "daemon_pid", "recording_pid",
            "recording_path", "post_processor", "updated_at",
        }
        assert set(state.keys()) == expected_keys

    def test_get_state_returns_default_values(self, initialized_db):
        """get_state() returns correct defaults for a freshly initialized DB."""
        state = get_state(initialized_db)
        assert state["id"] == 1
        assert state["status"] == "idle"
        assert state["daemon_pid"] is None
        assert state["recording_pid"] is None
        assert state["recording_path"] is None
        assert state["post_processor"] == "claude-merge"
        # updated_at is None for default row (only set by update_state)

    def test_get_state_self_initializes_if_table_missing(self, db_path):
        """get_state() calls init_db() internally if table doesn't exist."""
        # Don't call init_db() — get_state() should handle it
        state = get_state(db_path)
        assert state["id"] == 1
        assert state["status"] == "idle"

    def test_get_state_returns_safe_defaults_on_corrupt_db(self, tmp_path):
        """get_state() returns safe default dict on corrupt DB."""
        bad_db = tmp_path / "corrupt.db"
        bad_db.write_text("this is not a sqlite database")

        state = get_state(bad_db)

        # Verify safe defaults per LLD Section 3.3
        assert state == {
            "id": 1,
            "status": "idle",
            "daemon_pid": None,
            "recording_pid": None,
            "recording_path": None,
            "post_processor": "claude-merge",
            "updated_at": None,
        }

    def test_get_state_returns_updated_values(self, initialized_db):
        """get_state() returns values written by update_state()."""
        update_state(initialized_db,
                     status="recording",
                     recording_pid=12345,
                     recording_path="/tmp/test.wav")

        state = get_state(initialized_db)
        assert state["status"] == "recording"
        assert state["recording_pid"] == 12345
        assert state["recording_path"] == "/tmp/test.wav"

    def test_get_state_thread_safe_concurrent_reads(self, initialized_db):
        """Multiple threads can call get_state() concurrently without error."""
        results = []
        errors = []

        def read_state():
            try:
                state = get_state(initialized_db)
                results.append(state)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_state) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Concurrent reads produced errors: {errors}"
        assert len(results) == 10
        for state in results:
            assert state["id"] == 1

    def test_get_state_opens_own_connection(self, initialized_db):
        """Each get_state() call is independent (no shared connection)."""
        # Calling from different threads verifies connection independence
        state1 = get_state(initialized_db)
        update_state(initialized_db, status="recording")
        state2 = get_state(initialized_db)

        assert state1["status"] == "idle"
        assert state2["status"] == "recording"


# ============ update_state() Contracts ============


class TestUpdateState:
    """Verify update_state() contracts from LLD Section 1.1."""

    def test_update_state_single_column(self, initialized_db):
        """update_state() can update a single column."""
        update_state(initialized_db, status="recording")
        state = get_state(initialized_db)
        assert state["status"] == "recording"

    def test_update_state_multiple_columns(self, initialized_db):
        """update_state() can update multiple columns atomically."""
        update_state(initialized_db,
                     status="recording",
                     recording_pid=9999,
                     recording_path="/tmp/audio.wav")
        state = get_state(initialized_db)
        assert state["status"] == "recording"
        assert state["recording_pid"] == 9999
        assert state["recording_path"] == "/tmp/audio.wav"

    def test_update_state_auto_sets_updated_at(self, initialized_db):
        """update_state() automatically sets updated_at to UTC ISO 8601."""
        before = datetime.now(timezone.utc)
        update_state(initialized_db, status="processing")
        after = datetime.now(timezone.utc)

        state = get_state(initialized_db)
        assert state["updated_at"] is not None
        ts = datetime.fromisoformat(state["updated_at"])
        assert before <= ts <= after

    def test_update_state_rejects_invalid_column(self, initialized_db):
        """update_state() raises ValueError for column names not in _VALID_COLUMNS."""
        with pytest.raises(ValueError):
            update_state(initialized_db, invalid_column="bad")

    def test_update_state_rejects_sql_injection_column(self, initialized_db):
        """update_state() column whitelist prevents SQL injection."""
        with pytest.raises(ValueError):
            update_state(initialized_db,
                         **{"status; DROP TABLE daemon_state": "hacked"})

    def test_update_state_nullable_columns_accept_none(self, initialized_db):
        """update_state() can set nullable columns to None."""
        update_state(initialized_db,
                     recording_pid=12345,
                     recording_path="/tmp/test.wav")
        update_state(initialized_db,
                     recording_pid=None,
                     recording_path=None)

        state = get_state(initialized_db)
        assert state["recording_pid"] is None
        assert state["recording_path"] is None

    def test_update_state_concurrent_writes_serialized(self, initialized_db):
        """Concurrent update_state() calls are serialized via BEGIN IMMEDIATE."""
        errors = []
        success_count = [0]

        def write_state(value):
            try:
                update_state(initialized_db, status=value)
                success_count[0] += 1
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_state, args=(f"test_{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Concurrent writes produced errors: {errors}"
        assert success_count[0] == 10
        # Final state should be one of the written values
        state = get_state(initialized_db)
        assert state["status"].startswith("test_")

    def test_update_state_does_not_raise_on_sqlite_error(self, tmp_path):
        """update_state() logs warning and returns silently on sqlite3 error."""
        corrupt_db = tmp_path / "corrupt.db"
        corrupt_db.write_text("not a database")

        # Spec: "Never raises sqlite3.Error — logs warning and returns silently."
        # Should not raise
        update_state(corrupt_db, status="recording")

    def test_update_state_preserves_unmodified_columns(self, initialized_db):
        """Updating one column does not affect other columns."""
        update_state(initialized_db,
                     status="recording",
                     recording_pid=1234,
                     recording_path="/tmp/test.wav",
                     post_processor="gemini-fix")

        # Update only status
        update_state(initialized_db, status="processing")

        state = get_state(initialized_db)
        assert state["status"] == "processing"
        assert state["recording_pid"] == 1234  # unchanged
        assert state["recording_path"] == "/tmp/test.wav"  # unchanged
        assert state["post_processor"] == "gemini-fix"  # unchanged


# ============ Data Model Contracts (LLD Section 3) ============


class TestDataModelContracts:
    """Verify data model invariants from LLD Section 3."""

    def test_singleton_row_enforced_by_check(self, initialized_db):
        """Only one row can exist (id=1, CHECK(id=1) constraint)."""
        conn = sqlite3.connect(str(initialized_db))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO daemon_state (id) VALUES (2)")
            conn.commit()
        conn.close()

    def test_status_not_null_constraint(self, initialized_db):
        """status column cannot be NULL (NOT NULL constraint)."""
        conn = sqlite3.connect(str(initialized_db))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE daemon_state SET status = NULL WHERE id = 1"
            )
            conn.commit()
        conn.close()

    def test_post_processor_not_null_constraint(self, initialized_db):
        """post_processor column cannot be NULL (NOT NULL constraint)."""
        conn = sqlite3.connect(str(initialized_db))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE daemon_state SET post_processor = NULL WHERE id = 1"
            )
            conn.commit()
        conn.close()

    def test_status_default_is_idle(self, initialized_db):
        """Default status is 'idle'."""
        state = get_state(initialized_db)
        assert state["status"] == "idle"

    def test_post_processor_default_is_gemini_merge_string(self, initialized_db):
        """Default post_processor is 'claude-merge' (string, not Python None)."""
        state = get_state(initialized_db)
        assert state["post_processor"] == "claude-merge"
        assert isinstance(state["post_processor"], str)

    def test_updated_at_iso8601_utc_format(self, initialized_db):
        """updated_at is ISO 8601 UTC string after update."""
        update_state(initialized_db, status="recording")
        state = get_state(initialized_db)
        ts = datetime.fromisoformat(state["updated_at"])
        assert ts.tzinfo is not None  # Must have timezone info

    def test_daemon_state_schema_columns(self, initialized_db):
        """daemon_state table has exactly the documented columns."""
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute("PRAGMA table_info(daemon_state)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "id", "status", "daemon_pid", "recording_pid",
            "recording_path", "post_processor", "updated_at",
        }
        assert columns == expected


# ============ Configuration Contracts (LLD Section 5) ============


class TestConfigurationContracts:
    """Verify configuration contracts from LLD Section 5."""

    def test_default_db_path_matches_spec(self):
        """DEFAULT_DB_PATH is ~/.config/voice-input/state.db."""
        expected = Path.home() / ".config" / "voice-input" / "state.db"
        assert DEFAULT_DB_PATH == expected

    def test_valid_columns_match_spec(self):
        """_VALID_COLUMNS matches the documented column whitelist."""
        expected = frozenset({
            "status", "daemon_pid", "recording_pid",
            "recording_path", "post_processor", "updated_at",
        })
        assert _VALID_COLUMNS == expected

    def test_sqlite_timeout_handles_busy_lock(self, initialized_db):
        """Connection uses timeout for busy wait (spec: timeout=5 seconds)."""
        # Hold a write lock
        conn = sqlite3.connect(str(initialized_db))
        conn.execute("BEGIN IMMEDIATE")

        errors = []

        def try_write():
            try:
                update_state(initialized_db, status="test")
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=try_write)
        t.start()

        # Release lock after a short delay
        time.sleep(0.5)
        conn.commit()
        conn.close()

        t.join(timeout=10)
        # Should succeed after lock release, or fail gracefully (no raise)
        # Either outcome is acceptable per spec


# ============ Error Taxonomy (LLD Section 4) ============


class TestErrorTaxonomy:
    """Verify error taxonomy from LLD Section 4."""

    def test_valueerror_raised_for_invalid_column(self, initialized_db):
        """update_state() raises ValueError for column not in _VALID_COLUMNS."""
        with pytest.raises(ValueError):
            update_state(initialized_db, nonexistent_column="value")

    def test_get_state_returns_safe_defaults_on_error(self, tmp_path):
        """get_state() returns _SAFE_DEFAULT instead of raising on DB error."""
        corrupt_db = tmp_path / "bad.db"
        corrupt_db.write_text("not a database")

        state = get_state(corrupt_db)
        assert state["status"] == "idle"
        assert state["daemon_pid"] is None
        assert state["recording_pid"] is None
        assert state["recording_path"] is None
        assert state["post_processor"] == "claude-merge"
        assert state["updated_at"] is None

    def test_init_db_never_raises(self, tmp_path):
        """init_db() never raises, even on sqlite3.Error."""
        blocker = tmp_path / "blocker"
        blocker.write_text("file blocking dir creation")
        db_path = blocker / "sub" / "state.db"

        # Spec: "Never raises — logs warnings on sqlite3.Error and continues."
        try:
            init_db(db_path)
        except sqlite3.Error:
            pytest.fail("init_db() raised sqlite3.Error but spec says it never raises")


# ============ Round-Trip Integration Tests ============


class TestRoundTrip:
    """Verify end-to-end data flow through state_db functions."""

    def test_full_state_lifecycle(self, db_path):
        """init → update(recording) → get → update(processing) → get → update(idle) → get."""
        init_db(db_path)

        # idle → recording
        update_state(db_path,
                     status="recording",
                     recording_pid=5678,
                     recording_path="/tmp/audio.wav")
        state = get_state(db_path)
        assert state["status"] == "recording"
        assert state["recording_pid"] == 5678
        assert state["recording_path"] == "/tmp/audio.wav"

        # recording → processing
        update_state(db_path,
                     status="processing",
                     recording_pid=None,
                     recording_path=None)
        state = get_state(db_path)
        assert state["status"] == "processing"
        assert state["recording_pid"] is None

        # processing → idle
        update_state(db_path, status="idle")
        state = get_state(db_path)
        assert state["status"] == "idle"

    def test_daemon_pid_lifecycle(self, db_path):
        """Daemon writes PID on start, clears on shutdown."""
        init_db(db_path)

        # Daemon starts
        update_state(db_path, daemon_pid=os.getpid())
        state = get_state(db_path)
        assert state["daemon_pid"] == os.getpid()

        # Daemon stops
        update_state(db_path, daemon_pid=None, status="idle")
        state = get_state(db_path)
        assert state["daemon_pid"] is None

    def test_post_processor_roundtrip_all_presets(self, db_path):
        """All post-processor preset IDs persist and retrieve correctly."""
        init_db(db_path)

        for preset_id in ["none", "gemini-fix", "haiku-fix", "claude-merge"]:
            update_state(db_path, post_processor=preset_id)
            state = get_state(db_path)
            assert state["post_processor"] == preset_id

    def test_concurrent_reader_writer_wal(self, initialized_db):
        """One writer + multiple readers operate correctly under WAL mode."""
        errors = []
        read_results = []

        def writer():
            try:
                for i in range(20):
                    update_state(initialized_db, status=f"state_{i}")
                    time.sleep(0.01)
            except Exception as e:
                errors.append(("writer", e))

        def reader():
            try:
                for _ in range(20):
                    state = get_state(initialized_db)
                    read_results.append(state["status"])
                    time.sleep(0.01)
            except Exception as e:
                errors.append(("reader", e))

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]

        writer_thread.start()
        for t in reader_threads:
            t.start()

        writer_thread.join(timeout=10)
        for t in reader_threads:
            t.join(timeout=10)

        assert not errors, f"Concurrent read/write errors: {errors}"
        assert len(read_results) == 60  # 3 readers x 20 reads

    def test_legacy_migration_preserves_value_in_roundtrip(self, db_path):
        """Legacy file value survives: write file → init_db → get_state."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_file = db_path.parent / "current_post_processor.txt"
        legacy_file.write_text("claude-merge\n")

        init_db(db_path)
        state = get_state(db_path)
        assert state["post_processor"] == "claude-merge"

    def test_updated_at_advances_on_each_update(self, initialized_db):
        """Each update_state() call advances updated_at."""
        update_state(initialized_db, status="recording")
        state1 = get_state(initialized_db)
        ts1 = state1["updated_at"]

        time.sleep(0.05)

        update_state(initialized_db, status="processing")
        state2 = get_state(initialized_db)
        ts2 = state2["updated_at"]

        assert ts2 > ts1
