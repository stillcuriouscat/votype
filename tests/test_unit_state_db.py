"""Clean-room unit tests for state_db module.

Tests derived from FUNCTION_SPEC.md behavior tables.
Tests the contract: init_db(), get_state(), update_state().
"""

import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_row(db_path: Path) -> dict:
    """Direct DB read bypassing state_db — for verification only."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM daemon_state WHERE id=1").fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def _journal_mode(db_path: Path) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()


# ===========================================================================
# init_db() tests — FUNCTION_SPEC.md Behavior Table rows 1-8
# ===========================================================================

class TestInitDb:
    """Tests for state_db.init_db()."""

    def test_creates_table_and_default_row(self, tmp_path: Path) -> None:
        """BT#1: First init, no legacy file → creates DB, table, default row."""
        from state_db import init_db

        db_path = tmp_path / "state.db"
        init_db(db_path)

        assert db_path.exists()
        row = _read_row(db_path)
        assert row["id"] == 1
        assert row["status"] == "idle"
        assert row["daemon_pid"] is None
        assert row["recording_pid"] is None
        assert row["recording_path"] is None
        assert row["post_processor"] == "none"
        assert row["updated_at"] is None

    def test_enables_wal_mode(self, tmp_path: Path) -> None:
        """BT#1: WAL mode enabled after init."""
        from state_db import init_db

        db_path = tmp_path / "state.db"
        init_db(db_path)

        assert _journal_mode(db_path) == "wal"

    def test_migrates_legacy_post_processor_file(self, tmp_path: Path) -> None:
        """BT#2: Legacy file with 'gemini-merge' → migrated to DB."""
        from state_db import init_db

        legacy_file = tmp_path / "current_post_processor.txt"
        legacy_file.write_text("gemini-merge")
        db_path = tmp_path / "state.db"

        init_db(db_path)

        row = _read_row(db_path)
        assert row["post_processor"] == "gemini-merge"
        assert not legacy_file.exists()

    def test_migration_strips_whitespace(self, tmp_path: Path) -> None:
        """BT#3: Legacy file with whitespace → stripped value migrated."""
        from state_db import init_db

        legacy_file = tmp_path / "current_post_processor.txt"
        legacy_file.write_text("gemini-fix\n  ")
        db_path = tmp_path / "state.db"

        init_db(db_path)

        row = _read_row(db_path)
        assert row["post_processor"] == "gemini-fix"
        assert not legacy_file.exists()

    def test_idempotent_multiple_calls(self, tmp_path: Path) -> None:
        """BT#4: Called twice → second call is no-op, no error."""
        from state_db import init_db

        db_path = tmp_path / "state.db"
        init_db(db_path)
        init_db(db_path)

        row = _read_row(db_path)
        assert row["id"] == 1
        assert row["status"] == "idle"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """BT#5: Deep nested path → creates parent directories."""
        from state_db import init_db

        db_path = tmp_path / "deep" / "nested" / "state.db"
        init_db(db_path)

        assert db_path.exists()
        row = _read_row(db_path)
        assert row["id"] == 1

    def test_migration_skips_empty_legacy_file(self, tmp_path: Path) -> None:
        """BT#6: Empty legacy file → skip migration, delete file."""
        from state_db import init_db

        legacy_file = tmp_path / "current_post_processor.txt"
        legacy_file.write_text("")
        db_path = tmp_path / "state.db"

        init_db(db_path)

        row = _read_row(db_path)
        # Empty value is skipped; post_processor keeps the default "none"
        assert row["post_processor"] == "none"
        assert not legacy_file.exists()

    def test_handles_readonly_filesystem_gracefully(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """BT#7: Read-only path → no exception, logs warning."""
        from state_db import init_db

        # Use a path that cannot be created
        db_path = Path("/proc/nonexistent/state.db")

        with caplog.at_level(logging.WARNING):
            init_db(db_path)
            # Must not raise

        # Should have logged a warning
        assert len(caplog.records) > 0

    def test_handles_corrupt_db_gracefully(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """BT#8: Corrupt DB file → no exception, logs warning."""
        from state_db import init_db

        db_path = tmp_path / "state.db"
        db_path.write_bytes(b"this is not a sqlite database")

        with caplog.at_level(logging.WARNING):
            init_db(db_path)
            # Must not raise

        assert len(caplog.records) > 0

    def test_migration_deletes_legacy_file(self, tmp_path: Path) -> None:
        """Explicit check: legacy file is deleted after migration."""
        from state_db import init_db

        legacy_file = tmp_path / "current_post_processor.txt"
        legacy_file.write_text("haiku-fix")
        db_path = tmp_path / "state.db"

        init_db(db_path)

        assert not legacy_file.exists()

    def test_migration_skips_if_no_legacy_file(self, tmp_path: Path) -> None:
        """No legacy file → post_processor stays default 'none'."""
        from state_db import init_db

        db_path = tmp_path / "state.db"
        init_db(db_path)

        row = _read_row(db_path)
        assert row["post_processor"] == "none"


# ===========================================================================
# get_state() tests — FUNCTION_SPEC.md Behavior Table rows 1-7
# ===========================================================================

class TestGetState:
    """Tests for state_db.get_state()."""

    def test_returns_default_row(self, tmp_path: Path) -> None:
        """BT#1: DB initialized with defaults → returns all default values."""
        from state_db import get_state, init_db

        db_path = tmp_path / "state.db"
        init_db(db_path)
        state = get_state(db_path)

        assert state["id"] == 1
        assert state["status"] == "idle"
        assert state["daemon_pid"] is None
        assert state["recording_pid"] is None
        assert state["recording_path"] is None
        assert state["post_processor"] == "none"
        assert state["updated_at"] is None

    def test_returns_updated_values(self, tmp_path: Path) -> None:
        """BT#2: After update_state → get_state returns updated values."""
        from state_db import get_state, init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)
        update_state(db_path, status="recording", recording_pid=12345)

        state = get_state(db_path)
        assert state["status"] == "recording"
        assert state["recording_pid"] == 12345
        assert state["updated_at"] is not None

    def test_self_initializes_if_table_missing(self, tmp_path: Path) -> None:
        """BT#3: Fresh empty DB (no prior init_db) → self-initializes."""
        from state_db import get_state

        db_path = tmp_path / "state.db"
        # Create an empty DB file without calling init_db
        sqlite3.connect(str(db_path)).close()

        state = get_state(db_path)
        assert state["id"] == 1
        assert state["status"] == "idle"

    def test_returns_safe_default_when_db_nonexistent_parent(
        self, tmp_path: Path
    ) -> None:
        """BT#4: DB path with nonexistent parent → returns safe defaults."""
        from state_db import get_state, _SAFE_DEFAULT

        db_path = tmp_path / "nonexistent" / "state.db"
        state = get_state(db_path)

        # Should return safe defaults (init_db may succeed creating dirs,
        # or fail gracefully)
        assert state["id"] == 1
        assert state["status"] == "idle"
        assert state["post_processor"] == "none"

    def test_thread_safe_concurrent_reads(self, tmp_path: Path) -> None:
        """BT#5: 10 concurrent threads reading → all get consistent dict."""
        from state_db import get_state, init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)
        update_state(db_path, status="recording", recording_pid=999)

        results = []
        errors = []

        def reader():
            try:
                s = get_state(db_path)
                results.append(s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(results) == 10
        for s in results:
            assert s["status"] == "recording"
            assert s["recording_pid"] == 999

    def test_returns_safe_default_on_corrupt_db(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """BT#6: Corrupt DB → returns safe default dict, logs warning."""
        from state_db import get_state

        db_path = tmp_path / "state.db"
        db_path.write_bytes(b"corrupt data garbage bytes")

        with caplog.at_level(logging.WARNING):
            state = get_state(db_path)

        assert state["id"] == 1
        assert state["status"] == "idle"
        assert state["daemon_pid"] is None
        assert state["recording_pid"] is None
        assert state["recording_path"] is None
        assert state["post_processor"] == "none"
        assert state["updated_at"] is None

    def test_returns_safe_default_on_locked_db(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """BT#7: DB locked beyond timeout → returns safe default."""
        from state_db import get_state, init_db

        db_path = tmp_path / "state.db"
        init_db(db_path)

        # Hold an exclusive lock on the DB
        blocker = sqlite3.connect(str(db_path))
        blocker.execute("BEGIN EXCLUSIVE")

        with caplog.at_level(logging.WARNING):
            # get_state should not hang forever; timeout=5 in connect
            # But we mock the timeout to be short for test speed
            with patch("state_db.sqlite3.connect") as mock_connect:
                mock_connect.side_effect = sqlite3.OperationalError(
                    "database is locked"
                )
                state = get_state(db_path)

        blocker.rollback()
        blocker.close()

        assert state["status"] == "idle"
        assert state["id"] == 1

    def test_returns_dict_with_all_expected_keys(self, tmp_path: Path) -> None:
        """Returned dict has exactly the expected keys."""
        from state_db import get_state, init_db

        db_path = tmp_path / "state.db"
        init_db(db_path)
        state = get_state(db_path)

        expected_keys = {
            "id", "status", "daemon_pid", "recording_pid",
            "recording_path", "post_processor", "updated_at",
        }
        assert set(state.keys()) == expected_keys


# ===========================================================================
# update_state() tests — FUNCTION_SPEC.md Behavior Table rows 1-11
# ===========================================================================

class TestUpdateState:
    """Tests for state_db.update_state()."""

    def test_updates_single_column(self, tmp_path: Path) -> None:
        """BT#1: Update single column → reflected in DB."""
        from state_db import get_state, init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)
        update_state(db_path, status="recording")

        state = get_state(db_path)
        assert state["status"] == "recording"

    def test_updates_multiple_columns(self, tmp_path: Path) -> None:
        """BT#2: Update multiple columns → all reflected in DB."""
        from state_db import get_state, init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)
        update_state(
            db_path,
            status="recording",
            recording_pid=12345,
            recording_path="/tmp/rec.wav",
        )

        state = get_state(db_path)
        assert state["status"] == "recording"
        assert state["recording_pid"] == 12345
        assert state["recording_path"] == "/tmp/rec.wav"

    def test_set_column_to_none(self, tmp_path: Path) -> None:
        """BT#3: Set columns to None → NULL in DB."""
        from state_db import get_state, init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)
        update_state(db_path, recording_pid=12345, recording_path="/tmp/r.wav")
        update_state(db_path, recording_pid=None, recording_path=None)

        state = get_state(db_path)
        assert state["recording_pid"] is None
        assert state["recording_path"] is None

    def test_update_post_processor(self, tmp_path: Path) -> None:
        """BT#4: Update post_processor → reflected in DB."""
        from state_db import get_state, init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)
        update_state(db_path, post_processor="gemini-merge")

        state = get_state(db_path)
        assert state["post_processor"] == "gemini-merge"

    def test_no_kwargs_is_noop(self, tmp_path: Path) -> None:
        """BT#5: No kwargs → no DB write, no updated_at change."""
        from state_db import get_state, init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)

        state_before = get_state(db_path)
        update_state(db_path)  # no kwargs
        state_after = get_state(db_path)

        assert state_before["updated_at"] == state_after["updated_at"]

    def test_explicit_updated_at_not_overwritten(self, tmp_path: Path) -> None:
        """BT#6: Explicit updated_at → user value used, not auto-set."""
        from state_db import get_state, init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)
        custom_ts = "2026-01-01T00:00:00+00:00"
        update_state(db_path, updated_at=custom_ts)

        state = get_state(db_path)
        assert state["updated_at"] == custom_ts

    def test_concurrent_writes_serialized(self, tmp_path: Path) -> None:
        """BT#7: Two threads writing concurrently → both succeed, no corruption."""
        from state_db import get_state, init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)
        errors = []

        def writer_a():
            try:
                for _ in range(20):
                    update_state(db_path, status="recording")
            except Exception as e:
                errors.append(e)

        def writer_b():
            try:
                for _ in range(20):
                    update_state(db_path, status="processing")
            except Exception as e:
                errors.append(e)

        t_a = threading.Thread(target=writer_a)
        t_b = threading.Thread(target=writer_b)
        t_a.start()
        t_b.start()
        t_a.join(timeout=30)
        t_b.join(timeout=30)

        assert len(errors) == 0
        # Final state should be one of the two values
        state = get_state(db_path)
        assert state["status"] in ("recording", "processing")

    def test_rejects_invalid_column_name(self, tmp_path: Path) -> None:
        """BT#8: Invalid column name → ValueError with exact message."""
        from state_db import init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)

        with pytest.raises(ValueError, match="Invalid column: invalid_col"):
            update_state(db_path, invalid_col="value")

    def test_rejects_sql_injection_attempt(self, tmp_path: Path) -> None:
        """BT#9: SQL injection in column name → ValueError, no DB write."""
        from state_db import init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)

        with pytest.raises(ValueError, match="Invalid column"):
            update_state(db_path, **{"status; DROP TABLE": "x"})

        # Table must still exist
        row = _read_row(db_path)
        assert row["id"] == 1

    def test_handles_locked_db_gracefully(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """BT#10: DB locked beyond timeout → no exception, logs warning."""
        from state_db import init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)

        with patch("state_db.sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.OperationalError(
                "database is locked"
            )
            with caplog.at_level(logging.WARNING):
                update_state(db_path, status="recording")
                # Must not raise

    def test_handles_corrupt_db_gracefully(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """BT#11: Corrupt DB → no exception, logs warning."""
        from state_db import update_state

        db_path = tmp_path / "state.db"
        db_path.write_bytes(b"corrupt garbage")

        with caplog.at_level(logging.WARNING):
            update_state(db_path, status="recording")
            # Must not raise

    def test_auto_sets_updated_at(self, tmp_path: Path) -> None:
        """updated_at is automatically set to UTC ISO 8601 on update."""
        from state_db import get_state, init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)

        before = datetime.now(timezone.utc)
        update_state(db_path, status="recording")
        after = datetime.now(timezone.utc)

        state = get_state(db_path)
        assert state["updated_at"] is not None

        ts = datetime.fromisoformat(state["updated_at"])
        assert before <= ts <= after

    def test_update_preserves_other_columns(self, tmp_path: Path) -> None:
        """Updating one column does not change others."""
        from state_db import get_state, init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)
        update_state(
            db_path,
            status="recording",
            recording_pid=111,
            post_processor="gemini-fix",
        )
        update_state(db_path, status="processing")

        state = get_state(db_path)
        assert state["status"] == "processing"
        assert state["recording_pid"] == 111  # unchanged
        assert state["post_processor"] == "gemini-fix"  # unchanged

    def test_valid_columns_error_message_lists_columns(
        self, tmp_path: Path
    ) -> None:
        """ValueError message includes sorted list of valid columns."""
        from state_db import init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)

        with pytest.raises(
            ValueError,
            match="Valid columns: daemon_pid, post_processor, recording_path, "
                  "recording_pid, status, updated_at",
        ):
            update_state(db_path, bad_column="x")

    def test_begin_immediate_transaction(self, tmp_path: Path) -> None:
        """update_state uses BEGIN IMMEDIATE (verified via tracing)."""
        from state_db import init_db, update_state

        db_path = tmp_path / "state.db"
        init_db(db_path)

        traced_sql = []

        original_connect = sqlite3.connect

        def tracing_connect(*args, **kwargs):
            conn = original_connect(*args, **kwargs)
            conn.set_trace_callback(lambda stmt: traced_sql.append(stmt))
            return conn

        with patch("state_db.sqlite3.connect", side_effect=tracing_connect):
            update_state(db_path, status="recording")

        # Check that BEGIN IMMEDIATE was used
        begin_stmts = [s for s in traced_sql if "BEGIN" in s.upper()]
        assert any("IMMEDIATE" in s.upper() for s in begin_stmts), (
            f"Expected BEGIN IMMEDIATE in traced SQL, got: {begin_stmts}"
        )
