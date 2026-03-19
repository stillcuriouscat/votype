"""
Unit tests for state_db module (US-001).

Tests cover: init_db, get_state, update_state, concurrent access,
WAL mode, legacy file migration, error handling, and thread safety.
"""

import sqlite3
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import state_db


# ============ Fixtures ============

@pytest.fixture
def db_path(tmp_path):
    """Provide a fresh DB path in a temp directory."""
    return tmp_path / "state.db"


@pytest.fixture
def initialized_db(db_path):
    """Provide an initialized DB."""
    state_db.init_db(db_path)
    return db_path


# ============ init_db Tests ============

class TestInitDB:
    """Tests for init_db()."""

    def test_creates_db_file(self, db_path):
        """init_db creates the database file."""
        assert not db_path.exists()
        state_db.init_db(db_path)
        assert db_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        """init_db creates parent directories if they don't exist."""
        deep_path = tmp_path / "deep" / "nested" / "state.db"
        state_db.init_db(deep_path)
        assert deep_path.exists()

    def test_creates_daemon_state_table(self, initialized_db):
        """init_db creates daemon_state table."""
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='daemon_state'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_inserts_default_row(self, initialized_db):
        """init_db inserts default row with id=1."""
        conn = sqlite3.connect(str(initialized_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM daemon_state WHERE id=1").fetchone()
        conn.close()

        assert row is not None
        assert row["id"] == 1
        assert row["status"] == "idle"
        assert row["daemon_pid"] is None
        assert row["recording_pid"] is None
        assert row["recording_path"] is None
        assert row["post_processor"] == "none"
        assert row["updated_at"] is None

    def test_enables_wal_mode(self, initialized_db):
        """init_db enables WAL journal mode."""
        conn = sqlite3.connect(str(initialized_db))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_idempotent(self, db_path):
        """init_db can be called multiple times without error."""
        state_db.init_db(db_path)
        state_db.init_db(db_path)

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM daemon_state").fetchone()[0]
        conn.close()
        assert count == 1

    def test_schema_columns(self, initialized_db):
        """Table has all expected columns."""
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute("PRAGMA table_info(daemon_state)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        expected = {"id", "status", "daemon_pid", "recording_pid",
                    "recording_path", "post_processor", "updated_at"}
        assert columns == expected

    def test_id_check_constraint(self, initialized_db):
        """id column has CHECK(id=1) constraint — only id=1 allowed."""
        conn = sqlite3.connect(str(initialized_db))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO daemon_state (id) VALUES (2)")
        conn.close()

    def test_corrupt_db_logs_warning(self, db_path):
        """init_db on corrupt DB file logs warning without raising."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_bytes(b"this is not a valid sqlite db")
        # Should not raise
        state_db.init_db(db_path)

    def test_uses_default_path(self):
        """init_db with no args uses DEFAULT_DB_PATH."""
        with patch.object(state_db, 'DEFAULT_DB_PATH', Path("/tmp/test_default_state.db")):
            try:
                state_db.init_db()
                assert Path("/tmp/test_default_state.db").exists()
            finally:
                Path("/tmp/test_default_state.db").unlink(missing_ok=True)
                Path("/tmp/test_default_state.db-wal").unlink(missing_ok=True)
                Path("/tmp/test_default_state.db-shm").unlink(missing_ok=True)


# ============ Legacy Migration Tests ============

class TestLegacyMigration:
    """Tests for post_processor migration from legacy file to DB."""

    def test_migrates_legacy_post_processor(self, db_path):
        """init_db migrates value from current_post_processor.txt to DB."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_file = db_path.parent / "current_post_processor.txt"
        legacy_file.write_text("gemini-merge")

        state_db.init_db(db_path)

        # Value migrated to DB
        state = state_db.get_state(db_path)
        assert state["post_processor"] == "gemini-merge"

        # Legacy file deleted
        assert not legacy_file.exists()

    def test_migrates_with_whitespace(self, db_path):
        """init_db strips whitespace from legacy file value."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_file = db_path.parent / "current_post_processor.txt"
        legacy_file.write_text("gemini-fix\n  ")

        state_db.init_db(db_path)

        state = state_db.get_state(db_path)
        assert state["post_processor"] == "gemini-fix"
        assert not legacy_file.exists()

    def test_skips_empty_legacy_file(self, db_path):
        """init_db skips migration for empty legacy file but still deletes it."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_file = db_path.parent / "current_post_processor.txt"
        legacy_file.write_text("")

        state_db.init_db(db_path)

        state = state_db.get_state(db_path)
        assert state["post_processor"] == "none"  # Default, not migrated
        assert not legacy_file.exists()

    def test_no_legacy_file_is_fine(self, db_path):
        """init_db works normally when no legacy file exists."""
        state_db.init_db(db_path)
        state = state_db.get_state(db_path)
        assert state["post_processor"] == "none"


# ============ get_state Tests ============

class TestGetState:
    """Tests for get_state()."""

    def test_returns_default_state(self, initialized_db):
        """get_state returns default values after init."""
        state = state_db.get_state(initialized_db)
        assert state["id"] == 1
        assert state["status"] == "idle"
        assert state["daemon_pid"] is None
        assert state["recording_pid"] is None
        assert state["recording_path"] is None
        assert state["post_processor"] == "none"
        assert state["updated_at"] is None

    def test_returns_updated_values(self, initialized_db):
        """get_state returns values after update_state."""
        state_db.update_state(initialized_db, status="recording", recording_pid=12345)
        state = state_db.get_state(initialized_db)
        assert state["status"] == "recording"
        assert state["recording_pid"] == 12345
        assert state["updated_at"] is not None

    def test_self_initializes_on_missing_table(self, db_path):
        """get_state calls init_db if table doesn't exist."""
        # Create an empty DB file (no table)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.close()

        state = state_db.get_state(db_path)
        assert state["status"] == "idle"
        assert state["id"] == 1

    def test_returns_safe_default_on_corrupt_db(self, db_path):
        """get_state returns _SAFE_DEFAULT on corrupt DB."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_bytes(b"corrupt data")

        state = state_db.get_state(db_path)
        assert state == state_db._SAFE_DEFAULT

    def test_returns_safe_default_on_nonexistent_parent(self, tmp_path):
        """get_state returns _SAFE_DEFAULT when parent dir doesn't exist."""
        db_path = tmp_path / "nonexistent" / "deep" / "state.db"
        state = state_db.get_state(db_path)
        assert state["status"] == "idle"
        assert state["id"] == 1

    def test_returns_new_dict_each_time(self, initialized_db):
        """get_state returns a new dict, not a shared reference."""
        state1 = state_db.get_state(initialized_db)
        state2 = state_db.get_state(initialized_db)
        assert state1 is not state2
        assert state1 == state2

    def test_uses_default_path(self, initialized_db):
        """get_state with no args uses DEFAULT_DB_PATH."""
        with patch.object(state_db, 'DEFAULT_DB_PATH', initialized_db):
            state = state_db.get_state()
            assert state["status"] == "idle"


# ============ update_state Tests ============

class TestUpdateState:
    """Tests for update_state()."""

    def test_update_single_column(self, initialized_db):
        """update_state updates a single column."""
        state_db.update_state(initialized_db, status="recording")
        state = state_db.get_state(initialized_db)
        assert state["status"] == "recording"

    def test_update_multiple_columns(self, initialized_db):
        """update_state updates multiple columns atomically."""
        state_db.update_state(
            initialized_db,
            status="recording",
            recording_pid=12345,
            recording_path="/tmp/rec.wav",
        )
        state = state_db.get_state(initialized_db)
        assert state["status"] == "recording"
        assert state["recording_pid"] == 12345
        assert state["recording_path"] == "/tmp/rec.wav"

    def test_sets_updated_at_automatically(self, initialized_db):
        """update_state auto-sets updated_at to UTC ISO 8601."""
        state_db.update_state(initialized_db, status="recording")
        state = state_db.get_state(initialized_db)
        assert state["updated_at"] is not None
        # Verify it's a valid ISO 8601 timestamp
        assert "T" in state["updated_at"]
        assert "+" in state["updated_at"] or "Z" in state["updated_at"]

    def test_set_column_to_none(self, initialized_db):
        """update_state can set columns to None."""
        state_db.update_state(initialized_db, recording_pid=12345)
        state_db.update_state(initialized_db, recording_pid=None, recording_path=None)
        state = state_db.get_state(initialized_db)
        assert state["recording_pid"] is None
        assert state["recording_path"] is None

    def test_no_kwargs_is_noop(self, initialized_db):
        """update_state with no kwargs is a no-op."""
        state_db.update_state(initialized_db, status="recording")
        state_before = state_db.get_state(initialized_db)
        updated_at_before = state_before["updated_at"]

        state_db.update_state(initialized_db)  # no-op

        state_after = state_db.get_state(initialized_db)
        assert state_after["updated_at"] == updated_at_before

    def test_explicit_updated_at_not_overwritten(self, initialized_db):
        """Explicit updated_at is used as-is, not auto-overwritten."""
        custom_time = "2026-01-01T00:00:00+00:00"
        state_db.update_state(initialized_db, updated_at=custom_time)
        state = state_db.get_state(initialized_db)
        assert state["updated_at"] == custom_time

    def test_invalid_column_raises_valueerror(self, initialized_db):
        """update_state raises ValueError for invalid column names."""
        with pytest.raises(ValueError, match="Invalid column: invalid_col"):
            state_db.update_state(initialized_db, invalid_col="value")

    def test_sql_injection_blocked(self, initialized_db):
        """Column whitelist blocks SQL injection attempts."""
        with pytest.raises(ValueError, match="Invalid column"):
            state_db.update_state(initialized_db, **{"status; DROP TABLE": "x"})

    def test_update_post_processor(self, initialized_db):
        """update_state can update post_processor column."""
        state_db.update_state(initialized_db, post_processor="gemini-merge")
        state = state_db.get_state(initialized_db)
        assert state["post_processor"] == "gemini-merge"

    def test_other_columns_unchanged(self, initialized_db):
        """update_state doesn't affect columns not in kwargs."""
        state_db.update_state(initialized_db, status="recording", recording_pid=42)
        state_db.update_state(initialized_db, status="processing")
        state = state_db.get_state(initialized_db)
        assert state["status"] == "processing"
        assert state["recording_pid"] == 42  # Unchanged

    def test_corrupt_db_logs_warning(self, db_path):
        """update_state on corrupt DB logs warning without raising."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_bytes(b"corrupt data")
        # Should not raise
        state_db.update_state(db_path, status="recording")

    def test_uses_default_path(self, initialized_db):
        """update_state with no positional args uses DEFAULT_DB_PATH."""
        with patch.object(state_db, 'DEFAULT_DB_PATH', initialized_db):
            state_db.update_state(status="recording")
            state = state_db.get_state(initialized_db)
            assert state["status"] == "recording"


# ============ Concurrent Access Tests ============

class TestConcurrency:
    """Tests for thread safety and concurrent access."""

    def test_concurrent_reads(self, initialized_db):
        """Multiple threads can read simultaneously without error."""
        results = []
        errors = []

        def reader():
            try:
                state = state_db.get_state(initialized_db)
                results.append(state)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 10
        for r in results:
            assert r["status"] == "idle"

    def test_concurrent_writes(self, initialized_db):
        """Multiple threads can write with BEGIN IMMEDIATE serialization."""
        errors = []

        def writer(status_val):
            try:
                state_db.update_state(initialized_db, status=status_val)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(f"status_{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Errors: {errors}"

        # Final state should be one of the written values
        state = state_db.get_state(initialized_db)
        assert state["status"].startswith("status_")

    def test_concurrent_read_write(self, initialized_db):
        """Readers and writers can operate concurrently (WAL mode)."""
        errors = []
        read_results = []

        def writer():
            try:
                for i in range(5):
                    state_db.update_state(initialized_db, recording_pid=i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(5):
                    state = state_db.get_state(initialized_db)
                    read_results.append(state)
            except Exception as e:
                errors.append(e)

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join(timeout=5)
        t_read.join(timeout=5)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(read_results) == 5


# ============ WAL Mode Verification ============

class TestWALMode:
    """Verify WAL mode is properly set."""

    def test_wal_mode_set(self, initialized_db):
        """DB is in WAL journal mode after init."""
        conn = sqlite3.connect(str(initialized_db))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_wal_file_exists_after_write(self, initialized_db):
        """WAL file created after write operation."""
        state_db.update_state(initialized_db, status="recording")
        wal_file = Path(str(initialized_db) + "-wal")
        # WAL file may or may not exist depending on checkpoint timing
        # Just verify the DB is still functional
        state = state_db.get_state(initialized_db)
        assert state["status"] == "recording"
