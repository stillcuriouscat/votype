"""Clean-room integration tests for state_db.py migration (Module E).

Derived from LOW_LEVEL_DESIGN.md §1 Module E. Verifies:
  - _DEPRECATED_PP maps the documented sources to 'claude-merge'.
  - get_state() returns 'claude-merge' and writes it back when the DB
    holds 'gemini-merge' or 'firered-punc'.
  - get_state() does NOT migrate 'gemini-fix' or 'haiku-fix'.
  - The default for fresh DBs is 'claude-merge'.
  - Migration emits an INFO log line.
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_state_db():
    try:
        import state_db  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"state_db import failed: {e}")
    return state_db


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestStateDbConstants:
    """LLD §1 Module E: three constants must reference 'claude-merge'."""

    def test_deprecated_pp_contains_gemini_merge_to_claude_merge(self):
        sdb = _import_state_db()
        assert hasattr(sdb, "_DEPRECATED_PP"), "state_db._DEPRECATED_PP must exist"
        assert sdb._DEPRECATED_PP.get("gemini-merge") == "claude-merge"

    def test_deprecated_pp_contains_firered_punc_to_claude_merge(self):
        """LLD: firered-punc target bumped from 'gemini-merge' to 'claude-merge'."""
        sdb = _import_state_db()
        assert sdb._DEPRECATED_PP.get("firered-punc") == "claude-merge"

    def test_deprecated_pp_does_not_contain_gemini_fix(self):
        """PRD US-005 AC: gemini-fix is NOT in _DEPRECATED_PP."""
        sdb = _import_state_db()
        assert "gemini-fix" not in sdb._DEPRECATED_PP

    def test_deprecated_pp_does_not_contain_haiku_fix(self):
        """PRD US-005 AC: haiku-fix is NOT in _DEPRECATED_PP."""
        sdb = _import_state_db()
        assert "haiku-fix" not in sdb._DEPRECATED_PP

    def test_safe_default_post_processor_is_claude_merge(self):
        sdb = _import_state_db()
        assert hasattr(sdb, "_SAFE_DEFAULT")
        assert sdb._SAFE_DEFAULT["post_processor"] == "claude-merge"

    def test_create_table_sql_default_is_claude_merge(self):
        """LLD: CREATE TABLE DEFAULT clause must be 'claude-merge'."""
        sdb = _import_state_db()
        assert hasattr(sdb, "_CREATE_TABLE_SQL")
        sql = sdb._CREATE_TABLE_SQL
        # Look for: DEFAULT 'claude-merge'
        assert "'claude-merge'" in sql, (
            "CREATE TABLE SQL must contain DEFAULT 'claude-merge'"
        )
        assert "'gemini-merge'" not in sql, (
            "CREATE TABLE SQL must no longer reference 'gemini-merge' as default"
        )


# ---------------------------------------------------------------------------
# get_state() migration behavior
# ---------------------------------------------------------------------------


def _write_pp(db_path: Path, pp_value: str) -> None:
    """Initialise a daemon_state DB with a specific post_processor value.

    We use a portable, minimal table definition so this fixture does not
    depend on the production schema's evolving DEFAULT clause.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daemon_state (
                id INTEGER PRIMARY KEY CHECK(id=1),
                status TEXT NOT NULL DEFAULT 'idle',
                daemon_pid INTEGER,
                recording_pid INTEGER,
                recording_path TEXT,
                post_processor TEXT NOT NULL DEFAULT 'claude-merge',
                updated_at TEXT
            )
            """
        )
        conn.execute("DELETE FROM daemon_state")
        conn.execute(
            "INSERT INTO daemon_state (id, status, post_processor, updated_at) "
            "VALUES (1, 'idle', ?, '2026-01-01T00:00:00')",
            (pp_value,),
        )
        conn.commit()
    finally:
        conn.close()


def _read_pp(db_path: Path) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT post_processor FROM daemon_state WHERE id=1"
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


class TestGetStateMigration:
    """LLD §1 Module E migration semantics table."""

    def test_gemini_merge_migrates_to_claude_merge(self, tmp_path):
        sdb = _import_state_db()
        db = tmp_path / "state.db"
        _write_pp(db, "gemini-merge")

        state = sdb.get_state(db)
        assert state["post_processor"] == "claude-merge"

        # And the migration must have been written back synchronously.
        assert _read_pp(db) == "claude-merge"

    def test_firered_punc_migrates_to_claude_merge(self, tmp_path):
        sdb = _import_state_db()
        db = tmp_path / "state.db"
        _write_pp(db, "firered-punc")

        state = sdb.get_state(db)
        assert state["post_processor"] == "claude-merge"
        assert _read_pp(db) == "claude-merge"

    def test_gemini_fix_is_not_migrated(self, tmp_path):
        sdb = _import_state_db()
        db = tmp_path / "state.db"
        _write_pp(db, "gemini-fix")

        state = sdb.get_state(db)
        assert state["post_processor"] == "gemini-fix"
        assert _read_pp(db) == "gemini-fix"  # unchanged in DB

    def test_haiku_fix_is_not_migrated(self, tmp_path):
        sdb = _import_state_db()
        db = tmp_path / "state.db"
        _write_pp(db, "haiku-fix")

        state = sdb.get_state(db)
        assert state["post_processor"] == "haiku-fix"
        assert _read_pp(db) == "haiku-fix"

    def test_claude_merge_is_passthrough(self, tmp_path):
        sdb = _import_state_db()
        db = tmp_path / "state.db"
        _write_pp(db, "claude-merge")

        state = sdb.get_state(db)
        assert state["post_processor"] == "claude-merge"
        assert _read_pp(db) == "claude-merge"

    def test_claude_fix_is_passthrough(self, tmp_path):
        sdb = _import_state_db()
        db = tmp_path / "state.db"
        _write_pp(db, "claude-fix")

        state = sdb.get_state(db)
        assert state["post_processor"] == "claude-fix"
        assert _read_pp(db) == "claude-fix"

    def test_none_is_passthrough(self, tmp_path):
        sdb = _import_state_db()
        db = tmp_path / "state.db"
        _write_pp(db, "none")

        state = sdb.get_state(db)
        assert state["post_processor"] == "none"
        assert _read_pp(db) == "none"

    def test_arbitrary_value_is_passthrough(self, tmp_path):
        """LLD: 'any other → as-stored, unchanged'."""
        sdb = _import_state_db()
        db = tmp_path / "state.db"
        _write_pp(db, "some-custom-preset")

        state = sdb.get_state(db)
        assert state["post_processor"] == "some-custom-preset"
        assert _read_pp(db) == "some-custom-preset"

    def test_migration_logs_at_info_level(self, tmp_path, caplog):
        """LLD: migration must emit a log line at INFO level."""
        sdb = _import_state_db()
        db = tmp_path / "state.db"
        _write_pp(db, "gemini-merge")

        with caplog.at_level(logging.INFO):
            sdb.get_state(db)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert info_records, (
            "Migration must emit an INFO-level log record when writing back"
        )
        # Suggested format mentions both old + new values.
        joined = "\n".join(r.getMessage() for r in info_records)
        assert "gemini-merge" in joined
        assert "claude-merge" in joined

    def test_migration_does_not_log_error_or_warning(self, tmp_path, caplog):
        """LLD: not WARNING / not ERROR — informational only."""
        sdb = _import_state_db()
        db = tmp_path / "state.db"
        _write_pp(db, "gemini-merge")

        with caplog.at_level(logging.DEBUG):
            sdb.get_state(db)

        # Filter to migration-relevant records (contain 'gemini-merge')
        relevant = [
            r for r in caplog.records
            if "gemini-merge" in r.getMessage() and "claude-merge" in r.getMessage()
        ]
        for r in relevant:
            assert r.levelno < logging.WARNING, (
                f"Migration log must be < WARNING; got {r.levelname}: {r.getMessage()}"
            )


# ---------------------------------------------------------------------------
# Fresh DB default
# ---------------------------------------------------------------------------


class TestFreshDbDefault:
    """LLD: After init_db + get_state on a fresh path, post_processor is 'claude-merge'."""

    def test_init_db_then_get_state_defaults_to_claude_merge(self, tmp_path):
        sdb = _import_state_db()
        db = tmp_path / "fresh_state.db"
        # Some implementations require init_db; others lazily create.
        if hasattr(sdb, "init_db"):
            sdb.init_db(db)
        state = sdb.get_state(db)
        assert state["post_processor"] == "claude-merge"

    def test_missing_db_returns_safe_default_with_claude_merge(self, tmp_path):
        """LLD: get_state never raises; returns _SAFE_DEFAULT on DB error.

        We use a non-existent path; depending on implementation this may
        either create the DB (with default 'claude-merge') or return the
        safe default. Either way the value is 'claude-merge'.
        """
        sdb = _import_state_db()
        db = tmp_path / "does_not_exist" / "state.db"
        state = sdb.get_state(db)
        assert state["post_processor"] == "claude-merge"
