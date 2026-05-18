"""Clean-room unit tests for US-005: state_db gemini-merge → claude-merge migration.

Derived from FUNCTION_SPEC.md Module E behavior + migration tables.
"""

import logging
import sqlite3
from pathlib import Path

import pytest

from state_db import (
    _DEPRECATED_PP,
    _SAFE_DEFAULT,
    _CREATE_TABLE_SQL,
    init_db,
    get_state,
    update_state,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_deprecated_pp_includes_firered_punc(self):
        assert _DEPRECATED_PP["firered-punc"] == "claude-merge"

    def test_deprecated_pp_includes_gemini_merge(self):
        assert _DEPRECATED_PP["gemini-merge"] == "claude-merge"

    def test_gemini_fix_not_deprecated(self):
        assert "gemini-fix" not in _DEPRECATED_PP

    def test_haiku_fix_not_deprecated(self):
        assert "haiku-fix" not in _DEPRECATED_PP

    def test_safe_default_post_processor(self):
        assert _SAFE_DEFAULT["post_processor"] == "claude-merge"

    def test_create_table_default(self):
        assert "DEFAULT 'claude-merge'" in _CREATE_TABLE_SQL
        assert "DEFAULT 'gemini-merge'" not in _CREATE_TABLE_SQL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """Provide an isolated DB path for each test."""
    return tmp_path / "state.db"


def _set_pp(db_path: Path, value: str) -> None:
    """Bypass migration: write raw value to DB."""
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("UPDATE daemon_state SET post_processor=? WHERE id=1", (value,))
        conn.commit()
    finally:
        conn.close()


def _read_pp(db_path: Path) -> str:
    """Read post_processor directly bypassing migration."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT post_processor FROM daemon_state WHERE id=1").fetchone()
        return row["post_processor"] if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_fresh_init_default_is_claude_merge(self, db_path):
        init_db(db_path)
        assert _read_pp(db_path) == "claude-merge"

    def test_get_state_after_fresh_init_is_claude_merge(self, db_path):
        init_db(db_path)
        state = get_state(db_path)
        assert state["post_processor"] == "claude-merge"

    def test_idempotent_re_init(self, db_path):
        init_db(db_path)
        init_db(db_path)
        assert _read_pp(db_path) == "claude-merge"


# ---------------------------------------------------------------------------
# Migration matrix
# ---------------------------------------------------------------------------

class TestMigrationMatrix:
    @pytest.mark.parametrize("initial,expected", [
        ("firered-punc", "claude-merge"),
        ("gemini-merge", "claude-merge"),
    ])
    def test_migrated_values(self, db_path, initial, expected):
        _set_pp(db_path, initial)
        state = get_state(db_path)
        assert state["post_processor"] == expected
        assert _read_pp(db_path) == expected  # DB rewritten

    @pytest.mark.parametrize("initial", [
        "gemini-fix",
        "haiku-fix",
        "none",
        "claude-merge",
        "claude-fix",
        "unknown-x",
    ])
    def test_unchanged_values(self, db_path, initial):
        _set_pp(db_path, initial)
        state = get_state(db_path)
        assert state["post_processor"] == initial
        assert _read_pp(db_path) == initial


# ---------------------------------------------------------------------------
# Migration log
# ---------------------------------------------------------------------------

class TestMigrationLog:
    def test_migration_emits_info_log(self, db_path, caplog):
        _set_pp(db_path, "gemini-merge")
        with caplog.at_level(logging.INFO, logger="state_db"):
            get_state(db_path)
        msgs = [r.message for r in caplog.records if "[STATE-DB] migrated" in r.message]
        assert msgs, "expected migration log entry"
        assert "'gemini-merge'" in msgs[0]
        assert "'claude-merge'" in msgs[0]

    def test_no_log_for_unchanged_value(self, db_path, caplog):
        _set_pp(db_path, "haiku-fix")
        with caplog.at_level(logging.INFO, logger="state_db"):
            get_state(db_path)
        msgs = [r.message for r in caplog.records if "[STATE-DB] migrated" in r.message]
        assert not msgs

    def test_idempotent_call_logs_once(self, db_path, caplog):
        _set_pp(db_path, "gemini-merge")
        with caplog.at_level(logging.INFO, logger="state_db"):
            get_state(db_path)
            caplog.clear()
            get_state(db_path)
        msgs = [r.message for r in caplog.records if "[STATE-DB] migrated" in r.message]
        # Second call: value is now claude-merge → no migration log
        assert not msgs


# ---------------------------------------------------------------------------
# Safe default
# ---------------------------------------------------------------------------

class TestSafeDefault:
    def test_corrupt_db_returns_safe_default(self, tmp_path):
        bad = tmp_path / "not_a_db.db"
        bad.write_bytes(b"not really a sqlite db")
        state = get_state(bad)
        assert state["post_processor"] == "claude-merge"
