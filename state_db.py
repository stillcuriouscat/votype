"""
SQLite-based state management for voice-input daemon.

Single source of truth replacing fragmented file-based state (PID files,
processing.flag, IPC status commands). Uses WAL mode for concurrent
reader/writer access. Each function opens/closes its own connection
for thread safety.

DB path: ~/.config/voice-input/state.db
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_DB_PATH: Path = Path.home() / ".config" / "voice-input" / "state.db"

_VALID_COLUMNS: frozenset[str] = frozenset({
    "status", "daemon_pid", "recording_pid",
    "recording_path", "post_processor", "updated_at",
})

# Matches SQL DEFAULT; if this PP fails to load, load_post_processor() falls back to regex-only
_SAFE_DEFAULT: dict[str, object] = {
    "id": 1,
    "status": "idle",
    "daemon_pid": None,
    "recording_pid": None,
    "recording_path": None,
    "post_processor": "gemini-merge",
    "updated_at": None,
}

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS daemon_state (
    id INTEGER PRIMARY KEY CHECK(id=1),
    status TEXT NOT NULL DEFAULT 'idle',
    daemon_pid INTEGER,
    recording_pid INTEGER,
    recording_path TEXT,
    post_processor TEXT NOT NULL DEFAULT 'gemini-merge',
    updated_at TEXT
)"""


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize the SQLite state database.

    Creates schema, inserts default row, enables WAL mode, and migrates
    legacy post-processor file if it exists. Safe to call multiple times.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(
                "INSERT OR IGNORE INTO daemon_state (id) VALUES (1)"
            )
            conn.commit()
        finally:
            conn.close()

        # Migrate legacy post-processor file
        legacy_file = db_path.parent / "current_post_processor.txt"
        if legacy_file.exists():
            value = legacy_file.read_text().strip()
            if value:
                update_state(db_path, post_processor=value)
            legacy_file.unlink()

    except (sqlite3.Error, OSError) as e:
        logger.warning("init_db failed: %s", e)


def get_state(db_path: Optional[Path] = None) -> dict[str, object]:
    """Read all columns from the singleton daemon_state row.

    Self-initializing: if the table doesn't exist, calls init_db() and
    retries. Returns _SAFE_DEFAULT on any error for graceful degradation.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            try:
                row = conn.execute(
                    "SELECT * FROM daemon_state WHERE id=1"
                ).fetchone()
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    conn.close()
                    init_db(db_path)
                    conn = sqlite3.connect(str(db_path), timeout=5)
                    conn.row_factory = sqlite3.Row
                    row = conn.execute(
                        "SELECT * FROM daemon_state WHERE id=1"
                    ).fetchone()
                else:
                    raise

            if row is None:
                return dict(_SAFE_DEFAULT)
            return dict(row)
        finally:
            conn.close()

    except Exception as e:
        logger.warning("get_state failed: %s", e)
        return dict(_SAFE_DEFAULT)


def update_state(db_path: Optional[Path] = None, **kwargs: object) -> None:
    """Atomically update columns in the singleton daemon_state row.

    Uses BEGIN IMMEDIATE for serialized writes. Automatically sets
    updated_at to current UTC timestamp unless explicitly provided.
    Raises ValueError for invalid column names.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    # Validate column names (SQL injection prevention)
    invalid = set(kwargs.keys()) - _VALID_COLUMNS
    if invalid:
        valid_sorted = ", ".join(sorted(_VALID_COLUMNS))
        raise ValueError(
            f"Invalid column: {', '.join(sorted(invalid))}. "
            f"Valid columns: {valid_sorted}"
        )

    if not kwargs:
        return

    # Auto-set updated_at unless caller explicitly provides it
    if "updated_at" not in kwargs:
        kwargs["updated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        set_clauses = ", ".join(f"{col} = ?" for col in kwargs)
        values = list(kwargs.values())

        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                f"UPDATE daemon_state SET {set_clauses} WHERE id=1",
                values,
            )
            conn.commit()
        finally:
            conn.close()

    except (sqlite3.Error, OSError) as e:
        logger.error("update_state FAILED (data may be stale): %s — kwargs=%s", e, kwargs)
