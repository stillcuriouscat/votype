# Low-Level Design: Votype SQLite State Management
Generated: 2026-03-19T23:30:00+08:00
PRD: prd.json — branch `ralph/sqlite-state`, 4 stories, replace file-based state with SQLite
Architecture: HIGH_LEVEL_DESIGN.md

## 1. Module Interface Catalog

### 1.1 `state_db.py` (NEW)

#### Constants

```python
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger: logging.Logger = logging.getLogger(__name__)

# Own path constant — NEVER import from voice_input.py (circular import guard)
DEFAULT_DB_PATH: Path = Path.home() / ".config" / "voice-input" / "state.db"

# Column whitelist for update_state() — prevents SQL injection
_VALID_COLUMNS: frozenset[str] = frozenset({
    "status", "daemon_pid", "recording_pid",
    "recording_path", "post_processor", "updated_at",
})

_CREATE_TABLE_SQL: str = """
    CREATE TABLE IF NOT EXISTS daemon_state (
        id INTEGER PRIMARY KEY CHECK(id=1),
        status TEXT NOT NULL DEFAULT 'idle',
        daemon_pid INTEGER,
        recording_pid INTEGER,
        recording_path TEXT,
        post_processor TEXT NOT NULL DEFAULT 'none',
        updated_at TEXT
    )
"""

_DEFAULT_ROW_SQL: str = """
    INSERT OR IGNORE INTO daemon_state (id) VALUES (1)
"""
```

#### Public Interface

```python
def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize the SQLite state database.

    - Creates daemon_state table if not exists
    - Inserts default row (id=1) if empty
    - Enables WAL journal mode
    - Migrates legacy current_post_processor.txt if present:
      reads value, calls update_state(post_processor=value),
      then deletes the legacy file

    Args:
        db_path: Path to the SQLite database file.
                 Defaults to DEFAULT_DB_PATH.

    Side effects:
        - Creates db_path parent directories if needed
        - Creates/migrates the database file
        - Deletes ~/.config/voice-input/current_post_processor.txt
          if it exists (one-time migration)

    Raises:
        Never raises — logs warnings on sqlite3.Error and continues.
        This function must be safe to call from any context.
    """
    ...


def get_state(db_path: Optional[Path] = None) -> dict[str, object]:
    """Read all columns from the daemon_state row.

    Self-initializing: calls init_db() internally if the table
    does not exist (defensive against access before explicit init).

    Args:
        db_path: Path to the SQLite database file.
                 Defaults to DEFAULT_DB_PATH.

    Returns:
        Dict with keys: "id", "status", "daemon_pid", "recording_pid",
        "recording_path", "post_processor", "updated_at".
        Values are str, int, or None matching SQLite column types.

        On error, returns a safe default dict:
        {
            "id": 1,
            "status": "idle",
            "daemon_pid": None,
            "recording_pid": None,
            "recording_path": None,
            "post_processor": "none",
            "updated_at": None,
        }

    Thread safety:
        Opens and closes its own connection per call.
        Safe to call from any thread without external locking.
    """
    ...


def update_state(db_path: Optional[Path] = None, **kwargs: object) -> None:
    """Atomically update one or more columns in the daemon_state row.

    Uses BEGIN IMMEDIATE transaction to serialize concurrent writes.
    Automatically sets updated_at to current UTC ISO 8601 timestamp.

    Args:
        db_path: Path to the SQLite database file.
                 Defaults to DEFAULT_DB_PATH.
        **kwargs: Column-value pairs to update. Keys must be in
                  _VALID_COLUMNS. Values must be str, int, or None.

    Raises:
        ValueError: If any key in kwargs is not in _VALID_COLUMNS.
        Never raises sqlite3.Error — logs warning and returns silently.

    Examples:
        update_state(status="recording", recording_pid=12345)
        update_state(status="idle", recording_pid=None, recording_path=None)
        update_state(post_processor="gemini-fix")

    Thread safety:
        Opens and closes its own connection per call.
        BEGIN IMMEDIATE prevents write starvation.
        sqlite3.connect(timeout=5) handles transient busy states.
    """
    ...
```

#### Internal Details (for implementer, not public contract)

- Connection strategy: `sqlite3.connect(str(db_path), timeout=5)` per call, closed in `finally` block
- WAL mode: `PRAGMA journal_mode=WAL` executed once in `init_db()`
- Legacy migration in `init_db()`:
  1. `legacy_file = db_path.parent / "current_post_processor.txt"`
  2. If `legacy_file.exists()`: read `.strip()`, validate against a known preset list is NOT required (state_db has no knowledge of presets), just store raw value
  3. Call `update_state(db_path=db_path, post_processor=value)`
  4. `legacy_file.unlink()`
- `updated_at` format: `datetime.now(timezone.utc).isoformat()` (e.g., `"2026-03-19T15:30:00+00:00"`)

#### Events Emitted / Consumed

- Emits: None (pure data layer)
- Consumes: None

---

### 1.2 `voice_input.py` — CLI Functions (MODIFY)

#### New Constant

```python
from state_db import init_db, get_state, update_state

# SQLite state database path (mirrors state_db.DEFAULT_DB_PATH)
STATE_DB_PATH: Path = Path.home() / ".config" / "voice-input" / "state.db"
```

#### Modified Public Interface

```python
def ensure_config_dir() -> None:
    """Ensure config directory exists, initialize state DB, clean legacy files.

    Changes from current:
    - Calls init_db(STATE_DB_PATH) after mkdir
    - Deletes PID_FILE, PROCESSING_FILE, AUDIO_PATH_FILE if they exist
      (legacy file cleanup — state now lives in DB)

    Note: Legacy file deletion happens HERE (not in state_db.py) because
    state_db.py cannot import voice_input constants (circular import).
    """
    ...


def is_recording() -> bool:
    """Check whether recording is in progress.

    Changes from current:
    - Reads DB: get_state(STATE_DB_PATH)
    - Returns True only if status == "recording" AND recording_pid is alive
      (verified via os.kill(pid, 0))
    - If recording_pid is dead: calls update_state(status="idle",
      recording_pid=None, recording_path=None) and returns False

    Returns:
        True if recording is actively in progress, False otherwise.
    """
    ...


def start_recording() -> None:
    """Start recording with a timestamped filename.

    Changes from current:
    - Writes to DB instead of PID_FILE + AUDIO_PATH_FILE:
      update_state(STATE_DB_PATH,
                   status="recording",
                   recording_pid=proc.pid,
                   recording_path=str(audio_file))
    - On spawn failure: update_state(status="idle",
                                      recording_pid=None,
                                      recording_path=None)
    - REMOVES: PID_FILE.write_text(), AUDIO_PATH_FILE.write_text()
    - REMOVES: send_to_daemon("recording_start")
    - REMOVES: send_to_daemon("set_idle") in error path
    """
    ...


def stop_recording() -> None:
    """Stop recording and transcribe.

    Changes from current:
    - Reads recording_pid AND recording_path from DB via single
      get_state(STATE_DB_PATH) call BEFORE the kill sequence
      (CRITIC-R2-C1: PID_FILE no longer written, must read from DB)
    - Writes status="processing" to DB instead of PROCESSING_FILE
    - After kill: update_state(recording_pid=None, recording_path=None)
    - After transcription complete: update_state(status="idle")
    - REMOVES: PROCESSING_FILE.write_text()
    - REMOVES: PID_FILE.read_text(), AUDIO_PATH_FILE.read_text()
    - REMOVES: PROCESSING_FILE.unlink()
    - REMOVES: send_to_daemon("recording_stop")
    - REMOVES: send_to_daemon("set_idle")

    Error recovery:
    - Audio file not found: update_state(status="idle") and return
    - Daemon not running: update_state(status="idle") and return
    - All error paths must reset DB status to "idle"
    """
    ...


def toggle_recording() -> None:
    """Toggle recording state.

    Changes from current:
    - Checks DB status instead of PROCESSING_FILE.exists():
      state = get_state(STATE_DB_PATH)
      if state["status"] == "processing":
        parse state["updated_at"], compute age
        if age < 120s: notify and return
        else: update_state(status="idle")  # stale cleanup
    - REMOVES: PROCESSING_FILE.exists() check
    - REMOVES: PROCESSING_FILE.stat().st_mtime
    - REMOVES: PROCESSING_FILE.unlink()
    - REMOVES: send_to_daemon("set_idle") in stale cleanup path
    """
    ...


def show_status() -> None:
    """Show current status from DB instead of IPC.

    Changes from current:
    - Reads state from DB: get_state(STATE_DB_PATH)
    - Prints status, daemon_pid, post_processor from DB
    - Falls back to IPC for model info only (get_model command retained)
    - REMOVES: is_recording() call (uses DB status directly)
    - REMOVES: send_to_daemon("get_post_processor") (reads DB)
    """
    ...


def is_daemon_running() -> bool:
    """Check whether the daemon is running.

    Changes from current:
    - First tries DB: reads daemon_pid from get_state(STATE_DB_PATH)
    - If daemon_pid is not None: verify process alive via os.kill(pid, 0)
    - If alive: verify it's our daemon via /proc/{pid}/cmdline check
    - If dead PID: update_state(daemon_pid=None) and clean up files
    - Falls back to flock + DAEMON_PID_FILE for backward compatibility
      during transition (daemon may not have written to DB yet)
    """
    ...
```

#### Removed Functions

```python
# REMOVED — replaced by DB operations in state_db.py
def is_process_running(pid_file: Path) -> bool: ...
    # Note: Only remove if no other callers remain.
    # is_daemon_running() currently calls this via DAEMON_PID_FILE.
    # After migration, is_daemon_running() reads DB directly.
    # Grep codebase for is_process_running() callers before removing.

# REMOVED — replaced by state_db.update_state(post_processor=...)
@staticmethod
def _persist_post_processor_id(preset_id: str) -> None: ...

# REMOVED — replaced by state_db.get_state()["post_processor"]
@staticmethod
def _restore_post_processor_id() -> str: ...
```

#### Removed Constants

```python
# REMOVED from active use (constants kept for reference but no longer written/read)
# POST_PROCESSOR_STATE_FILE — replaced by DB column "post_processor"
# Note: PID_FILE, PROCESSING_FILE, AUDIO_PATH_FILE constants kept but
# only used in ensure_config_dir() for legacy cleanup (delete if exist)
```

#### Events Emitted / Consumed

- REMOVES emitting IPC: `recording_start`, `recording_stop`, `set_idle`
- RETAINS emitting IPC: `transcribe`, `ping`, `get_model`, `set_post_processor`, `stop`

---

### 1.3 `voice_input.py` — ASRDaemon Class (MODIFY)

#### Modified Public Interface

```python
class ASRDaemon:

    def __init__(self, model_id: Optional[str] = None) -> None:
        """Initialize the daemon.

        Changes from current:
        - Adds: self._current_db_status: str = "idle"
        - Replaces: self._restore_post_processor_id()
          with: get_state(STATE_DB_PATH)["post_processor"]
        """
        ...

    def _sync_status_from_db(self) -> None:
        """Poll DB for status changes and update GTK icon if changed.

        NEW method. Called every 1 second in socket_server() timeout loop.

        Reads status from DB via get_state(STATE_DB_PATH).
        Compares with self._current_db_status.
        If different: calls self.set_status(new_status) and updates
        self._current_db_status.

        Never raises — wraps all DB operations in try/except and logs.
        """
        ...

    def socket_server(self) -> None:
        """Socket server thread.

        Changes from current:
        - Calls self._sync_status_from_db() in the socket.timeout
          exception handler (every ~1 second when no clients connect)
        """
        ...

    def run(self) -> None:
        """Run the daemon.

        Changes from current:
        - After flock acquired: update_state(STATE_DB_PATH,
                                             daemon_pid=os.getpid())
        - Reads post_processor from DB instead of _restore_post_processor_id()
        - In finally cleanup: update_state(STATE_DB_PATH,
                                           daemon_pid=None,
                                           status="idle")
        """
        ...

    def load_post_processor(self, preset_id: Optional[str] = None) -> None:
        """Load a post-processor model.

        Changes from current:
        - Replaces: self._persist_post_processor_id(preset_id)
          with: update_state(STATE_DB_PATH, post_processor=preset_id)
        """
        ...

    def handle_client(self, client: socket.socket) -> None:
        """Handle client request.

        Changes from current:
        - REMOVES: status_commands dict
        - REMOVES: "recording_start", "recording_stop", "set_idle" handlers
        - REMOVES: the `elif command in status_commands:` branch
        - Unknown command now includes these removed commands if sent
        """
        ...
```

#### Events Emitted / Consumed

- Consumes: DB status changes via polling in `_sync_status_from_db()`
- REMOVES consuming IPC: `recording_start`, `recording_stop`, `set_idle`

---

### 1.4 `model_configs.py` (NO CHANGE)

No interface changes. Included for completeness.

### 1.5 `post_processor_configs.py` (NO CHANGE)

No interface changes. Included for completeness.

---

### 1.6 Tests (MODIFY/NEW)

#### 1.6.1 `tests/test_state_db.py` (NEW — ~22 tests)

```python
"""Unit tests for state_db module."""

class TestInitDb:
    def test_creates_table_and_default_row(self, tmp_path: Path) -> None: ...
    def test_enables_wal_mode(self, tmp_path: Path) -> None: ...
    def test_idempotent_multiple_calls(self, tmp_path: Path) -> None: ...
    def test_creates_parent_directories(self, tmp_path: Path) -> None: ...
    def test_migrates_legacy_post_processor_file(self, tmp_path: Path) -> None: ...
    def test_migration_deletes_legacy_file(self, tmp_path: Path) -> None: ...
    def test_migration_skips_if_no_legacy_file(self, tmp_path: Path) -> None: ...
    def test_handles_corrupt_db_gracefully(self, tmp_path: Path) -> None: ...

class TestGetState:
    def test_returns_default_row(self, tmp_path: Path) -> None: ...
    def test_self_initializes_if_table_missing(self, tmp_path: Path) -> None: ...
    def test_returns_safe_defaults_on_error(self, tmp_path: Path) -> None: ...
    def test_returns_updated_values(self, tmp_path: Path) -> None: ...
    def test_thread_safe_concurrent_reads(self, tmp_path: Path) -> None: ...

class TestUpdateState:
    def test_updates_single_column(self, tmp_path: Path) -> None: ...
    def test_updates_multiple_columns(self, tmp_path: Path) -> None: ...
    def test_auto_sets_updated_at(self, tmp_path: Path) -> None: ...
    def test_rejects_invalid_column_name(self, tmp_path: Path) -> None: ...
    def test_begin_immediate_transaction(self, tmp_path: Path) -> None: ...
    def test_concurrent_writes_serialized(self, tmp_path: Path) -> None: ...
    def test_nullable_columns(self, tmp_path: Path) -> None: ...
    def test_handles_busy_timeout(self, tmp_path: Path) -> None: ...
```

#### 1.6.2 `tests/test_post_processor_persistence.py` (REWRITE)

```python
"""Post-processor persistence tests — DB-based."""

class TestPostProcessorPersistence:
    def test_gemini_merge_persists_via_db(self, tmp_path: Path) -> None:
        """update_state(post_processor='gemini-merge') then get_state() returns it."""
        ...

    def test_gemini_fix_persists_via_db(self, tmp_path: Path) -> None: ...
    def test_empty_db_returns_default(self, tmp_path: Path) -> None: ...
    def test_all_presets_persist_roundtrip(self, tmp_path: Path) -> None: ...
    def test_load_failure_does_not_overwrite_db(self, tmp_path: Path) -> None: ...
```

#### 1.6.3 `tests/test_unit.py` (UPDATE)

```python
class TestIsRecording:
    """Updated: now tests DB-based is_recording()."""

    def test_returns_false_when_status_not_recording(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """DB status='idle' → is_recording() returns False."""
        ...

    def test_returns_true_when_recording_and_pid_alive(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """DB status='recording' + live PID → True."""
        ...

    def test_returns_false_when_recording_but_pid_dead(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """DB status='recording' but dead PID → False + DB reset to idle."""
        ...


class TestIsProcessRunning:
    # REMOVED or marked as legacy tests.
    # If is_process_running() is fully removed from voice_input.py,
    # delete this test class entirely.
    # If kept for backward compat, leave tests unchanged.
    ...
```

#### 1.6.4 `tests/test_race.py` (UPDATE)

```python
class TestProcessingFlagGuard:
    """Updated: processing guard now checks DB status, not PROCESSING_FILE."""

    def test_toggle_rejected_during_processing(
        self, isolated_environment, mock_notify
    ) -> None:
        """DB status='processing' + updated_at < 120s → toggle rejected."""
        # Setup: update_state(status="processing", updated_at=<recent>)
        # Assert: start_recording not called
        ...

    def test_toggle_allowed_after_processing_complete(
        self, isolated_environment, mock_notify
    ) -> None:
        """DB status='idle' → toggle proceeds normally."""
        ...

    def test_stale_processing_status_cleaned_up(
        self, isolated_environment, mock_notify
    ) -> None:
        """DB status='processing' + updated_at > 120s → cleanup and proceed."""
        # Setup: update_state(status="processing", updated_at=<old>)
        # Assert: DB status reset to "idle", start_recording called
        ...

    def test_stop_recording_sets_processing_status_in_db(
        self, isolated_environment, mock_notify
    ) -> None:
        """stop_recording() writes status='processing' to DB before kill."""
        ...

    def test_stop_recording_resets_status_on_error(
        self, isolated_environment, mock_notify
    ) -> None:
        """DB status reset to 'idle' even when transcription fails."""
        ...

    def test_processing_status_blocks_concurrent_toggle(
        self, isolated_environment, mock_notify
    ) -> None:
        """Concurrent toggle during processing is blocked by DB status."""
        ...
```

#### 1.6.5 `tests/test_integration.py` (UPDATE — 6 methods)

```python
class TestDaemonSocketCommunication:
    def test_daemon_handles_status_commands(self, ...):
        # REMOVED or REWRITTEN: "recording_start", "recording_stop",
        # "set_idle" are no longer valid IPC commands.
        # Replace with: test that daemon returns error for unknown
        # commands (these are now unknown).
        ...

class TestRecordingFlow:
    def test_start_recording_notifies_daemon(self, ...):
        # REWRITTEN: start_recording no longer sends IPC "recording_start".
        # Instead verify: update_state() called with status="recording".
        ...

class TestToggleRecording:
    def test_toggle_stops_recording_when_already_recording(self, ...):
        # REWRITTEN: stop_recording no longer sends IPC "recording_stop".
        # Instead verify: DB status transitions.
        ...

class TestStateMachine:
    def test_idle_to_recording_transition(self, ...):
        # REWRITTEN: verify DB status='recording' after toggle.
        # Remove: assertion on send_to_daemon("recording_start").
        ...

    def test_recording_to_processing_transition(self, ...):
        # REWRITTEN: verify DB status='processing' after toggle.
        # Remove: assertion on send_to_daemon("recording_stop").
        ...
```

#### 1.6.6 `tests/conftest.py` (UPDATE)

```python
# New imports
from state_db import init_db, DEFAULT_DB_PATH

# Save default for reset
_DEFAULT_STATE_DB_PATH = DEFAULT_DB_PATH

@pytest.fixture(autouse=True)
def reset_voice_input_state(request):
    """Restore voice_input AND state_db module state after each test."""
    yield
    # Existing resets...
    # Add: restore STATE_DB_PATH on voice_input module
    import state_db
    state_db.DEFAULT_DB_PATH = _DEFAULT_STATE_DB_PATH


@pytest.fixture
def isolated_environment(tmp_path, monkeypatch):
    """Fully isolated test environment with SQLite state DB."""
    # ... existing setup ...

    # Add: SQLite state DB in temp dir
    state_db_path = config_dir / "state.db"
    monkeypatch.setattr("voice_input.STATE_DB_PATH", state_db_path)
    monkeypatch.setattr("state_db.DEFAULT_DB_PATH", state_db_path)
    init_db(state_db_path)

    yield {
        # ... existing keys ...
        "state_db_path": state_db_path,
    }


@pytest.fixture
def mock_socket_server(temp_socket):
    """Updated mock daemon socket server."""
    responses = {
        "ping": {"status": "ok"},
        "transcribe": {"text": "mock transcription result"},
        # REMOVED: "recording_start", "recording_stop", "set_idle"
        "stop": {"status": "stopping"},
    }
    # ... rest unchanged ...
```

#### 1.6.7 `tests/test_show_status.py` (NEW — ~3 tests)

```python
"""Tests for DB-based show_status()."""

class TestShowStatus:
    def test_shows_db_status_when_daemon_running(
        self, isolated_environment, capsys
    ) -> None:
        """show_status() reads and displays state from DB."""
        ...

    def test_shows_idle_when_daemon_not_running(
        self, isolated_environment, capsys
    ) -> None:
        """show_status() shows default state when no daemon."""
        ...

    def test_shows_post_processor_from_db(
        self, isolated_environment, capsys
    ) -> None:
        """show_status() displays post_processor from DB."""
        ...
```

#### 1.6.8 `tests/test_e2e_sqlite_state.py` (EXISTING — already written)

No changes needed. These are spec-first E2E tests that validate the final behavior.

---

## 2. Inter-Module Contracts

| Caller | Callee | Method | Input Type | Output Type | Error Cases |
|--------|--------|--------|-----------|-------------|-------------|
| `voice_input.ensure_config_dir()` | `state_db.init_db()` | `init_db(STATE_DB_PATH)` | `Path` | `None` | Never raises; logs on sqlite3.Error |
| `voice_input.start_recording()` | `state_db.update_state()` | `update_state(STATE_DB_PATH, status="recording", recording_pid=int, recording_path=str)` | `Path, **kwargs` | `None` | ValueError if bad column; logs on DB error |
| `voice_input.stop_recording()` | `state_db.get_state()` | `get_state(STATE_DB_PATH)` | `Path` | `dict[str, object]` | Returns safe defaults on error |
| `voice_input.stop_recording()` | `state_db.update_state()` | `update_state(STATE_DB_PATH, status="processing")` then `update_state(STATE_DB_PATH, status="idle")` | `Path, **kwargs` | `None` | Never raises |
| `voice_input.toggle_recording()` | `state_db.get_state()` | `get_state(STATE_DB_PATH)` | `Path` | `dict[str, object]` | Returns safe defaults |
| `voice_input.toggle_recording()` | `state_db.update_state()` | `update_state(STATE_DB_PATH, status="idle")` (stale cleanup) | `Path, **kwargs` | `None` | Never raises |
| `voice_input.is_recording()` | `state_db.get_state()` | `get_state(STATE_DB_PATH)` | `Path` | `dict[str, object]` | Returns safe defaults (status="idle" → False) |
| `voice_input.is_daemon_running()` | `state_db.get_state()` | `get_state(STATE_DB_PATH)` | `Path` | `dict[str, object]` | Falls back to flock+PID file on error |
| `voice_input.show_status()` | `state_db.get_state()` | `get_state(STATE_DB_PATH)` | `Path` | `dict[str, object]` | Displays defaults on error |
| `ASRDaemon.__init__()` | `state_db.get_state()` | `get_state(STATE_DB_PATH)` | `Path` | `dict[str, object]` | Uses DEFAULT_POST_PROCESSOR on error |
| `ASRDaemon.run()` | `state_db.update_state()` | `update_state(STATE_DB_PATH, daemon_pid=os.getpid())` | `Path, **kwargs` | `None` | Logs warning on error |
| `ASRDaemon.run()` (finally) | `state_db.update_state()` | `update_state(STATE_DB_PATH, daemon_pid=None, status="idle")` | `Path, **kwargs` | `None` | Logs warning on error |
| `ASRDaemon._sync_status_from_db()` | `state_db.get_state()` | `get_state(STATE_DB_PATH)` | `Path` | `dict[str, object]` | Silently skips on error |
| `ASRDaemon.load_post_processor()` | `state_db.update_state()` | `update_state(STATE_DB_PATH, post_processor=preset_id)` | `Path, str` | `None` | Logs warning on error |

---

## 3. Data Models

### 3.1 SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS daemon_state (
    id INTEGER PRIMARY KEY CHECK(id=1),    -- singleton row
    status TEXT NOT NULL DEFAULT 'idle',    -- 'idle' | 'recording' | 'processing'
    daemon_pid INTEGER,                     -- daemon process PID, NULL when not running
    recording_pid INTEGER,                  -- recorder process PID, NULL when not recording
    recording_path TEXT,                    -- absolute path to current WAV file, NULL when not recording
    post_processor TEXT NOT NULL DEFAULT 'none',  -- preset ID from POST_PROCESSOR_PRESETS
    updated_at TEXT                         -- ISO 8601 UTC timestamp, auto-set on every update
);
```

### 3.2 State Dict (returned by `get_state()`)

```python
from typing import TypedDict, Optional

class DaemonState(TypedDict):
    id: int                           # Always 1
    status: str                       # "idle" | "recording" | "processing"
    daemon_pid: Optional[int]         # PID or None
    recording_pid: Optional[int]      # PID or None
    recording_path: Optional[str]     # Absolute path or None
    post_processor: str               # Preset ID, default "none"
    updated_at: Optional[str]         # ISO 8601 UTC or None
```

**Invariants:**
- Only one row exists (id=1, enforced by CHECK constraint)
- `status` is always one of: `"idle"`, `"recording"`, `"processing"`
- When `status == "recording"`: `recording_pid` is not None, `recording_path` is not None
- When `status == "idle"`: `recording_pid` is None, `recording_path` is None
- `updated_at` is refreshed on every `update_state()` call
- `post_processor` is never None (NOT NULL DEFAULT 'none')

### 3.3 Safe Default State (returned on error)

```python
_SAFE_DEFAULT: DaemonState = {
    "id": 1,
    "status": "idle",
    "daemon_pid": None,
    "recording_pid": None,
    "recording_path": None,
    "post_processor": "none",
    "updated_at": None,
}
```

---

## 4. Error Taxonomy

| Module | Error Class | When Raised | Recovery |
|--------|-----------|-------------|----------|
| `state_db` | `ValueError` | `update_state()` receives column name not in `_VALID_COLUMNS` | Caller must fix column name; raised immediately, no DB write |
| `state_db` | `sqlite3.OperationalError` (logged, not raised) | DB file locked beyond timeout, corrupt DB, disk full | `get_state()` returns `_SAFE_DEFAULT`; `update_state()` silently skips; `init_db()` logs warning |
| `state_db` | `sqlite3.DatabaseError` (logged, not raised) | DB file corrupt (malformed) | Same as OperationalError — daemon continues with safe defaults |
| `voice_input` | `ProcessLookupError` | `is_recording()` verifies dead PID via `os.kill(pid, 0)` | Auto-cleanup: `update_state(status="idle", recording_pid=None)` |
| `voice_input` | `FileNotFoundError` | Recorder binary (pw-record/arecord) not found in `start_recording()` | `update_state(status="idle")`, notify user |
| `voice_input` | `OSError` | Socket connection fails in `send_to_daemon()` | Returns `{"error": str(e)}`, caller handles |

---

## 5. Configuration Contract

| Key | Type | Default | Required | Used By |
|-----|------|---------|----------|---------|
| `STATE_DB_PATH` | `Path` | `~/.config/voice-input/state.db` | Yes | `voice_input.py` (all CLI functions, ASRDaemon) |
| `DEFAULT_DB_PATH` | `Path` | `~/.config/voice-input/state.db` | Yes | `state_db.py` (fallback when `db_path` arg is None) |
| `_VALID_COLUMNS` | `frozenset[str]` | `{"status", "daemon_pid", "recording_pid", "recording_path", "post_processor", "updated_at"}` | Yes | `state_db.update_state()` (column whitelist) |
| `sqlite3.connect(timeout=...)` | `int` (seconds) | `5` | Yes | All `state_db` functions (busy wait timeout) |
| `_sync_status_from_db()` poll interval | `float` (seconds) | `1.0` | Yes | `ASRDaemon.socket_server()` (socket timeout = poll interval) |
| Processing guard timeout | `int` (seconds) | `120` | Yes | `toggle_recording()` (`updated_at` age threshold) |

---

## 6. Deployment Contract

### 6.1 Symlink Requirement

`state_db.py` must be symlinked to the install directory for the daemon to import it:

```bash
ln -sf ~/code/voice_input/state_db.py ~/.local/share/voice-input/state_db.py
```

This mirrors the existing pattern for `voice_input.py`, `model_configs.py`, etc.

### 6.2 Migration Path

1. `init_db()` creates table + default row on first call
2. `init_db()` migrates `current_post_processor.txt` → DB `post_processor` column
3. `ensure_config_dir()` deletes legacy PID_FILE, PROCESSING_FILE, AUDIO_PATH_FILE
4. IPC status commands removed — old CLI versions will get "Unknown command" errors
   (acceptable: CLI and daemon are always the same version via symlink)

### 6.3 Test Fixture Requirements

- All test fixtures must patch `STATE_DB_PATH` and `state_db.DEFAULT_DB_PATH` to temp dir
- `isolated_environment` fixture must call `init_db(state_db_path)` during setup
- `mock_socket_server` fixture must remove `recording_start`, `recording_stop`, `set_idle` from responses
- `reset_voice_input_state` autouse fixture must restore `state_db.DEFAULT_DB_PATH`
