# High-Level Design: Votype SQLite State Management
Generated: 2026-03-19T22:00:00+08:00
PRD: prd.json — branch `ralph/sqlite-state`, 4 stories total, 0 passing

## 1. System Context

```
                           +---------------------------------------------------+
                           |                   Votype System                    |
                           |                                                   |
  Hotkey (toggle) -------->|   +-------+     +-------+      +---------+        |
                           |   |  CLI  |---->|       |<-----|  Daemon  |       |
                           |   | cmds  |     | state |      | (GTK +  |       |
                           |   +-------+     | .db   |      |  ASR)   |       |
  pw-record/arecord <----->|       |         | (WAL) |      |    |    |       |
                           |       |         +-------+      |    v    |       |
                           |       |              ^         | socket  |       |
                           |       v              |         | server  |       |
                           |   xdotool           poll 1s   +---------+       |
                           |   (type text)                       |            |
                           |                                     v            |
                           +-------------------------------------+------------+
                                                                 |
                                              +------------------+----------+
                                              |  External Services          |
                                              |  - Vertex AI (gemini-fix)   |
                                              |  - SSH Oracle (haiku-fix)   |
                                              +-----------------------------+
```

**Data flow summary:**
- CLI writes state (status, PIDs, paths) **directly to SQLite** — replaces PID_FILE, PROCESSING_FILE, AUDIO_PATH_FILE, IPC status commands
- Daemon **polls SQLite every 1 second** to sync GTK icon — replaces `recording_start/recording_stop/set_idle` IPC messages
- IPC socket **retained** for data-carrying commands only: `transcribe`, `ping`, `get_model`, `set_post_processor`, `stop`

## 2. Module Decomposition

### 2.1 `state_db.py` (NEW)

- **Responsibility**: Single source of truth for daemon/recording/processing state via SQLite
- **Public interface**:
  - `init_db(db_path=None)` — create table, insert default row, enable WAL, migrate legacy `current_post_processor.txt`
  - `get_state(db_path=None) -> dict` — read all columns from `daemon_state` row (self-initializing)
  - `update_state(db_path=None, **kwargs)` — atomic write with `BEGIN IMMEDIATE`, auto-sets `updated_at`
- **Dependencies**: Python stdlib only (`sqlite3`, `pathlib`, `datetime`, `logging`)
- **Owns**: `~/.config/voice-input/state.db` (single table `daemon_state`, single row `id=1`)
- **Key constraint**: Defines its own `DEFAULT_DB_PATH = Path.home() / '.config' / 'voice-input' / 'state.db'` — **never imports from `voice_input.py`** to prevent circular import

### 2.2 `voice_input.py` (MODIFY — CLI functions)

- **Responsibility**: CLI commands (`start`, `stop`, `toggle`, `status`) and daemon lifecycle
- **Changed interface**:
  - `start_recording()` — writes `status='recording'` + `recording_pid` + `recording_path` to DB (replaces PID_FILE + AUDIO_PATH_FILE + IPC `recording_start`)
  - `stop_recording()` — reads `recording_pid` + `recording_path` from DB before kill; writes `status='processing'` then `status='idle'` (replaces PROCESSING_FILE + IPC `recording_stop/set_idle`)
  - `toggle_recording()` — checks `status=='processing'` from DB + `updated_at` age (replaces PROCESSING_FILE.exists() + mtime)
  - `is_recording()` — reads DB `status=='recording'` + `os.kill(pid, 0)` liveness check (replaces PID_FILE-based check)
  - `show_status()` — reads all state from DB (replaces IPC calls for status)
  - `ensure_config_dir()` — calls `init_db()` + deletes legacy state files
- **Dependencies**: Adds `from state_db import init_db, get_state, update_state`
- **Owns**: CLI argument dispatch, recording process lifecycle, xdotool text typing

### 2.3 `voice_input.py` (MODIFY — ASRDaemon class)

- **Responsibility**: Background service with ASR model, GTK tray icon, socket server
- **Changed interface**:
  - `__init__()` — adds `self._current_db_status = 'idle'`; reads `post_processor` from DB (replaces `_restore_post_processor_id()`)
  - `_sync_status_from_db()` (NEW) — polls DB for status, calls `set_status()` on change
  - `socket_server()` — calls `_sync_status_from_db()` in timeout loop (every 1s)
  - `run()` — writes `daemon_pid` to DB on startup, clears on shutdown
  - `load_post_processor()` — writes `post_processor` to DB (replaces `_persist_post_processor_id()`)
  - `is_daemon_running()` — tries DB `daemon_pid` first, falls back to flock + PID file
  - `handle_client()` — removes `status_commands` dict and `recording_start/recording_stop/set_idle` handlers
- **Dependencies**: Same as 2.2 (`state_db` module)
- **Owns**: ASR model lifecycle, GTK icon, IPC socket

### 2.4 `model_configs.py` (NO CHANGE)

- **Responsibility**: ASR model loading (FunASR, faster-whisper, FireRedASR) and inference
- **Public interface**: `ModelLoader.load_model()`, `ModelInference.transcribe()`
- **Dependencies**: `model_presets.py`
- **Owns**: Model instances, transcription logic

### 2.5 `post_processor_configs.py` (NO CHANGE)

- **Responsibility**: Post-processing pipeline (filler removal, vocab, LLM refinement)
- **Public interface**: `PostProcessorLoader.load_post_processor()`, `process_with_vertex_ai()`, `process_with_gemini_merge()`
- **Dependencies**: `post_processor_presets.py`, `vertex_proxy.py`, SSH
- **Owns**: Post-processor model instances, vocab.json read/write

### 2.6 `tests/` (MODIFY multiple files)

- **New**: `tests/test_state_db.py` (~22 tests for state_db module)
- **New**: `tests/test_show_status.py` (~3 tests for DB-based show_status)
- **Rewrite**: `tests/test_post_processor_persistence.py` (DB-based persistence)
- **Update**: `tests/test_unit.py` (TestIsRecording, TestIsProcessRunning → DB-based)
- **Update**: `tests/test_race.py` (TestProcessingFlagGuard → DB status checks)
- **Update**: `tests/conftest.py` (`isolated_environment` fixture + STATE_DB_PATH patch + `init_db()` call)
- **Update**: `tests/test_integration.py` (6 methods referencing removed IPC commands)
- **Existing**: `tests/test_e2e_sqlite_state.py` (spec-first E2E, already written)

## 3. Data Flow

### 3.1 Recording Start (`voice-input toggle` when idle)

```
1. toggle_recording()
   → get_state(STATE_DB_PATH)  →  {status: 'idle', ...}    # Check not processing
   → is_daemon_ready()         →  True (IPC ping)
   → is_recording()            →  get_state() → status != 'recording'

2. start_recording()
   → subprocess.Popen(pw-record ...)  →  recorder PID
   → update_state(status='recording', recording_pid=PID, recording_path=path)
   (NO IPC "recording_start" — daemon polls DB)

3. Daemon socket_server() timeout (within ~1s)
   → _sync_status_from_db()
   → get_state() → {status: 'recording'}
   → 'recording' != self._current_db_status ('idle')
   → set_status('recording')  →  GTK icon turns red
```

### 3.2 Recording Stop (`voice-input toggle` when recording)

```
1. toggle_recording()
   → is_recording() → get_state() → status == 'recording' → True

2. stop_recording()
   → state = get_state()  →  {recording_pid: PID, recording_path: path}
   → update_state(status='processing')
   (NO IPC "recording_stop")
   → os.kill(PID, SIGTERM)  →  stop recorder
   → update_state(recording_pid=None, recording_path=None)
   → send_to_daemon("transcribe", audio_path)  →  ASR + post-process + type_text
   → update_state(status='idle')
   (NO IPC "set_idle")

3. Daemon socket_server() timeout (within ~1s of each status change)
   → _sync_status_from_db()
   → status changes: recording → processing → idle
   → set_status() calls: orange icon → grey icon
```

### 3.3 Processing Guard (toggle during processing)

```
1. toggle_recording()
   → state = get_state() → {status: 'processing', updated_at: '2026-03-19T...'}
   → parse updated_at, compute age
   → if age < 120s: notify("Processing in progress"), return
   → if age >= 120s: update_state(status='idle')  # stale cleanup
```

### 3.4 Daemon Startup/Shutdown

```
Startup:
  run() → flock(LOCK_EX) → update_state(daemon_pid=os.getpid())
  → load_model() → load_post_processor() (reads post_processor from DB)
  → socket_server() loop with _sync_status_from_db() every 1s

Shutdown:
  run() cleanup → update_state(daemon_pid=None, status='idle')
  → SOCKET_PATH.unlink() → DAEMON_PID_FILE.unlink() → lock_fd.close()
```

### 3.5 Error Propagation

| Error | Source | Handler | DB State |
|-------|--------|---------|----------|
| Recorder spawn fails | `start_recording()` | update_state(status='idle', recording_pid=None) | Reset to idle |
| Audio file not found | `stop_recording()` | update_state(status='idle') | Reset to idle |
| DB locked (busy) | `get_state()`/`update_state()` | `sqlite3.connect(timeout=5)` waits; on failure: log warning, return defaults | Unchanged |
| DB corruption | Any DB call | `try/except sqlite3.Error` → log, return safe defaults | Daemon continues |
| Daemon crash | OS | flock auto-released; next `is_daemon_running()` finds dead PID → cleanup | Stale daemon_pid in DB until next read |

## 4. Technology Decisions

| Decision | Choice | Rationale | Alternatives Considered |
|----------|--------|-----------|------------------------|
| State store | SQLite (stdlib `sqlite3`) | Zero new dependencies; ACID; concurrent read/write via WAL; single file | Redis (overkill), JSON file (no atomicity), shared memory (complex) |
| Journal mode | WAL | Allows concurrent reader (daemon) + writer (CLI) without blocking | DELETE mode (blocks readers during writes) |
| Write isolation | `BEGIN IMMEDIATE` | Serializes writes; prevents write starvation under concurrent CLI invocations | DEFERRED (can fail on first write), EXCLUSIVE (blocks readers too) |
| Connection strategy | Open/close per call | Thread-safe without connection pooling; state operations are fast (~1ms) | Connection pool (complex, unnecessary for ~1 call/sec), shared connection (not thread-safe) |
| Daemon icon sync | Poll DB every 1s in socket_server timeout loop | Reuses existing 1s timeout; no new threads; ~1ms per poll (local file read) | inotify on DB file (complex, not reliable for SQLite WAL), dedicated timer thread (unnecessary) |
| IPC status commands | Remove (`recording_start`, `recording_stop`, `set_idle`) | Replaced by DB polling — eliminates lost-message failure mode | Keep both (redundant, more code to maintain) |
| Legacy file migration | `init_db()` migrates `current_post_processor.txt`; `ensure_config_dir()` deletes PID/processing/audio-path files | Clean transition; old files removed after first successful migration | Keep legacy files alongside DB (indefinite complexity), big-bang removal (risky) |
| DB path definition | `state_db.py` defines own `Path.home() / '.config' / 'voice-input' / 'state.db'` | Avoids circular import (`state_db` ↔ `voice_input`) | Import CONFIG_DIR from voice_input (circular), env var (fragile) |

## 5. Non-Functional Requirements

### Performance
- `get_state()` latency: <5ms (local SQLite read, no network)
- `update_state()` latency: <10ms (WAL write + fsync)
- Icon update delay: max 1s (polling interval) — imperceptible vs current IPC ~100ms delay
- No impact on ASR pipeline latency (state operations are off the critical transcription path)

### Storage
- DB file size: ~12KB (single row + WAL + WAL-index)
- Growth: none (single row, updated in place)
- WAL file: up to 1MB during writes, auto-checkpointed by SQLite

### Reliability
- WAL mode reduces corruption risk vs rollback journal
- `BEGIN IMMEDIATE` prevents write conflicts
- `sqlite3.connect(timeout=5)` handles transient busy states
- Graceful degradation: all DB ops wrapped in try/except — daemon continues if DB is temporarily unavailable
- Self-initializing: `get_state()` calls `init_db()` if table missing — robust against out-of-order access

### Security
- DB file permissions: inherited from `~/.config/voice-input/` directory (user-only by default)
- No secrets stored in DB (only PIDs, status strings, file paths)
- No SQL injection risk: `update_state()` validates column names against whitelist

## 6. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | SQLite file corruption (power loss during write) | L | M | WAL mode provides crash recovery; DB is ephemeral state (not critical data — can be deleted and recreated) |
| 2 | Circular import `state_db` ↔ `voice_input` | M | H | state_db.py defines its own DB path constant; never imports from voice_input.py (CRITIC-R1-M1) |
| 3 | stop_recording() crash reading PID after migration | H (without mitigation) | H | Single `get_state()` call reads both recording_pid and recording_path from DB BEFORE kill sequence (CRITIC-R2-C1) |
| 4 | Existing tests reference removed functions/IPC commands | H | M | Test migration plan covers all affected files: test_unit.py, test_race.py, test_integration.py, test_post_processor_persistence.py, conftest.py |
| 5 | `state_db.py` not symlinked to install dir (`~/.local/share/voice-input/`) | M | H | Add symlink creation to deployment steps; daemon will fail to `import state_db` if missing |
| 6 | DB busy timeout under rapid toggle (user presses hotkey rapidly) | L | L | `sqlite3.connect(timeout=5)` waits up to 5s; state writes are <10ms; 5s budget is 500x headroom |
| 7 | Stale daemon_pid in DB after daemon crash (flock released but DB not updated) | M | L | `is_daemon_running()` verifies PID liveness with `os.kill(pid, 0)`; dead PID → cleanup DB |
| 8 | Legacy file cleanup races with running daemon on first upgrade | L | M | `ensure_config_dir()` calls `init_db()` first (migrates state), then deletes files; daemon's `_restore_post_processor_id()` already ran at startup |
| 9 | conftest.py `isolated_environment` fixture doesn't patch STATE_DB_PATH | H (without mitigation) | H | Must add STATE_DB_PATH to monkeypatch + call init_db(tmp_path) in fixture setup |
| 10 | Daemon icon 1s poll delay causes user-visible lag | L | L | 1s max delay is imperceptible; current IPC also has ~100ms latency; polling is simpler and more reliable |

## 7. User Story Mapping

| Story ID | Module(s) | Key Interfaces | Notes |
|----------|-----------|---------------|-------|
| US-001 | `state_db.py` (NEW) | `init_db()`, `get_state()`, `update_state()` | Foundation module. Schema: `daemon_state` table, WAL mode, BEGIN IMMEDIATE. Migrates legacy `current_post_processor.txt`. Self-initializing get_state(). |
| US-002 | `voice_input.py` (CLI functions) | `start_recording()`, `stop_recording()`, `toggle_recording()`, `is_recording()` | CRITIC-R2-C1: stop_recording reads PID+path from DB before kill. Remove IPC status commands. Add STATE_DB_PATH constant. Update `ensure_config_dir()` to call `init_db()`. |
| US-003 | `voice_input.py` (ASRDaemon class) | `_sync_status_from_db()`, `run()`, `is_daemon_running()`, `load_post_processor()`, `__init__()` | Daemon polls DB in socket_server timeout. Writes daemon_pid on startup/shutdown. Post-processor persistence moves to DB. |
| US-004 | `voice_input.py` (CLI + cleanup) | `show_status()`, `ensure_config_dir()` | DB-based status display. Remove `_persist/_restore_post_processor_id()`, `POST_PROCESSOR_STATE_FILE`, `is_process_running()`. Delete legacy files in ensure_config_dir(). Update all affected tests. |
