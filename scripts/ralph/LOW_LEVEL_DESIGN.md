# Low-Level Design: Anthropic Claude Fallback

Generated: 2026-05-18 12:20 CST
PRD: prd.json — `ralph/anthropic-fallback`, 5 stories, add Claude Haiku 4.5 as primary post-processor via SSH→Oracle, mirroring the existing `vertex_proxy.py` pattern
Architecture: HIGH_LEVEL_DESIGN.md

This document is the **single source of truth for module interfaces**. An integration-test writer must be able to write tests from this document alone, without reading the implementation. Every parameter and return value has a precise type; every error path is enumerated.

---

## 1. Module Interface Catalog

### Module A — `anthropic_proxy.py` (NEW, repo root)

Path: `/home/dev/code/voice_input/anthropic_proxy.py`. Deployed to `oracle-cloud:~/anthropic_proxy.py` via scp. **Self-contained**: must not import anything from the voice_input project.

#### Public Interface (CLI script)

```python
GCP_PROJECT_NA: None = None  # No GCP equivalent — Anthropic uses an API key, not ADC.

ANTHROPIC_KEY_PATH: Path = Path("~/.config/claude.secret").expanduser()
DEFAULT_MODEL: str = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS: int = 1024


def _trace(msg: str) -> None:
    """Write trace timing to stderr in '[TRACE] {msg}' format (mirrors vertex_proxy._trace)."""
    ...


def _import_anthropic() -> tuple[object, object]:
    """Lazy import of the anthropic SDK.

    Returns:
        (anthropic_module, Anthropic_client_class)

    Raises:
        ImportError: If `anthropic` package is not installed on this host.
    """
    ...


def _read_api_key() -> str:
    """Read API key from ANTHROPIC_KEY_PATH.

    Returns:
        The stripped key string. Never empty.

    Raises:
        FileNotFoundError: If the key file does not exist.
        PermissionError:   If the file is not readable.
        ValueError:        If the file is empty / whitespace-only.
    """
    ...


def print_help() -> None:
    """Print CLI usage to stdout and return. Caller exits 0 after this."""
    ...


def run_test() -> None:
    """Verify (a) `anthropic` SDK importable, (b) ANTHROPIC_KEY_PATH readable & non-empty.

    Postconditions on success:
        - prints 'OK: SDK import + API key file readable.' to stdout
        - calls sys.exit(0)

    Postconditions on failure:
        - prints 'FAIL: <reason>' to stderr
        - calls sys.exit(1)

    Does NOT perform a network call (parallels vertex_proxy's lazy-init reality,
    but explicitly: the Anthropic SDK has no cheap list-models equivalent that's
    worth the API cost on every --test).
    """
    ...


def main() -> None:
    """Entry point. Parses argv, then reads JSON from stdin and calls Anthropic.

    Argv handling (mirrors vertex_proxy.main):
        - no args            → process stdin JSON
        - '--help' / '-h'    → print_help(), exit 0
        - '--test'           → run_test() (which exits)
        - any other argv[1]  → 'Unknown argument: <arg>' to stderr, exit 1

    Stdin JSON schema (parsed via json.loads on full stdin read):
        {
            "system_prompt": str,         # default ""  (allowed to be empty)
            "user_input":    str,         # REQUIRED, non-empty
            "model":         str,         # default "claude-haiku-4-5-20251001"
            "max_tokens":    int,         # default 1024, must be 1..8192
        }

    Behavior:
        - Invalid JSON                          → 'Invalid JSON input: <e>' stderr, exit 1
        - Missing/empty 'user_input'            → "Missing 'user_input' in JSON" stderr, exit 1
        - SDK import failure                    → '<package> not installed: <e>' stderr, exit 1
        - API key read failure (any cause)      → 'API key error: <e>' stderr, exit 1
        - Anthropic API exception (any cause)   → 'Anthropic API error: <e>' stderr, exit 1
        - response.content is empty/None        → 'Anthropic returned empty response' stderr, exit 1
        - SUCCESS                               → print(text.strip()) to stdout, exit 0
                                                  where text = response.content[0].text

    Trace lines (to stderr, all prefixed '[TRACE] '):
        - sdk_init: <seconds>s          after _import_anthropic + Anthropic(api_key=...)
        - anthropic_api: <seconds>s     after client.messages.create returns

    Anthropic call shape:
        client = Anthropic(api_key=_read_api_key())
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,                          # plain str, not list-of-blocks
            messages=[{"role": "user", "content": user_input}],
        )
        text = response.content[0].text
    """
    ...
```

#### Events Emitted / Consumed
- **Consumes**: stdin JSON (schema above).
- **Emits**: stdout = polished text + newline (single line, may itself contain newlines from the model — caller `.strip()`s).
- **Emits**: stderr = `[TRACE]` lines + (on failure) one error message line.
- **Emits**: exit code (0 success, 1 any failure).

#### Invariants
- No imports from the voice_input project (verifiable by AST inspection in a unit test).
- No filesystem writes (only reads ANTHROPIC_KEY_PATH).
- Process is single-shot; does not loop or daemonize.

---

### Module B — `post_processor_configs.py` (MODIFY, local)

Only **new** symbols are detailed below. Existing functions (`process_with_ssh_claude`, `process_with_vertex_ai`, `process_with_gemini_merge`, `_run_vertex_proxy`, `apply_vocab`, `glossary_context`, `load_vocab`, `save_vocab`, `diff_to_vocab`, classes `PostProcessorLoader`, `PostProcessorInference`) are bit-for-bit unchanged.

#### Public Interface

```python
def process_with_anthropic(
    text: str,
    config: dict[str, object],
    glossary_ctx: str = "",
) -> str:
    """Call Anthropic Claude via SSH→Oracle proxy for single-text polishing.

    Args:
        text:         ASR transcription text to polish (post filler removal, post vocab apply).
        config:       Preset config dict. Required keys:
                          "ssh_host":         str    (e.g. "oracle-cloud")
                          "proxy_script":     str    (e.g. "~/anthropic_proxy.py")
                      Optional keys (with defaults matching the existing Vertex AI helper):
                          "model":            str    default "claude-haiku-4-5-20251001"
                          "timeout":          int    default 15
                          "min_text_len":     int    default 15
                          "system_prompt":            str  (used if system_prompt_file absent)
                          "system_prompt_file":       str  (path relative to VOICE_INPUT_DATA_DIR)
                          "user_prompt_template":     str  (used if *_file absent)
                          "user_prompt_template_file": str (path relative to VOICE_INPUT_DATA_DIR)
        glossary_ctx: Glossary context string appended to the system prompt (may be empty).

    Returns:
        Polished text on success, the input `text` on any failure / guard trip.
        Never raises (all exceptions are caught and logged).

    Failure modes (each returns the input `text`):
        - `text` is empty                       → returns "" (short-circuit, no SSH call)
        - len(text) < min_text_len              → returns text
        - subprocess.TimeoutExpired             → notify, return text
        - subprocess returncode != 0            → fall through to OpenRouter fallback
        - OpenRouter also returns None          → notify "Anthropic + OpenRouter both failed", return text
        - hallucination guard: len(output) > 2*len(text)   → return text
        - question guard: '？' in text but '？','?' not in output → return text

    JSON stdin payload sent to anthropic_proxy.py:
        {
            "system_prompt": <resolved system prompt + (optional) "\n\n" + glossary_ctx>,
            "user_input":    <user_prompt_template.format(text=text) or text>,
            "model":         config.get("model", "claude-haiku-4-5-20251001"),
            "max_tokens":    min(8192, max(512, len(user_input))),
        }
    SSH cmd argv list:
        ["ssh", "-o", "ConnectTimeout=5", config["ssh_host"],
         "python3", config["proxy_script"]]

    Reuses `_run_vertex_proxy(cmd, stdin_data, timeout, fallback_model=None)` —
    fallback_model is always None for Anthropic (no Anthropic fallback model in preset).

    Log lines (logging.info / logging.warning):
        - "[PROMPT] loaded system prompt: <path>"   (if system_prompt_file)
        - "[PROMPT] loaded user template: <path>"   (if user_prompt_template_file)
        - "[TRACE] anthropic round-trip: ..."        (via _run_vertex_proxy)
        - "[OPENROUTER] fallback success for anthropic-fix: <N> chars"   (if fallback wins)
        - "[SSH] anthropic-fix success: <N>→<M> chars"
    """
    ...


def process_with_anthropic_merge(
    primary_text: str,
    secondary_text: Optional[str],
    config: dict[str, object],
    glossary_ctx: str = "",
) -> str:
    """Merge dual-ASR transcriptions via Anthropic Claude → polished text.

    Args:
        primary_text:   Primary ASR (SenseVoice) text after fillers + vocab.
        secondary_text: Secondary ASR (faster-whisper) raw text, or None when secondary
                        model is not loaded.
        config:         Same shape as `process_with_anthropic`'s `config` (no
                        "user_prompt_template*" keys — merge user_input is built in code).
        glossary_ctx:   Glossary context string (may be empty).

    Returns:
        Merged/polished text on success, `primary_text` on any failure / guard trip.
        Never raises.

    Special early-return rules (mirror process_with_gemini_merge):
        - primary_text == "" and secondary_text truthy → returns secondary_text
        - primary_text == "" and not secondary_text    → returns ""
        - len(primary_text) < min_text_len and secondary_text longer → returns secondary_text
        - len(primary_text) < min_text_len otherwise → returns primary_text

    User-input construction:
        - secondary_text is not None → "Chinese ASR: {primary}\nEnglish ASR: {secondary}"
        - secondary_text is None     → "Chinese ASR: {primary}"

    Failure modes (each returns primary_text):
        - subprocess.TimeoutExpired                    → log warning, fall through to OpenRouter
        - subprocess returncode != 0                   → fall through to OpenRouter
        - OpenRouter also returns None                 → notify "Anthropic merge + OpenRouter both failed", return primary_text
        - len(output) > 2*len(primary_text)            → return primary_text
        - '？' in primary_text and not in output       → return primary_text

    JSON stdin payload, SSH cmd, _run_vertex_proxy reuse: identical to process_with_anthropic
    except model defaults to config.get("model", "claude-haiku-4-5-20251001") and
    user_input is the merge format above.

    Log lines:
        - "[MERGE] primary=<N> chars, secondary=<M|None> chars"
        - "[SSH] anthropic-merge success: primary=<N>, secondary=<M|None> → <K> chars"
        - "[OPENROUTER] fallback success for anthropic-merge: <K> chars"
    """
    ...
```

#### Events Emitted / Consumed
- **Emits** (via subprocess): SSH command to `oracle-cloud` invoking `python3 ~/anthropic_proxy.py`.
- **Emits** (on failure): `voice_input.notify(...)` (lazy import to avoid circular dep), `logging.info|warning|error`.
- **Consumes**: `openrouter_client.call_openrouter(system_prompt, user_input, timeout)` as fallback.
- **Consumes**: `_run_vertex_proxy` (existing) — its 429-retry branch is a no-op for Anthropic (Anthropic 429 error shape contains `rate_limit_error`, not the substring "429" or "RESOURCE_EXHAUSTED"); this degraded behavior is acceptable per HIGH_LEVEL_DESIGN risk #7.

#### Invariants
- Function signatures are positional/keyword compatible with `process_with_vertex_ai` / `process_with_gemini_merge` (`glossary_ctx` is keyword-only-by-convention with a default).
- No mutation of `config` or `glossary_ctx` arguments.
- Never raises; all paths return `str`.

---

### Module C — `post_processor_presets.py` (MODIFY, local)

#### Public Interface (changes only)

`POST_PROCESSOR_PRESETS: dict[str, dict[str, object]]` gains two entries; `DEFAULT_POST_PROCESSOR: str` is reassigned.

```python
POST_PROCESSOR_PRESETS["claude-fix"] = {
    "name":        "Claude Fix (Anthropic)",
    "description": "ASR error correction via Claude Haiku 4.5 (Anthropic) over SSH",
    "framework":   "anthropic",
    "config": {
        "ssh_host":                  "oracle-cloud",
        "proxy_script":              "~/anthropic_proxy.py",
        "model":                     "claude-haiku-4-5-20251001",
        "timeout":                   15,
        "min_text_len":              15,
        "vocab_min_count":           3,
        "system_prompt_file":        "prompts/gemini-fix-system.txt",
        "user_prompt_template_file": "prompts/haiku-fix-user.txt",
    },
}

POST_PROCESSOR_PRESETS["claude-merge"] = {
    "name":        "Claude Merge (Dual ASR)",
    "description": "Merge SenseVoice + faster-whisper via Claude Haiku 4.5",
    "framework":   "anthropic-merge",
    "config": {
        "ssh_host":           "oracle-cloud",
        "proxy_script":       "~/anthropic_proxy.py",
        "model":              "claude-haiku-4-5-20251001",
        "timeout":            15,
        "min_text_len":       15,
        "vocab_min_count":    3,
        "system_prompt_file": "prompts/gemini-merge-system.txt",
        # No user_prompt_template* — merge user_input is built in code.
    },
}

DEFAULT_POST_PROCESSOR = "claude-merge"   # was "gemini-merge"
```

#### Invariants (testable)
- The eight existing keys `{"none", "chinese-text-correction", "qwen3-0.6b", "minicpm4-0.5b", "haiku-fix", "haiku-expand", "gemini-fix", "gemini-merge"}` are present with **bit-for-bit identical** values (test: deep-compare against a snapshot).
- New keys exist and have exactly the schemas above.
- `DEFAULT_POST_PROCESSOR == "claude-merge"`.
- `POST_PROCESSOR_PRESETS[DEFAULT_POST_PROCESSOR]["framework"] == "anthropic-merge"` (consistency check).
- Module exports unchanged: `POST_PROCESSOR_PRESETS`, `DEFAULT_POST_PROCESSOR`, `VOICE_INPUT_DATA_DIR`, `MODELS_DIR`.

---

### Module D — `voice_input.py` (MODIFY, local)

Two methods on `class ASRDaemon` change. **No public signature change.**

#### Public Interface (changed methods)

```python
class ASRDaemon:

    def load_post_processor(self, preset_id: Optional[str] = None) -> None:
        """Load post-processor preset and (un)load secondary ASR as needed.

        Changes from current behavior:
          - Vocab is loaded for the framework set:
                {"ssh-claude", "vertex-ai", "vertex-ai-merge",
                 "anthropic", "anthropic-merge"}    ← was the first three only
          - Secondary ASR (faster-whisper) is loaded for the framework set:
                {"vertex-ai-merge", "anthropic-merge"}   ← was "vertex-ai-merge" only
          - **Critical** US-004 invariant: switching between two merge frameworks
            (e.g. vertex-ai-merge → anthropic-merge or vice versa) MUST NOT
            call `_unload_secondary_model()` followed by `_load_secondary_model()`.
            Implementation contract: if `self.post_processor_framework` (the *new*
            framework being set) is in the merge set AND `self._secondary_model`
            is already non-None, `_load_secondary_model()` is a no-op.
            Equivalently, an integration test can switch
            gemini-merge → claude-merge and assert
            `_load_secondary_model.call_count == 0` after the second switch.

        Args:
            preset_id: Preset key in POST_PROCESSOR_PRESETS, or None to reload
                       the current preset.

        Side effects (in order):
            1. self.post_processor_model = PostProcessorLoader.load_post_processor(preset_id)
            2. self.current_post_processor_id = preset_id
            3. update_state(STATE_DB_PATH, post_processor=preset_id)
            4. self.post_processor_framework = preset["framework"]
            5. If framework ∈ vocab-set: self._vocab = load_vocab()
            6. If framework ∈ merge-set: _load_secondary_model() (idempotent)
               else:                     _unload_secondary_model()

        Raises:
            RuntimeError: if preset_id not in POST_PROCESSOR_PRESETS.
            ValueError:   if framework == "ssh-claude" and config is missing
                          (i.e. "haiku-expand" placeholder selected).

        Failure recovery (any other Exception caught internally):
            - self.post_processor_model = None
            - self.current_post_processor_id = "none"
            - self.post_processor_framework = "regex"
            - _unload_secondary_model()
        """
        ...

    def _post_process(self, text: str) -> str:
        """Apply post-processing to transcribed text.

        Changes from current behavior (in the SSH-dispatch block):
          - Framework gate extends from
                ("ssh-claude", "vertex-ai", "vertex-ai-merge")
            to also include
                ("anthropic", "anthropic-merge").
          - Merge-vs-fix branch extends:
                merge frameworks: {"vertex-ai-merge", "anthropic-merge"}
                    vertex-ai-merge  → process_with_gemini_merge(result, secondary, config, glossary_ctx)
                    anthropic-merge  → process_with_anthropic_merge(result, secondary, config, glossary_ctx)
                fix frameworks: {"ssh-claude", "vertex-ai", "anthropic"} via dict-dispatch:
                    {
                        "ssh-claude": process_with_ssh_claude,
                        "vertex-ai":  process_with_vertex_ai,
                        "anthropic":  process_with_anthropic,
                    }[framework](result, config, glossary_ctx)

        Args:
            text: Raw ASR text (already trimmed of leading clipping).

        Returns:
            Post-processed text. On any internal failure, returns the input text
            (or the pre-LLM `result` — same fall-through behavior as today).

        Side effects (only on the SSH-dispatch path):
            - update_state(status="polishing") before LLM call (blue icon)
            - apply_vocab / glossary_context / diff_to_vocab / save_vocab as today
        """
        ...
```

#### Events Emitted / Consumed
- **Emits**: `update_state(status="polishing")` (blue icon) before any LLM call.
- **Consumes** (lazy import inside the function): `process_with_anthropic`, `process_with_anthropic_merge` from `post_processor_configs`.

#### Invariants
- Method signatures unchanged (`preset_id` still optional, `text` still positional).
- Single source of truth for framework→function mapping lives in `_post_process` (two dicts: the merge-set and the fix-dispatch dict). No new registry module.

---

### Module E — `state_db.py` (MODIFY, local)

#### Public Interface (changes only)

Public functions `init_db`, `get_state`, `update_state` keep their signatures and behavior. Three **module-level constants** change:

```python
_DEPRECATED_PP: dict[str, str] = {
    "firered-punc":  "claude-merge",   # was "gemini-merge" — bumped to new default
    "gemini-merge":  "claude-merge",   # NEW — auto-migrate Gemini-merge users
}

_SAFE_DEFAULT: dict[str, object] = {
    "id": 1,
    "status": "idle",
    "daemon_pid": None,
    "recording_pid": None,
    "recording_path": None,
    "post_processor": "claude-merge",   # was "gemini-merge"
    "updated_at": None,
}

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS daemon_state (
    id INTEGER PRIMARY KEY CHECK(id=1),
    status TEXT NOT NULL DEFAULT 'idle',
    daemon_pid INTEGER,
    recording_pid INTEGER,
    recording_path TEXT,
    post_processor TEXT NOT NULL DEFAULT 'claude-merge',  -- was 'gemini-merge'
    updated_at TEXT
)"""
```

#### Migration semantics (contract for tests)

`get_state(db_path)` behavior on the deprecated values:

| DB stored `post_processor` | get_state returns | DB written back |
|---|---|---|
| `"firered-punc"`   | `"claude-merge"` | `"claude-merge"` |
| `"gemini-merge"`   | `"claude-merge"` | `"claude-merge"` |
| `"gemini-fix"`     | `"gemini-fix"`   | unchanged |
| `"haiku-fix"`      | `"haiku-fix"`    | unchanged |
| `"none"`           | `"none"`         | unchanged |
| `"claude-merge"`   | `"claude-merge"` | unchanged |
| `"claude-fix"`     | `"claude-fix"`   | unchanged |
| any other          | as-stored        | unchanged |

Logging: migration must emit a log line at **`INFO`** level (not `WARNING` / `ERROR`) when a write-back occurs. Suggested format: `"[STATE-DB] migrated post_processor: '<old>' → '<new>'"`. (This line must be assertable in a unit test via `caplog`.)

#### Invariants
- `_DEPRECATED_PP` keys must NOT include `"gemini-fix"` or `"haiku-fix"` (explicit per PRD US-005 AC).
- All three constants (`_DEPRECATED_PP` target, `_SAFE_DEFAULT["post_processor"]`, `_CREATE_TABLE_SQL` DEFAULT clause) must equal `"claude-merge"` (testable by parsing the SQL string for `DEFAULT '...'`).
- Fresh DB (no prior row) `get_state()` → `_SAFE_DEFAULT` copy.
- After `init_db(fresh_path)` and `get_state(fresh_path)`, `state["post_processor"] == "claude-merge"`.

---

### Module F — `openrouter_client.py` (UNCHANGED — interface restated for completeness)

```python
OPENROUTER_API_URL: str   = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL:   str   = "nousresearch/hermes-3-llama-3.1-405b:free"
API_KEY_PATH:       Path  = Path("~/.config/voice-input/openrouter_api_key").expanduser()

def call_openrouter(system_prompt: str, user_input: str, timeout: int = 15) -> Optional[str]:
    """Call OpenRouter. Returns polished text or None on any failure."""
    ...
```

Used as final fallback by both new functions in Module B with the same arguments and timeout as the corresponding Vertex AI functions use today.

---

## 2. Inter-Module Contracts

| Caller | Callee | Method | Input Type | Output Type | Error Cases |
|---|---|---|---|---|---|
| `voice_input.ASRDaemon._post_process` | `post_processor_configs.process_with_anthropic` | function call | `(text: str, config: dict[str, object], glossary_ctx: str)` | `str` (polished or original text) | **Never raises.** Returns input `text` on any failure (timeout, non-zero rc, OpenRouter fail, hallucination guard, question guard, empty text, short text). |
| `voice_input.ASRDaemon._post_process` | `post_processor_configs.process_with_anthropic_merge` | function call | `(primary: str, secondary: Optional[str], config: dict[str, object], glossary_ctx: str)` | `str` (merged or `primary` text) | **Never raises.** Returns `primary` on failures listed above; returns `secondary` only when `primary == ""` or short and `secondary` is longer. |
| `post_processor_configs.process_with_anthropic{,_merge}` | `post_processor_configs._run_vertex_proxy` | function call | `(cmd: list[str], stdin_data: str, timeout: int, fallback_model: None)` | `subprocess.CompletedProcess` | `subprocess.TimeoutExpired` propagates to caller, which catches and falls back to OpenRouter. |
| `post_processor_configs.process_with_anthropic{,_merge}` | `openrouter_client.call_openrouter` | function call | `(system_prompt: str, user_input: str, timeout: int)` | `Optional[str]` | Returns `None` on any HTTP/network/parse failure (already handled in existing code). |
| `_run_vertex_proxy` | `subprocess.run` | argv exec | `cmd = ["ssh","-o","ConnectTimeout=5", ssh_host, "python3", proxy_script]`, `stdin=stdin_data`, `timeout=timeout` | `subprocess.CompletedProcess` (with `.returncode`, `.stdout`, `.stderr`) | Timeout → `TimeoutExpired`; non-zero rc → propagated up; stderr contains `[TRACE]` lines + error message. |
| `anthropic_proxy.py` | Anthropic Messages API | `client.messages.create` | `model: str, max_tokens: int, system: str, messages: list[dict[str,str]]` | `Message` object with `.content[0].text: str` | Any SDK/network/HTTP exception → caught in `main()`, printed to stderr, exit 1. |
| `anthropic_proxy.py` | filesystem | `Path("~/.config/claude.secret").read_text()` | path | str | FileNotFoundError / PermissionError / empty → caught in `main()`, exit 1. |
| `voice_input.ASRDaemon.load_post_processor` | `post_processor_configs.load_vocab` | function call | `()` | `dict` | Never raises; returns `{}` on missing/corrupt vocab. |
| `voice_input.ASRDaemon.load_post_processor` | `self._load_secondary_model` / `self._unload_secondary_model` | method calls | `()` | `None` | Idempotent. `_load_secondary_model` no-ops if model already loaded; `_unload_secondary_model` no-ops if not loaded. (Existing behavior — relied upon for US-004 invariant.) |
| `voice_input.ASRDaemon` `__init__` / `_handle_*` | `state_db.get_state` | function call | `(db_path: Optional[Path])` | `dict[str, object]` with `post_processor` already migrated | Never raises; returns `_SAFE_DEFAULT` on DB error. Migration writes back synchronously before return. |

---

## 3. Data Models

### 3.1 `AnthropicProxyRequest` (TypedDict — only inside `anthropic_proxy.py` for documentation; runtime is plain `dict`)

```python
from typing import TypedDict, NotRequired

class AnthropicProxyRequest(TypedDict):
    system_prompt: NotRequired[str]   # default "", may be empty
    user_input:    str                # REQUIRED, non-empty
    model:         NotRequired[str]   # default "claude-haiku-4-5-20251001"
    max_tokens:    NotRequired[int]   # default 1024, range [1, 8192]
```

Wire format: UTF-8 JSON, `ensure_ascii=False` from caller, single document on stdin (caller reads `sys.stdin.read()` once).

Validation rules (enforced in `main()`):
- JSON must parse → else stderr "Invalid JSON input: …", exit 1.
- `user_input` must be a non-empty string → else stderr "Missing 'user_input' in JSON", exit 1.
- Other fields fall back to defaults silently.

### 3.2 `AnthropicProxyResponse` (informal — stdout shape)

Plain text. One UTF-8 string (the model output, stripped). Caller (`_run_vertex_proxy`) reads `result.stdout.strip()`.

### 3.3 `PresetConfig` (existing shape, reused — no new dataclass)

The two new presets follow the existing `dict[str, object]` shape used by Vertex AI presets. Field types:

| Key | Type | Required for | Notes |
|---|---|---|---|
| `name` | `str` | UI | top-level key on preset |
| `description` | `str` | UI | top-level key |
| `framework` | `str` | dispatch | top-level key, one of `{"regex","ssh-claude","vertex-ai","vertex-ai-merge","anthropic","anthropic-merge","llama-cpp"}` |
| `config` | `dict` | runtime | top-level key |
| `config.ssh_host` | `str` | SSH | e.g. `"oracle-cloud"` |
| `config.proxy_script` | `str` | SSH | e.g. `"~/anthropic_proxy.py"` (tilde-prefixed, not expanded locally — Oracle resolves it) |
| `config.model` | `str` | API | `"claude-haiku-4-5-20251001"` |
| `config.timeout` | `int` | SSH | seconds |
| `config.min_text_len` | `int` | guard | default 15 |
| `config.vocab_min_count` | `int` | vocab apply | default 3 |
| `config.system_prompt_file` | `str` | prompt | path relative to `VOICE_INPUT_DATA_DIR` |
| `config.user_prompt_template_file` | `str` | optional | only for `anthropic` (fix), NOT for `anthropic-merge` |

Invariant: `config.fallback_model` is **absent** for both new presets (Anthropic has no fallback model in this design).

### 3.4 `DaemonState` (existing schema, semantics changed)

Table `daemon_state` (single row, `id=1`). Schema unchanged. Default of `post_processor` column changes from `'gemini-merge'` to `'claude-merge'`.

Serialization: SQLite native types (TEXT / INTEGER / NULL).

---

## 4. Error Taxonomy

| Module | Error / Exit Condition | Code | When Raised | Recovery |
|---|---|---|---|---|
| `anthropic_proxy.py` | `Invalid JSON input` | exit 1 | stdin not parseable as JSON | caller (`_run_vertex_proxy`) sees `rc=1`, stderr msg → returns `CompletedProcess(rc=1)` → caller falls back to OpenRouter |
| `anthropic_proxy.py` | `Missing 'user_input' in JSON` | exit 1 | `user_input` absent or empty | same as above |
| `anthropic_proxy.py` | `anthropic SDK not installed: <e>` | exit 1 | `import anthropic` ImportError | same as above; operator action: `pip install --user anthropic` on Oracle |
| `anthropic_proxy.py` | `API key error: <e>` | exit 1 | `~/.config/claude.secret` missing / empty / unreadable | same as above; operator action: deploy key file with chmod 600 |
| `anthropic_proxy.py` | `Anthropic API error: <e>` | exit 1 | any exception from `client.messages.create` (5xx, network, auth, 429, …) | same as above |
| `anthropic_proxy.py` | `Anthropic returned empty response` | exit 1 | `response.content` empty or `[0].text` falsy | same as above |
| `anthropic_proxy.py` | `Unknown argument: <arg>` | exit 1 | argv[1] not in `{--help,-h,--test}` | hard failure (caller misuse) |
| `post_processor_configs.process_with_anthropic` | (none — does not raise) | return input `text` | timeout / non-zero rc / OpenRouter fail / guards | three-tier fallback: Anthropic → OpenRouter → original text |
| `post_processor_configs.process_with_anthropic_merge` | (none — does not raise) | return `primary_text` | same | same |
| `voice_input.ASRDaemon.load_post_processor` | `RuntimeError("Unknown post-processor: <id>")` | propagated | preset_id absent from POST_PROCESSOR_PRESETS | caller logs + notifies; daemon command returns error |
| `voice_input.ASRDaemon.load_post_processor` | `ValueError("Haiku Expand is not yet implemented")` | propagated | haiku-expand selected (existing path, unchanged) | caller notifies and returns error |
| `voice_input.ASRDaemon.load_post_processor` | any other Exception | caught internally | model load failure | fallback to `"none"` preset + regex |
| `state_db.get_state` | (none — never raises) | return `_SAFE_DEFAULT` | any sqlite3 / OS error | best-effort, logged at WARNING; subsequent migration retry on next call |
| `state_db.update_state` | `ValueError` | propagated | unknown column in kwargs (existing behavior) | caller responsibility |

---

## 5. Configuration Contract

### 5.1 Local machine config (no new keys)

| Key | Type | Default | Required | Used By |
|---|---|---|---|---|
| `~/.config/voice-input/state.db` | SQLite file | created on init | yes | `state_db.{init_db,get_state,update_state}` |
| `~/.config/voice-input/openrouter_api_key` | UTF-8 text file | absent | optional (no key → OpenRouter fallback disabled) | `openrouter_client._read_api_key` |
| `~/.local/share/voice-input/vocab.json` | UTF-8 JSON | `{}` | optional | `post_processor_configs.{load_vocab,save_vocab}` |
| `~/.local/share/voice-input/prompts/gemini-fix-system.txt` | UTF-8 text | existing file | yes (reused by `claude-fix`) | `process_with_anthropic` (system prompt loader) |
| `~/.local/share/voice-input/prompts/haiku-fix-user.txt` | UTF-8 text | existing file | yes (reused by `claude-fix`) | `process_with_anthropic` (user template loader) |
| `~/.local/share/voice-input/prompts/gemini-merge-system.txt` | UTF-8 text | existing file | yes (reused by `claude-merge`) | `process_with_anthropic_merge` (system prompt loader) |

### 5.2 Oracle Cloud config (new)

| Key | Type | Default | Required | Used By |
|---|---|---|---|---|
| `oracle-cloud:~/anthropic_proxy.py` | Python script (deployed via scp) | (must be deployed) | YES — before merging US-003 (which flips default) | `process_with_anthropic{,_merge}` via SSH |
| `oracle-cloud:~/.config/claude.secret` | UTF-8 text file, mode 0600, single line (API key) | (must be provisioned, **already deployed per PRD**) | YES | `anthropic_proxy._read_api_key` |
| `oracle-cloud: anthropic` Python package | PyPI | (must be installed: `pip install --user anthropic`) | YES | `anthropic_proxy._import_anthropic` |
| `oracle-cloud:~/vertex_proxy.py` | existing | unchanged | (unchanged) | Gemini path remains functional for fallback selection |

### 5.3 Preset config keys (new — schema for the two new entries in `POST_PROCESSOR_PRESETS`)

Already itemized in §1 Module C and §3.3.

### 5.4 Deployment preconditions (testable gates per US)

| US | Gate before `passes: true` |
|---|---|
| US-001 | `ssh oracle-cloud python3 ~/anthropic_proxy.py --test` exits 0 |
| US-002 | Unit + integration test of `process_with_anthropic{,_merge}` (real proxy or mock subprocess) passes |
| US-003 | `DEFAULT_POST_PROCESSOR == "claude-merge"` AND deep-equal snapshot of existing 8 presets matches |
| US-004 | Switching `gemini-merge` → `claude-merge` calls `_load_secondary_model` exactly **0** additional times (asserted via mock) |
| US-005 | Reading `gemini-merge` from a test-fixture DB returns `"claude-merge"` AND writes it back; reading `gemini-fix` returns `"gemini-fix"` and does NOT write back |
| ALL | `voice-e2e-test` skill run end-to-end with `current_post_processor = "claude-merge"` succeeds on a real recording |

---

<promise>COMPLETE</promise>
