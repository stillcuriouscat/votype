# Low-Level Design: Gemini Output Truncation Fix

Generated: 2026-04-14 15:20 CST
PRD: prd.json — ralph/gemini-output-truncation-fix, 3 stories, fix hardcoded max_output_tokens=512 and log truncation
Architecture: HIGH_LEVEL_DESIGN.md

## 1. Module Interface Catalog

### Module A: `vertex_proxy.py` (Oracle Cloud: `~/vertex_proxy.py`)

Self-contained CLI proxy. No project imports.

#### Public Interface (CLI — stdin/stdout)

**Stdin JSON schema (after US-001):**

```python
class VertexProxyInput(TypedDict):
    system_prompt: str           # System instruction for Gemini
    user_input: str              # User text to process
    model: str                   # Gemini model name, default "gemini-2.5-flash"
    region: str                  # Vertex AI region, default "global"
    max_output_tokens: NotRequired[int]  # NEW (US-001). Max output tokens, default 512
```

**Stdout:** plain text (corrected text, no JSON wrapping)

**Exit codes:** 0 = success, 1 = failure (stderr has error message)

#### Internal functions (existing, modified)

```python
def main() -> None:
    """Read JSON from stdin, call Gemini, write result to stdout.

    Changes (US-001):
    - Reads data.get("max_output_tokens", 512) from parsed stdin JSON.
    - Passes dynamic value to GenerateContentConfig instead of hardcoded 512.
    """
    ...

def print_help() -> None:
    """Print usage information.

    Changes (US-001):
    - Adds max_output_tokens to the 'Stdin JSON fields' section.
    """
    ...
```

#### Exact change locations

| Line | Current code | New code |
|------|-------------|----------|
| 120-123 | `system_prompt = data.get(...)` ... `region = data.get(...)` | Add: `max_output_tokens = data.get("max_output_tokens", 512)` after line 123 |
| 150 | `max_output_tokens=512,` | `max_output_tokens=max_output_tokens,` |
| 65 | (help text, "region" entry) | Add line: `"  max_output_tokens  Max output tokens (default: 512, optional)\n"` |

#### Events Emitted / Consumed
- Emits: `[TRACE] sdk_init: {elapsed}s` (stderr), `[TRACE] gemini_api: {elapsed}s` (stderr)
- Consumes: stdin JSON (one-shot, then exits)

---

### Module B: `post_processor_configs.py` (Local)

#### Public Interface

```python
def process_with_vertex_ai(
    text: str,
    config: dict[str, str | int],
    glossary_ctx: str = "",
) -> str:
    """Call Vertex AI Gemini via SSH proxy for text polishing.

    Changes (US-002):
    - Computes max_output_tokens = min(8192, max(512, len(user_input)))
    - Includes 'max_output_tokens' key in stdin_data JSON dict.

    Args:
        text: ASR transcription text to polish.
        config: Preset config dict with ssh_host, proxy_script, model, etc.
        glossary_ctx: Glossary context string to append to system prompt.

    Returns:
        Polished text on success, original text on any failure.
    """
    ...

def process_with_gemini_merge(
    primary_text: str,
    secondary_text: str | None,
    config: dict[str, str | int],
    glossary_ctx: str = "",
) -> str:
    """Merge two ASR transcriptions via Vertex AI Gemini.

    Changes (US-002):
    - Computes max_output_tokens = min(8192, max(512, len(user_input)))
    - Includes 'max_output_tokens' key in stdin_data JSON dict.

    Args:
        primary_text: Primary ASR (SenseVoice) transcription.
        secondary_text: Secondary ASR (faster-whisper) transcription, or None.
        config: Preset config dict with ssh_host, proxy_script, model, etc.
        glossary_ctx: Glossary context string to append to system prompt.

    Returns:
        Merged/polished text on success, primary_text on any failure.
    """
    ...
```

#### Unchanged public interfaces (for completeness, not modified)

```python
def process_with_ssh_claude(text: str, config: dict[str, str | int], glossary_ctx: str = "") -> str: ...
def _run_vertex_proxy(cmd: list[str], stdin_data: str, timeout: int, max_retries: int = 1, fallback_model: str | None = None) -> subprocess.CompletedProcess: ...
def load_vocab(vocab_path: str | None = None) -> dict[str, dict[str, dict[str, int]]]: ...
def apply_vocab(text: str, vocab: dict[str, dict[str, dict[str, int]]], min_count: int) -> str: ...
def glossary_context(vocab: dict[str, dict[str, dict[str, int]]]) -> str: ...
def diff_to_vocab(original: str, polished: str, vocab: dict[str, dict[str, dict[str, int]]]) -> dict[str, dict[str, dict[str, int]]]: ...
def save_vocab(vocab: dict[str, dict[str, dict[str, int]]], vocab_path: str | None = None) -> None: ...
```

#### Exact change locations in `process_with_vertex_ai()`

| Line | Current code | New code |
|------|-------------|----------|
| 393-398 | `stdin_data = json.dumps({"system_prompt": ..., "user_input": ..., "model": ..., "region": ...}, ...)` | Add `"max_output_tokens": min(8192, max(512, len(user_input)))` to the dict literal |

Concrete insertion — between current line 397 (`"region": ...`) and line 398 (`}, ensure_ascii=False)`):

```python
# Current lines 393-398:
stdin_data = json.dumps({
    "system_prompt": system_prompt,
    "user_input": user_input,
    "model": config.get("model", "gemini-2.5-flash"),
    "region": config.get("vertex_region", "global"),
}, ensure_ascii=False)

# After US-002:
max_output_tokens = min(8192, max(512, len(user_input)))
stdin_data = json.dumps({
    "system_prompt": system_prompt,
    "user_input": user_input,
    "model": config.get("model", "gemini-2.5-flash"),
    "region": config.get("vertex_region", "global"),
    "max_output_tokens": max_output_tokens,
}, ensure_ascii=False)
```

#### Exact change locations in `process_with_gemini_merge()`

| Line | Current code | New code |
|------|-------------|----------|
| 515-520 | `stdin_data = json.dumps({"system_prompt": ..., "user_input": ..., "model": ..., "region": ...}, ...)` | Add `"max_output_tokens": min(8192, max(512, len(user_input)))` to the dict literal |

Concrete insertion — between current line 519 (`"region": ...`) and line 520 (`}, ensure_ascii=False)`):

```python
# Current lines 515-520:
stdin_data = json.dumps({
    "system_prompt": system_prompt,
    "user_input": user_input,
    "model": config.get("model", "gemini-2.5-flash"),
    "region": config.get("vertex_region", "global"),
}, ensure_ascii=False)

# After US-002:
max_output_tokens = min(8192, max(512, len(user_input)))
stdin_data = json.dumps({
    "system_prompt": system_prompt,
    "user_input": user_input,
    "model": config.get("model", "gemini-2.5-flash"),
    "region": config.get("vertex_region", "global"),
    "max_output_tokens": max_output_tokens,
}, ensure_ascii=False)
```

#### Events Emitted / Consumed
- Emits: logging.info `[TRACE]`, `[PROMPT]`, `[MERGE]`, `[SSH]`, `[OPENROUTER]`, `[VOCAB]`
- Consumes: preset config dicts from `post_processor_presets.POST_PROCESSOR_PRESETS`

#### `_run_vertex_proxy` fallback preserves `max_output_tokens` (no change needed)

Line 320-322 in `_run_vertex_proxy`: fallback replaces only `payload["model"]`, not `payload["max_output_tokens"]`. The dynamic value is preserved through all retry/fallback paths automatically.

---

### Module C: `voice_input.py` (Local)

#### Relevant functions (US-003 changes only)

```python
def _log(tag: str, message: str) -> None:
    """Write a structured log line to the notify log file.

    No changes to this function. Changes are to its CALL SITES.
    """
    ...
```

#### Exact change locations (US-003)

| Line | Current code | New code |
|------|-------------|----------|
| 1054 | `_log("PP", f"input ({self.current_post_processor_id}): {text[:120]}")` | `_log("PP", f"input ({self.current_post_processor_id}): {text}")` |
| 1128 | `_log("PP", f"output ({elapsed:.2f}s): {result[:120]}")` | `_log("PP", f"output ({elapsed:.2f}s): {result}")` |
| 1219 | `_log("ASR-2", f"secondary: {self._last_secondary_text[:120]}")` | `_log("ASR-2", f"secondary: {self._last_secondary_text}")` |
| 1225 | `_log("ASR", f"raw: {raw_primary[:120]}")` | `_log("ASR", f"raw: {raw_primary}")` |

**Out of scope (not in PRD):**
- Line 1066: `_log("PUNC", f"applied punctuation: {result[:120]}")` — punctuation log, not ASR/PP output.

---

## 2. Inter-Module Contracts

| Caller | Callee | Method | Input Type | Output Type | Error Cases |
|--------|--------|--------|-----------|-------------|-------------|
| `post_processor_configs.process_with_vertex_ai()` | `vertex_proxy.py` (via SSH subprocess) | stdin JSON → stdout text | `VertexProxyInput` (JSON string via stdin) | `str` (stdout, plain text) | Exit 1 → stderr error message; `subprocess.TimeoutExpired` on timeout |
| `post_processor_configs.process_with_gemini_merge()` | `vertex_proxy.py` (via SSH subprocess) | stdin JSON → stdout text | `VertexProxyInput` (JSON string via stdin) | `str` (stdout, plain text) | Exit 1 → stderr error message; `subprocess.TimeoutExpired` on timeout |
| `post_processor_configs._run_vertex_proxy()` | `subprocess.run` | Shell command execution | `list[str]` (cmd) + `str` (stdin_data) | `subprocess.CompletedProcess` | `subprocess.TimeoutExpired` (re-raised); 429/RESOURCE_EXHAUSTED → retry + fallback |
| `post_processor_configs.process_with_vertex_ai()` | `openrouter_client.call_openrouter()` | HTTP API call | `(str, str, int)` — (system_prompt, user_input, timeout) | `str | None` | Returns `None` on any failure (HTTP, network, parse) |
| `post_processor_configs.process_with_gemini_merge()` | `openrouter_client.call_openrouter()` | HTTP API call | `(str, str, int)` — (system_prompt, user_input, timeout) | `str | None` | Returns `None` on any failure |
| `voice_input.VoiceInputDaemon._post_process()` | `post_processor_configs.process_with_vertex_ai()` | Direct function call | `(str, dict, str)` — (text, config, glossary_ctx) | `str` | Returns original text on any failure (never raises) |
| `voice_input.VoiceInputDaemon._post_process()` | `post_processor_configs.process_with_gemini_merge()` | Direct function call | `(str, str|None, dict, str)` — (primary, secondary, config, glossary_ctx) | `str` | Returns primary_text on any failure (never raises) |
| `voice_input._log()` | File I/O | `open(NOTIFY_LOG_FILE, "a")` | `(str, str)` — (tag, message) | `None` | Silently catches all exceptions |

---

## 3. Data Models

### VertexProxyInput (stdin JSON schema)

```python
from typing import TypedDict, NotRequired

class VertexProxyInput(TypedDict):
    """JSON schema for vertex_proxy.py stdin.

    Serialized via json.dumps() in post_processor_configs.py,
    deserialized via json.loads() in vertex_proxy.py main().
    """
    system_prompt: str              # System instruction for Gemini
    user_input: str                 # User text to process (required, non-empty)
    model: str                      # Gemini model name (default "gemini-2.5-flash")
    region: str                     # Vertex AI region (default "global")
    max_output_tokens: NotRequired[int]  # US-001: max output tokens (default 512)
```

**Validation rules:**
- `user_input` must be non-empty (vertex_proxy.py exits 1 if missing/empty)
- `max_output_tokens` is optional; absent → defaults to 512 in vertex_proxy.py
- `max_output_tokens` must be a positive integer when present; Gemini SDK validates type
- All string fields are UTF-8 (Chinese text + English; `ensure_ascii=False` in json.dumps)

**Invariant:** `max_output_tokens` in the JSON is computed by the caller as `min(8192, max(512, len(user_input)))` — output budget roughly matches input length for an editing task.

### max_output_tokens formula

```python
def compute_max_output_tokens(user_input: str) -> int:
    """Compute dynamic max_output_tokens from user_input length.

    Formula: min(8192, max(512, len(user_input)))

    - Floor 512: backward compat, short text doesn't need more
    - Ceiling 8192: cost cap (prevents runaway generation)
    - Middle: len(user_input) — editing task output ≈ input length

    Examples:
        len=50    → 512   (floor)
        len=512   → 512   (exact match)
        len=2000  → 2000  (pass-through)
        len=4637  → 4637  (real incident case)
        len=8192  → 8192  (exact match)
        len=20000 → 8192  (ceiling)
    """
    return min(8192, max(512, len(user_input)))
```

Note: This is NOT a separate function in the code — the formula is inlined at each call site. Documented here for test writers.

### PostProcessorPreset (existing, unchanged)

```python
class PostProcessorPreset(TypedDict):
    """Structure of POST_PROCESSOR_PRESETS dict values."""
    name: str                       # Human-readable name
    description: str                # One-line description
    framework: str                  # "regex" | "ssh-claude" | "vertex-ai" | "vertex-ai-merge" | "llama-cpp"
    config: NotRequired[dict[str, str | int]]  # Framework-specific config
```

### VertexAI Config (existing, relevant subset)

```python
class VertexAIConfig(TypedDict):
    """Config dict for vertex-ai and vertex-ai-merge frameworks."""
    ssh_host: str                   # SSH host alias (e.g. "oracle-cloud")
    proxy_script: str               # Remote script path (e.g. "~/vertex_proxy.py")
    model: str                      # Gemini model name
    fallback_model: str             # Fallback model for 429 retry
    vertex_region: str              # Vertex AI region
    timeout: int                    # Subprocess timeout in seconds
    min_text_len: int               # Minimum text length to trigger SSH call
    vocab_min_count: int            # Minimum vocab variant count threshold
    system_prompt_file: NotRequired[str]   # Path to system prompt file (relative to VOICE_INPUT_DATA_DIR)
    user_prompt_template_file: NotRequired[str]  # Path to user prompt template file
```

---

## 4. Error Taxonomy

| Module | Error Condition | Exit/Return | When Raised | Recovery |
|--------|----------------|-------------|-------------|----------|
| vertex_proxy.py | `max_output_tokens` absent in JSON | Uses default 512 | `data.get("max_output_tokens", 512)` | N/A — backward compatible |
| vertex_proxy.py | `max_output_tokens` is non-integer | Gemini SDK raises TypeError | `GenerateContentConfig(max_output_tokens=...)` | Caught by existing `try/except` → exit 1 → caller falls back |
| vertex_proxy.py | `user_input` empty/missing | exit 1, stderr "Missing 'user_input'" | `main()` validation | Caller treats exit 1 as failure, falls back to original text |
| vertex_proxy.py | Gemini API error (429, 500, etc.) | exit 1, stderr error message | `client.models.generate_content()` | Caller's `_run_vertex_proxy` retries 429, then falls back to lite model |
| vertex_proxy.py | Gemini returns empty (response.text is None) | exit 1, stderr message | After API call | Caller falls back |
| post_processor_configs.py | SSH timeout | Returns original `text` | `subprocess.TimeoutExpired` in `_run_vertex_proxy` | Logged as warning; OpenRouter fallback attempted |
| post_processor_configs.py | vertex_proxy.py exit 1 | Tries OpenRouter fallback | `result.returncode != 0` | `call_openrouter()` attempted; if both fail, returns original text |
| post_processor_configs.py | Hallucination guard triggered | Returns original `text` | `len(output) > len(text) * 2` | Logged as warning |
| post_processor_configs.py | Question guard triggered | Returns original `text` | `'？' in text and '？' not in output and '?' not in output` | Logged as warning |
| voice_input.py | `_log()` file I/O failure | Silently ignored | Any exception in `_log()` | `except Exception: pass` |

---

## 5. Configuration Contract

| Key | Type | Default | Required | Used By | Notes |
|-----|------|---------|----------|---------|-------|
| `max_output_tokens` (JSON field) | `int` | `512` | No | vertex_proxy.py `main()` | US-001: new optional field in stdin JSON |
| `ssh_host` | `str` | — | Yes | `process_with_vertex_ai`, `process_with_gemini_merge` | SSH host alias from config |
| `proxy_script` | `str` | — | Yes | `process_with_vertex_ai`, `process_with_gemini_merge` | Remote script path |
| `model` | `str` | `"gemini-2.5-flash"` | No | Both process functions, vertex_proxy.py | Gemini model name |
| `fallback_model` | `str` | `None` | No | `_run_vertex_proxy` | Model for 429 retry fallback |
| `vertex_region` | `str` | `"global"` | No | Both process functions, vertex_proxy.py | Vertex AI region |
| `timeout` | `int` | `15` | No | Both process functions | Subprocess timeout (seconds) |
| `min_text_len` | `int` | `15` | No | Both process functions | Min chars to trigger SSH call |
| `vocab_min_count` | `int` | `3` | No | `_post_process` in voice_input.py | Min vocab variant count for apply_vocab |
| `system_prompt_file` | `str` | — | No | Both process functions | System prompt file path (relative to VOICE_INPUT_DATA_DIR) |
| `user_prompt_template_file` | `str` | — | No | `process_with_vertex_ai` | User prompt template file path |

---

## 6. Test Contracts

### US-001: vertex_proxy.py accepts max_output_tokens

**Unit test targets** (test vertex_proxy.py in isolation):

| Test Case | Input JSON | Expected Behavior |
|-----------|-----------|-------------------|
| `max_output_tokens` absent | `{"system_prompt": "...", "user_input": "hello", "model": "...", "region": "..."}` | `data.get("max_output_tokens", 512)` returns 512; `GenerateContentConfig` receives `max_output_tokens=512` |
| `max_output_tokens` = 2048 | `{..., "max_output_tokens": 2048}` | `GenerateContentConfig` receives `max_output_tokens=2048` |
| `max_output_tokens` = 512 (explicit) | `{..., "max_output_tokens": 512}` | Identical to absent case |
| `max_output_tokens` = 8192 | `{..., "max_output_tokens": 8192}` | `GenerateContentConfig` receives `max_output_tokens=8192` |
| `--help` output | `python3 vertex_proxy.py --help` | Stdout contains "max_output_tokens" |

**How to test without Gemini API:** Mock `client.models.generate_content` and capture the `config` argument passed to it. Verify `config.max_output_tokens` matches the input JSON value.

### US-002: post_processor_configs.py computes and passes max_output_tokens

**Unit test targets** (test formula computation + JSON construction):

| Test Case | user_input length | Expected max_output_tokens in JSON |
|-----------|------------------|------------------------------------|
| Short text (50 chars) | 50 | 512 (floor) |
| Medium text (1000 chars) | 1000 | 1000 |
| Real incident case (~4637 chars) | 4637 | 4637 |
| Exact floor boundary (512 chars) | 512 | 512 |
| Exact ceiling boundary (8192 chars) | 8192 | 8192 |
| Very long text (20000 chars) | 20000 | 8192 (ceiling) |

**How to test:** Mock `subprocess.run` (the SSH call) and capture `stdin_data` argument. Parse as JSON and verify `max_output_tokens` field.

**Test for `process_with_vertex_ai`:**
1. Construct `text` of known length, mock SSH subprocess.
2. Call `process_with_vertex_ai(text, config, "")`.
3. Assert `json.loads(captured_stdin)["max_output_tokens"] == min(8192, max(512, len(user_input)))`.
4. Note: `user_input` may differ from `text` if `user_prompt_template_file` wraps it. The formula uses `len(user_input)` (after template formatting), not `len(text)`.

**Test for `process_with_gemini_merge`:**
1. Construct `primary_text` and `secondary_text` of known lengths.
2. `user_input = f"Chinese ASR: {primary_text}\nEnglish ASR: {secondary_text}"`.
3. Expected: `max_output_tokens = min(8192, max(512, len(user_input)))`.
4. Verify via captured stdin JSON.

**Fallback preservation test:**
1. Make first SSH call return exit 1 with "429" in stderr.
2. Make retry also return exit 1 with "429".
3. Verify fallback model stdin JSON still contains original `max_output_tokens` value.
4. (Tests `_run_vertex_proxy` line 320-322: `payload["model"]` is replaced but `payload["max_output_tokens"]` is NOT.)

### US-003: voice_input.py remove log truncation

**Unit test targets** (verify `[:120]` removal):

| Test Case | Input | Expected Log Content |
|-----------|-------|---------------------|
| PP input log (line 1054) | text = "A" * 200 | `_log` called with full 200-char string, not truncated to 120 |
| PP output log (line 1128) | result = "B" * 300 | `_log` called with full 300-char string |
| ASR-2 secondary log (line 1219) | secondary = "C" * 250 | `_log` called with full 250-char string |
| ASR raw log (line 1225) | raw_primary = "D" * 500 | `_log` called with full 500-char string |
| Short text (< 120 chars) | text = "short" | Behavior unchanged (no truncation was happening) |

**How to test:** Mock `_log` function, trigger the code path, verify the message argument contains the full text.

**Negative test (out of scope):**
- Line 1066 (PUNC log) should still have `[:120]`. Verify it is NOT changed.

---

## 7. Deployment Contract

### vertex_proxy.py (Oracle Cloud)

After modifying `vertex_proxy.py` locally:

1. `scp vertex_proxy.py oracle-cloud:~/vertex_proxy.py`
2. Verify: `ssh oracle-cloud "python3 ~/vertex_proxy.py --help"` — output must mention `max_output_tokens`
3. Verify: `echo '{"system_prompt":"test","user_input":"hello","max_output_tokens":1024}' | ssh oracle-cloud "python3 ~/vertex_proxy.py"` — should succeed (exit 0)

### post_processor_configs.py (Local)

No deployment step — file is already symlinked from install dir to source. Daemon imports it directly. Daemon restart not strictly required (module is re-imported per call via lazy import in `_post_process`), but recommended to clear any cached state.

### voice_input.py (Local)

No deployment step — file is already symlinked. Change takes effect on next `_log()` call (file append). No daemon restart needed for log format changes.
