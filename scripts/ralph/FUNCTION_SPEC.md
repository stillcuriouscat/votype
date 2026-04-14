# Function Specification (CLEAN-ROOM DOCUMENT)
> This document is shared between the Build agent and the Unit-Test agent.
> The Build agent implements these specs. The Unit-Test agent writes tests against them.
> Neither agent should read the other's output.

Generated: 2026-04-14 15:35 CST
PRD: prd.json — ralph/gemini-output-truncation-fix, 3 stories, fix hardcoded max_output_tokens=512 and log truncation
Architecture: HIGH_LEVEL_DESIGN.md, LOW_LEVEL_DESIGN.md

---

## US-001: vertex_proxy.py — accept max_output_tokens from JSON stdin

### `vertex_proxy.main() -> None`

**Purpose**: Read JSON from stdin, call Gemini API, write result to stdout. Now reads optional `max_output_tokens` from stdin JSON.

**Preconditions**:
- stdin contains valid JSON with at least `user_input` field (non-empty string)
- google-genai SDK is importable
- ADC credentials are configured on the host

**Postconditions**:
- Gemini `GenerateContentConfig` is constructed with `max_output_tokens` from stdin JSON (or 512 if absent)
- stdout contains the corrected text (stripped), exit 0
- On any failure: stderr has error message, exit 1

**Behavior Table**:

| # | Scenario | Input JSON | Expected Output | Side Effects |
|---|----------|-----------|----------------|-------------|
| 1 | Normal: max_output_tokens absent | `{"system_prompt": "fix", "user_input": "hello", "model": "gemini-2.5-flash", "region": "global"}` | `GenerateContentConfig` receives `max_output_tokens=512` | Gemini API called with 512 token limit |
| 2 | Normal: max_output_tokens=2048 | `{"system_prompt": "fix", "user_input": "hello", "model": "gemini-2.5-flash", "region": "global", "max_output_tokens": 2048}` | `GenerateContentConfig` receives `max_output_tokens=2048` | Gemini API called with 2048 token limit |
| 3 | Normal: max_output_tokens=512 (explicit) | `{..., "max_output_tokens": 512}` | `GenerateContentConfig` receives `max_output_tokens=512` | Identical to absent case |
| 4 | Normal: max_output_tokens=8192 | `{..., "max_output_tokens": 8192}` | `GenerateContentConfig` receives `max_output_tokens=8192` | Gemini API called with 8192 token limit |
| 5 | Edge: max_output_tokens=1 | `{..., "max_output_tokens": 1}` | `GenerateContentConfig` receives `max_output_tokens=1` | Value passed through; Gemini SDK may truncate output |
| 6 | Error: max_output_tokens is a string | `{..., "max_output_tokens": "abc"}` | exit 1, stderr contains error message | Gemini SDK raises TypeError, caught by existing try/except |

**Data Flow**: stdin JSON → `json.loads()` → `data.get("max_output_tokens", 512)` → stored in local `max_output_tokens` variable → passed to `GenerateContentConfig(max_output_tokens=max_output_tokens)` → Gemini API call → stdout

**Implementation Detail**: The variable is read at line ~124 (after `region = data.get("region", "global")`), and used at line ~150 (replacing the hardcoded `512`).

```python
# Line ~124 (after existing data.get calls):
max_output_tokens = data.get("max_output_tokens", 512)

# Line ~150 (in GenerateContentConfig):
max_output_tokens=max_output_tokens,   # was: max_output_tokens=512,
```

**Performance**: No change. One additional `dict.get()` call — O(1).

---

### `vertex_proxy.print_help() -> None`

**Purpose**: Print usage information including the new `max_output_tokens` field.

**Preconditions**: None.

**Postconditions**:
- stdout contains help text
- Help text includes a line documenting `max_output_tokens`

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|----------------|-------------|
| 1 | Normal: --help flag | `python3 vertex_proxy.py --help` | stdout contains the string `"max_output_tokens"` | exit 0 |
| 2 | Normal: help text format | `python3 vertex_proxy.py --help` | stdout contains `"max_output_tokens  Max output tokens (default: 512, optional)"` | exit 0 |
| 3 | Edge: existing fields still present | `python3 vertex_proxy.py --help` | stdout still contains `"system_prompt"`, `"user_input"`, `"model"`, `"region"` | exit 0 |

**Data Flow**: `--help` flag → `print_help()` → formatted string to stdout → `sys.exit(0)`

**Implementation Detail**: Add one line after the `"region"` entry in the help text:

```python
"  max_output_tokens  Max output tokens (default: 512, optional)\n"
```

**Performance**: N/A (print only).

---

## US-002: post_processor_configs.py — dynamically compute and pass max_output_tokens

### `post_processor_configs.process_with_vertex_ai(text: str, config: dict, glossary_ctx: str = "") -> str`

**Purpose**: Call Vertex AI Gemini via SSH proxy for text polishing. Now computes and includes `max_output_tokens` in the stdin JSON payload.

**Preconditions**:
- `text` is a non-empty string with `len(text) >= config.get("min_text_len", 15)`
- `config` dict contains `ssh_host`, `proxy_script` keys at minimum

**Postconditions**:
- The JSON sent to vertex_proxy.py via stdin includes `"max_output_tokens"` key
- The value is `min(8192, max(512, len(user_input)))` where `user_input` is the **formatted** user text (after template application)
- Return value is the polished text, or original `text` on any failure

**Behavior Table**:

| # | Scenario | text length | user_input after template | Expected max_output_tokens in JSON | Return Value |
|---|----------|-------------|--------------------------|-----------------------------------|--------------|
| 1 | Normal: short text (50 chars) | 50 | ~65 chars (with template) | 512 (floor applies) | polished text from Gemini |
| 2 | Normal: medium text (1000 chars) | 1000 | ~1015 chars (with template) | 1015 | polished text from Gemini |
| 3 | Normal: real incident case | ~2250 | ~4637 chars (merge-like) | 4637 | polished text from Gemini |
| 4 | Edge: exact floor boundary | varies | 512 chars exactly | 512 | polished text from Gemini |
| 5 | Edge: exact ceiling boundary | varies | 8192 chars exactly | 8192 | polished text from Gemini |
| 6 | Edge: very long text | varies | 20000 chars | 8192 (ceiling applies) | polished text from Gemini |
| 7 | Edge: text at min_text_len boundary (15 chars) | 15 | ~30 chars (with template) | 512 (floor applies) | polished text from Gemini |
| 8 | Error: SSH timeout | any | any | value computed but never received | original `text` |
| 9 | Error: vertex_proxy exit 1 + OpenRouter fails | any | any | value computed but proxy failed | original `text` |

**Data Flow**: `text` → template formatting → `user_input` → `len(user_input)` → `min(8192, max(512, len(user_input)))` → `max_output_tokens` → included in `json.dumps({..., "max_output_tokens": max_output_tokens})` → SSH stdin → vertex_proxy.py

**Implementation Detail**: The `max_output_tokens` variable is computed from `user_input` (the already-formatted string, NOT from raw `text`). It is added to the `json.dumps()` dict at line ~393-398.

```python
# Compute dynamic max_output_tokens from user_input length
max_output_tokens = min(8192, max(512, len(user_input)))
stdin_data = json.dumps({
    "system_prompt": system_prompt,
    "user_input": user_input,
    "model": config.get("model", "gemini-2.5-flash"),
    "region": config.get("vertex_region", "global"),
    "max_output_tokens": max_output_tokens,
}, ensure_ascii=False)
```

**Performance**: One `len()` + two `min()`/`max()` calls — O(1). No latency impact.

---

### `post_processor_configs.process_with_gemini_merge(primary_text: str, secondary_text: str | None, config: dict, glossary_ctx: str = "") -> str`

**Purpose**: Merge two ASR transcriptions via Vertex AI Gemini. Now computes and includes `max_output_tokens` in the stdin JSON payload.

**Preconditions**:
- `primary_text` is a non-empty string with `len(primary_text) >= config.get("min_text_len", 15)`
- `secondary_text` is a string or None
- `config` dict contains `ssh_host`, `proxy_script` keys at minimum

**Postconditions**:
- The JSON sent to vertex_proxy.py via stdin includes `"max_output_tokens"` key
- The value is `min(8192, max(512, len(user_input)))` where `user_input` is the **formatted** dual/single ASR string
- Return value is the merged text, or `primary_text` on any failure

**Behavior Table**:

| # | Scenario | primary_text | secondary_text | user_input format | Expected max_output_tokens | Return Value |
|---|----------|-------------|----------------|-------------------|---------------------------|--------------|
| 1 | Normal: dual ASR, short | "hello" * 10 (50 chars) | "world" * 10 (50 chars) | `"Chinese ASR: {p}\nEnglish ASR: {s}"` → ~115 chars | 512 (floor) | merged text |
| 2 | Normal: dual ASR, real incident | ~2250 chars | ~2373 chars | `"Chinese ASR: {p}\nEnglish ASR: {s}"` → ~4637 chars | 4637 | merged text |
| 3 | Normal: secondary=None (single) | ~2250 chars | None | `"Chinese ASR: {p}"` → ~2264 chars | 2264 | polished text |
| 4 | Edge: very long dual | 5000 chars | 5000 chars | ~10029 chars | 8192 (ceiling) | merged text |
| 5 | Edge: exact floor | varies | varies | 512 chars exactly | 512 | merged text |
| 6 | Edge: exact ceiling | varies | varies | 8192 chars exactly | 8192 | merged text |
| 7 | Error: SSH timeout | any | any | any | value computed but proxy failed | `primary_text` |
| 8 | Error: hallucination guard triggered | 100 chars | 100 chars | ~230 chars | 512 | `primary_text` (guard returns original) |

**Data Flow**: `primary_text` + `secondary_text` → format as `"Chinese ASR: {p}\nEnglish ASR: {s}"` (or `"Chinese ASR: {p}"` if secondary=None) → `user_input` → `len(user_input)` → `min(8192, max(512, len(user_input)))` → `max_output_tokens` → included in stdin JSON

**Implementation Detail**: Same pattern as `process_with_vertex_ai`. The `max_output_tokens` variable is computed from `user_input` after the dual/single format string is constructed.

```python
# Compute dynamic max_output_tokens from user_input length
max_output_tokens = min(8192, max(512, len(user_input)))
stdin_data = json.dumps({
    "system_prompt": system_prompt,
    "user_input": user_input,
    "model": config.get("model", "gemini-2.5-flash"),
    "region": config.get("vertex_region", "global"),
    "max_output_tokens": max_output_tokens,
}, ensure_ascii=False)
```

**Performance**: O(1). No latency impact.

---

### `post_processor_configs._run_vertex_proxy(cmd: list[str], stdin_data: str, timeout: int, max_retries: int = 1, fallback_model: str | None = None) -> subprocess.CompletedProcess`

**Purpose**: Run vertex_proxy.py via subprocess with retry on 429. **No code changes needed** — this spec documents that `max_output_tokens` is preserved through retry/fallback.

**Preconditions**:
- `stdin_data` is a valid JSON string (may or may not contain `max_output_tokens`)
- `cmd` is a valid subprocess command list

**Postconditions**:
- On 429 fallback: only `payload["model"]` is replaced; `payload["max_output_tokens"]` (if present) is preserved
- Returns the `CompletedProcess` from the last attempt

**Behavior Table**:

| # | Scenario | stdin_data | Expected Behavior | Side Effects |
|---|----------|-----------|-------------------|-------------|
| 1 | Normal: success on first try | `{..., "max_output_tokens": 4637}` | Returns result with rc=0 | One subprocess call |
| 2 | Normal: 429 retry succeeds | `{..., "max_output_tokens": 4637}` | Retries with same stdin_data (max_output_tokens=4637 preserved) | Two subprocess calls, 2s sleep between |
| 3 | Normal: 429 retry + fallback | `{..., "model": "gemini-2.5-flash", "max_output_tokens": 4637}` | Fallback replaces `model` only; `max_output_tokens` stays 4637 in fallback stdin | Three subprocess calls |
| 4 | Edge: stdin_data without max_output_tokens | `{"system_prompt": "...", "user_input": "..."}` | Works as before — no max_output_tokens to preserve | Backward compatible |
| 5 | Error: timeout | any | raises `subprocess.TimeoutExpired` | Not retried |

**Data Flow**: `stdin_data` → subprocess → on 429 → `json.loads(stdin_data)` → `payload["model"] = fallback_model` → `json.dumps(payload)` → subprocess (max_output_tokens untouched)

**Performance**: No change. Existing retry behavior preserved.

---

## US-003: voice_input.py — remove log [:120] truncation

### `voice_input.VoiceInputDaemon._post_process(self, text: str) -> str`

**Purpose**: Apply post-processing to transcribed text. Log call sites at entry and exit now log full text without `[:120]` truncation.

**Preconditions**:
- `self.current_post_processor_id` is set
- `text` is a string (may be empty)

**Postconditions**:
- Line ~1054: `_log("PP", f"input ({self.current_post_processor_id}): {text}")` logs full `text` — no `[:120]`
- Line ~1128: `_log("PP", f"output ({elapsed:.2f}s): {result}")` logs full `result` — no `[:120]`
- Return value unchanged (post-processed text)

**Behavior Table**:

| # | Scenario | text | Expected _log call at line 1054 | Expected _log call at line 1128 |
|---|----------|------|---------------------------------|---------------------------------|
| 1 | Normal: short text (50 chars) | `"A" * 50` | `_log("PP", "input (gemini-merge): " + "A"*50)` | `_log("PP", "output (X.XXs): " + result)` — full result |
| 2 | Normal: long text (200 chars) | `"A" * 200` | `_log("PP", "input (gemini-merge): " + "A"*200)` — all 200 chars | `_log("PP", "output (X.XXs): " + result)` — full result |
| 3 | Normal: very long text (2000 chars) | `"A" * 2000` | `_log("PP", "input (gemini-merge): " + "A"*2000)` — all 2000 chars | full result logged |
| 4 | Edge: text exactly 120 chars | `"A" * 120` | `_log("PP", "input (...): " + "A"*120)` — same as before (no truncation was happening) | full result |
| 5 | Edge: empty text | `""` | `_log("PP", "input (...): ")` | Returns early; output log may or may not be reached |
| 6 | Error: _log fails (file I/O) | any | Silently caught by `_log`'s `except Exception: pass` | No effect on return value |

**Data Flow**: `text` → `_log("PP", f"input (...): {text}")` (no `[:120]`) → ... processing ... → `result` → `_log("PP", f"output (...): {result}")` (no `[:120]`)

**Implementation Detail**:
```python
# Line 1054 — BEFORE:
_log("PP", f"input ({self.current_post_processor_id}): {text[:120]}")
# Line 1054 — AFTER:
_log("PP", f"input ({self.current_post_processor_id}): {text}")

# Line 1128 — BEFORE:
_log("PP", f"output ({elapsed:.2f}s): {result[:120]}")
# Line 1128 — AFTER:
_log("PP", f"output ({elapsed:.2f}s): {result}")
```

**Performance**: Log file grows slightly larger for long texts. No runtime performance impact.

---

### `voice_input.VoiceInputDaemon._handle_transcribe(self, msg: dict) -> None` (log sites only)

**Purpose**: Handle transcription requests. Two log call sites now log full text without `[:120]` truncation.

**Preconditions**:
- `self._last_secondary_text` is set (string or None) after secondary ASR completes
- `response["text"]` (raw_primary) is a non-empty string

**Postconditions**:
- Line ~1219: `_log("ASR-2", f"secondary: {self._last_secondary_text}")` logs full secondary text — no `[:120]`
- Line ~1225: `_log("ASR", f"raw: {raw_primary}")` logs full raw primary text — no `[:120]`

**Behavior Table**:

| # | Scenario | secondary / raw_primary | Expected _log call |
|---|----------|-------------------------|-------------------|
| 1 | Normal: secondary 250 chars | `self._last_secondary_text = "C" * 250` | `_log("ASR-2", "secondary: " + "C"*250)` — all 250 chars |
| 2 | Normal: raw_primary 500 chars | `raw_primary = "D" * 500` | `_log("ASR", "raw: " + "D"*500)` — all 500 chars |
| 3 | Normal: real incident (2373 chars secondary) | `self._last_secondary_text` = 2373-char string | `_log("ASR-2", "secondary: " + full_2373_chars)` |
| 4 | Normal: real incident (2250 chars primary) | `raw_primary` = 2250-char string | `_log("ASR", "raw: " + full_2250_chars)` |
| 5 | Edge: text exactly 120 chars | 120-char string | Identical to before (no truncation was happening at this length) |
| 6 | Edge: text < 120 chars | 50-char string | Identical to before |

**Implementation Detail**:
```python
# Line 1219 — BEFORE:
_log("ASR-2", f"secondary: {self._last_secondary_text[:120]}")
# Line 1219 — AFTER:
_log("ASR-2", f"secondary: {self._last_secondary_text}")

# Line 1225 — BEFORE:
_log("ASR", f"raw: {raw_primary[:120]}")
# Line 1225 — AFTER:
_log("ASR", f"raw: {raw_primary}")
```

**Performance**: Log file grows slightly larger. No runtime performance impact.

---

### Out of Scope: Line 1066 (PUNC log)

Line 1066 (`_log("PUNC", f"applied punctuation: {result[:120]}")`) is **NOT** modified by this PRD. Tests should verify this line **still has** `[:120]` truncation.

---

## Helper: max_output_tokens formula

This is **NOT** a separate function in the codebase — the formula is inlined at each call site in `process_with_vertex_ai()` and `process_with_gemini_merge()`. Documented here for test writers.

### `compute_max_output_tokens(user_input: str) -> int` (conceptual)

**Purpose**: Compute dynamic max_output_tokens budget from user_input character length.

**Formula**: `min(8192, max(512, len(user_input)))`

**Behavior Table**:

| # | Scenario | len(user_input) | Expected Result | Rationale |
|---|----------|----------------|-----------------|-----------|
| 1 | Normal: medium text | 2000 | 2000 | Pass-through: output budget = input length |
| 2 | Normal: real incident | 4637 | 4637 | Exact incident case from CSV evidence |
| 3 | Edge: floor boundary exact | 512 | 512 | Floor matches exactly |
| 4 | Edge: below floor | 50 | 512 | Floor applied |
| 5 | Edge: below floor (1 char) | 1 | 512 | Floor applied |
| 6 | Edge: ceiling boundary exact | 8192 | 8192 | Ceiling matches exactly |
| 7 | Edge: above ceiling | 20000 | 8192 | Ceiling applied |
| 8 | Edge: just above floor | 513 | 513 | Pass-through |
| 9 | Edge: just below ceiling | 8191 | 8191 | Pass-through |
| 10 | Edge: empty string | 0 | 512 | Floor applied (though this path won't be reached — empty text returns early) |

**Performance**: O(1).

---

## Cross-Module Contract: stdin JSON Schema

The JSON sent from `post_processor_configs.py` to `vertex_proxy.py` via SSH stdin has this schema after all changes:

```python
{
    "system_prompt": str,        # System instruction for Gemini (required)
    "user_input": str,           # User text to process (required, non-empty)
    "model": str,                # Gemini model name (default "gemini-2.5-flash")
    "region": str,               # Vertex AI region (default "global")
    "max_output_tokens": int,    # NEW: max output tokens (computed by caller)
}
```

**Contract**:
- Sender (`post_processor_configs.py`): MUST include `max_output_tokens` computed as `min(8192, max(512, len(user_input)))`
- Receiver (`vertex_proxy.py`): MUST read via `data.get("max_output_tokens", 512)` — defaults to 512 when absent for backward compatibility
- Intermediary (`_run_vertex_proxy` fallback): MUST NOT modify `max_output_tokens` when replacing `model` for fallback

---

## Existing Functions — No Changes (for reference)

The following functions are NOT modified by this PRD. Listed for completeness so the Unit-Test agent knows they are out of scope:

- `post_processor_configs.process_with_ssh_claude()` — uses SSH+Claude, not vertex_proxy.py
- `post_processor_configs.load_vocab()` — vocab I/O
- `post_processor_configs.apply_vocab()` — regex replacement
- `post_processor_configs.glossary_context()` — prompt context generation
- `post_processor_configs.diff_to_vocab()` — vocab accumulation
- `post_processor_configs.save_vocab()` — vocab persistence
- `post_processor_configs.PostProcessorLoader` — model loading
- `post_processor_configs.PostProcessorInference` — regex/LLM inference
- `openrouter_client.call_openrouter()` — OpenRouter fallback (does NOT receive max_output_tokens; uses its own defaults)
- `voice_input._log()` — the function itself is unchanged; only its **call sites** are modified
- `voice_input._log_csv()` — CSV training data logging, unchanged
