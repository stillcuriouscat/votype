# Function Specification (CLEAN-ROOM DOCUMENT)
> This document is shared between the Build agent and the Unit-Test agent.
> The Build agent implements these specs. The Unit-Test agent writes tests against them.
> Neither agent should read the other's output.

Generated: 2026-05-18 12:30 CST
PRD: prd.json — `ralph/anthropic-fallback`, 5 stories, add Claude Haiku 4.5 as primary post-processor via SSH→Oracle, mirroring the existing `vertex_proxy.py` pattern.
Architecture: HIGH_LEVEL_DESIGN.md, LOW_LEVEL_DESIGN.md

---

## Conventions

- All paths are absolute when referring to files on disk in tests; tests use `tmp_path` for any DB or file fixture (see HIGH_LEVEL_DESIGN risk #12).
- Error strings shown in **single quotes** are the exact substring the test must assert on (`assert 'xxx' in stderr`). Strings shown in **double quotes** in `raises ExactError("...")` cells are the exact full message.
- `Optional[X]` means `X | None`. Any function flagged "Never raises" must catch every exception internally and return the fallback shown.
- The codebase uses Python 3.10+; positional-or-keyword parameters with defaults follow existing convention (see `process_with_vertex_ai` — `glossary_ctx` defaults to `""`).

---

## Module A — `anthropic_proxy.py` (NEW)

Path: `/home/dev/code/voice_input/anthropic_proxy.py`. Deployed to `oracle-cloud:~/anthropic_proxy.py`.

Module-level constants:

```python
ANTHROPIC_KEY_PATH: Path = Path("~/.config/claude.secret").expanduser()
DEFAULT_MODEL:      str  = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS: int  = 1024
```

---

### `anthropic_proxy._trace(msg: str) -> None`

**Purpose**: Emit a trace line to stderr in `[TRACE] {msg}` format (mirrors `vertex_proxy._trace`).

**Preconditions**:
- `msg` is a string.

**Postconditions**:
- Exactly one line `'[TRACE] ' + msg + '\n'` is written to `sys.stderr`.
- No write to stdout.
- Returns `None`.

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal | `_trace("sdk_init: 0.12s")` | returns `None` | stderr += `'[TRACE] sdk_init: 0.12s\n'` |
| 2 | Edge: empty string | `_trace("")` | returns `None` | stderr += `'[TRACE] \n'` |
| 3 | Edge: contains newline | `_trace("a\nb")` | returns `None` | stderr += `'[TRACE] a\nb\n'` (single print call) |

**Data Flow**: `msg` → `print(f"[TRACE] {msg}", file=sys.stderr)`.

**Performance**: O(len(msg)).

---

### `anthropic_proxy._import_anthropic() -> tuple[object, object]`

**Purpose**: Lazy import the `anthropic` SDK so `--help` works without the SDK installed.

**Preconditions**:
- None.

**Postconditions**:
- Returns `(anthropic_module, Anthropic_class)` where `Anthropic_class is anthropic_module.Anthropic`.

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: SDK installed | (call) | returns `(<module 'anthropic'>, <class 'anthropic.Anthropic'>)` | none |
| 2 | Edge: import called repeatedly | call twice | both return equal tuples (same class object) | none |
| 3 | Error: SDK missing | (call with anthropic not installed) | raises `ImportError("No module named 'anthropic'")` (exact message from Python) | none |

**Data Flow**: `from anthropic import Anthropic` → return `(anthropic, Anthropic)`.

**Performance**: First call O(SDK import cost ≈ 0.3–1.0s); subsequent calls O(1) due to `sys.modules` cache.

---

### `anthropic_proxy._read_api_key() -> str`

**Purpose**: Read the Anthropic API key from `ANTHROPIC_KEY_PATH`.

**Preconditions**:
- None (function probes the filesystem).

**Postconditions**:
- On success: returns `key.strip()`, guaranteed non-empty.

**Behavior Table**:

| # | Scenario | Input (filesystem state) | Expected Output | Side Effects |
|---|----------|--------------------------|-----------------|--------------|
| 1 | Normal | `~/.config/claude.secret` contains `'sk-ant-xxxx\n'` | returns `'sk-ant-xxxx'` | none |
| 2 | Edge: leading/trailing whitespace | file contains `'  sk-ant-yyy  \n\n'` | returns `'sk-ant-yyy'` | none |
| 3 | Error: file missing | file does not exist | raises `FileNotFoundError` (stdlib message) | none |
| 4 | Error: unreadable | file mode 000 | raises `PermissionError` (stdlib message) | none |
| 5 | Error: empty | file contains `''` or `'   \n'` | raises `ValueError("API key file is empty: " + str(ANTHROPIC_KEY_PATH))` | none |

**Data Flow**: `ANTHROPIC_KEY_PATH.read_text(encoding="utf-8")` → `.strip()` → guard for emptiness → return.

**Performance**: O(file size); file is a single line.

---

### `anthropic_proxy.print_help() -> None`

**Purpose**: Print CLI usage information to stdout.

**Preconditions**:
- None.

**Postconditions**:
- Returns `None`.
- Caller (`main()`) is responsible for `sys.exit(0)` after this returns.
- stdout contains all of the following substrings (one print, multi-line OK):
  - `'anthropic_proxy.py'`
  - `'--help'`
  - `'--test'`
  - `'system_prompt'`
  - `'user_input'`
  - `'model'`
  - `'max_tokens'`
  - `'Stdout'`
  - `'Exit 0'`

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal | (call) | stdout text contains all 9 substrings above | none |
| 2 | Edge: stdout closed | (call after `sys.stdout.close()`) | raises `ValueError` from print | none (not a normal path; tested only as documentation) |

**Data Flow**: single `print(...)` to stdout.

**Performance**: O(1).

---

### `anthropic_proxy.run_test() -> None`

**Purpose**: Verify (a) `anthropic` SDK is importable and (b) `ANTHROPIC_KEY_PATH` is readable & non-empty. **Calls `sys.exit` itself** — does not return on success or failure.

**Preconditions**:
- None.

**Postconditions**:
- On full success: prints `'OK: SDK import + API key file readable.'` to stdout, calls `sys.exit(0)`.
- On any failure: prints `'FAIL: ' + <reason>` to stderr, calls `sys.exit(1)`.
- Does NOT make a network call to Anthropic.

**Behavior Table**:

| # | Scenario | Input (env) | Expected Output | Side Effects |
|---|----------|-------------|-----------------|--------------|
| 1 | Normal | SDK installed, key file present & non-empty | stdout = `'OK: SDK import + API key file readable.\n'`; calls `sys.exit(0)` | none |
| 2 | Error: SDK missing | `import anthropic` raises ImportError | stderr contains `'FAIL: '` and `'anthropic'`; calls `sys.exit(1)` | none |
| 3 | Error: key file missing | SDK ok; key file absent | stderr contains `'FAIL: '` and the path or `'No such file'`; calls `sys.exit(1)` | none |
| 4 | Error: key file empty | SDK ok; key file empty | stderr contains `'FAIL: '` and `'empty'`; calls `sys.exit(1)` | none |

**Data Flow**: try `_import_anthropic()` → try `_read_api_key()` → success branch prints OK + exit(0); except branches print FAIL + exit(1).

**Performance**: O(SDK import cost + file read).

---

### `anthropic_proxy.main() -> None`

**Purpose**: CLI entry point. Parses argv, reads JSON from stdin, calls Anthropic, writes text to stdout. Always exits via `sys.exit`; never returns normally on the JSON-processing path.

**Preconditions**:
- None.

**Postconditions**:
- Always calls `sys.exit(0 | 1)`. Stdout / stderr per the table below.

**Argv handling** (parsed BEFORE reading stdin):

| argv[1] | Behavior |
|---------|----------|
| absent | continue to stdin-processing |
| `--help` | call `print_help()`, `sys.exit(0)` |
| `-h` | call `print_help()`, `sys.exit(0)` |
| `--test` | call `run_test()` (which exits) |
| anything else | stderr = `'Unknown argument: ' + argv[1]`, `sys.exit(1)` |

**Stdin JSON schema** (parsed via single `json.loads(sys.stdin.read())`):

```json
{
  "system_prompt": "<str, default ''>",
  "user_input":    "<str, REQUIRED, non-empty>",
  "model":         "<str, default 'claude-haiku-4-5-20251001'>",
  "max_tokens":    "<int, default 1024>"
}
```

Defaults applied via `data.get(key, default)`. No validation of `model` (anything passed to SDK). No validation of `max_tokens` range (passed straight through to SDK; out-of-range values will surface as an Anthropic API error and exit 1 by the API-error path).

**Anthropic call shape** (must match exactly):

```python
client = anthropic_module.Anthropic(api_key=_read_api_key())
response = client.messages.create(
    model=model,
    max_tokens=max_tokens,
    system=system_prompt,
    messages=[{"role": "user", "content": user_input}],
)
text = response.content[0].text
```

**Behavior Table**:

| # | Scenario | Stdin / argv | Expected stdout | Expected stderr | Exit code |
|---|----------|--------------|-----------------|-----------------|-----------|
| 1 | Normal: success | argv=[], stdin=`{"user_input":"你好世界abc"}`; SDK returns `Message(content=[TextBlock(text="你好，世界 abc")])` | `'你好，世界 abc\n'` | `'[TRACE] sdk_init: '`, `'[TRACE] anthropic_api: '` lines | 0 |
| 2 | Normal: all fields | argv=[], stdin=`{"system_prompt":"sys","user_input":"u","model":"claude-haiku-4-5-20251001","max_tokens":2048}` | model output stripped | `[TRACE]` lines | 0 |
| 3 | Normal: extra fields ignored | stdin=`{"user_input":"x","unknown":"y"}` | model output | `[TRACE]` lines | 0 |
| 4 | Edge: stdout has surrounding whitespace | SDK returns `'  hi  \n'` | `'hi\n'` (stripped before print) | `[TRACE]` lines | 0 |
| 5 | Edge: `--help` | argv=`["--help"]`, stdin=anything | help text containing required substrings | empty | 0 |
| 6 | Edge: `-h` | argv=`["-h"]` | help text | empty | 0 |
| 7 | Edge: `--test` (ok) | argv=`["--test"]` | `'OK: SDK import + API key file readable.\n'` | empty | 0 |
| 8 | Error: unknown argv | argv=`["--foo"]` | empty | `'Unknown argument: --foo\n'` | 1 |
| 9 | Error: invalid JSON | argv=[], stdin=`'not json'` | empty | starts with `'Invalid JSON input: '` | 1 |
| 10 | Error: missing user_input | stdin=`{}` | empty | `"Missing 'user_input' in JSON\n"` | 1 |
| 11 | Error: empty user_input | stdin=`{"user_input":""}` | empty | `"Missing 'user_input' in JSON\n"` | 1 |
| 12 | Error: SDK import fails | (anthropic not installed) | empty | starts with `'anthropic SDK not installed: '` | 1 |
| 13 | Error: API key file missing | stdin valid; key file absent | empty | starts with `'API key error: '` | 1 |
| 14 | Error: API key file empty | stdin valid; key file `''` | empty | starts with `'API key error: '` and contains `'empty'` | 1 |
| 15 | Error: Anthropic 5xx | SDK `client.messages.create` raises `RuntimeError("503 Service Unavailable")` | empty | starts with `'Anthropic API error: '` and contains `'503'` | 1 |
| 16 | Error: Anthropic 429 | SDK raises an exception whose `str(e)` contains `'rate_limit_error'` | empty | starts with `'Anthropic API error: '` and contains `'rate_limit_error'` | 1 |
| 17 | Error: empty content | SDK returns `Message(content=[])` | empty | `'Anthropic returned empty response\n'` | 1 |
| 18 | Error: content[0].text is empty | SDK returns `Message(content=[TextBlock(text="")])` | empty | `'Anthropic returned empty response\n'` | 1 |
| 19 | Error: content[0].text is None | SDK returns `Message(content=[TextBlock(text=None)])` | empty | `'Anthropic returned empty response\n'` | 1 |

**Exact error message contracts** (Unit-Test agent will assert on these):

- argv parse:          `f"Unknown argument: {arg}"`
- json parse:          `f"Invalid JSON input: {e}"`  (e is the `json.JSONDecodeError`)
- missing user_input:  `"Missing 'user_input' in JSON"`
- import failure:      `f"anthropic SDK not installed: {e}"`
- key file failure:    `f"API key error: {e}"`
- API failure:         `f"Anthropic API error: {e}"`
- empty response:      `"Anthropic returned empty response"`

**Trace lines emitted to stderr on the success path** (assertable):
- After `_import_anthropic()` + `Anthropic(api_key=...)`: `_trace(f"sdk_init: {elapsed:.2f}s")`
- After `client.messages.create(...)`: `_trace(f"anthropic_api: {elapsed:.2f}s")`

**Data Flow**:
```
argv → flag dispatch
     ↘ stdin → json.loads → defaults → validate user_input
                       → _import_anthropic → Anthropic(api_key=_read_api_key())
                       → trace(sdk_init)
                       → client.messages.create(...)
                       → trace(anthropic_api)
                       → response.content[0].text.strip() → print to stdout
                       → sys.exit(0)
```

**Performance**: O(network round-trip), expected 500–2000 ms.

**Invariants** (testable):
- No imports from the voice_input project. A test can `ast.parse` the file and assert that every `Import` / `ImportFrom` node's module is in `{"json","sys","time","warnings","argparse","pathlib","anthropic"}` (the union of stdlib + `anthropic`); zero references to `voice_input`, `post_processor_*`, `state_db`, `openrouter_client`.
- No filesystem writes. A test running the script under a write-blocking mock confirms only `Path.read_text` is called on `ANTHROPIC_KEY_PATH` (no `open(..., "w")`, no `Path.write_*`).
- Single-shot: control flow never loops on input; one read of stdin, one API call.

---

## Module B — `post_processor_configs.py` (MODIFY)

Two new public functions. Existing functions (`process_with_ssh_claude`, `process_with_vertex_ai`, `process_with_gemini_merge`, `_run_vertex_proxy`, `apply_vocab`, `glossary_context`, `load_vocab`, `save_vocab`, `diff_to_vocab`, classes `PostProcessorLoader`, `PostProcessorInference`) are unchanged.

---

### `post_processor_configs.process_with_anthropic(text: str, config: dict, glossary_ctx: str = "") -> str`

**Purpose**: Call Claude Haiku via SSH→Oracle proxy for single-text polishing. Mirror of `process_with_vertex_ai` with `anthropic_proxy.py` as the remote script and `max_tokens` as the SDK key.

**Preconditions**:
- `text` is a `str` (may be empty).
- `config` is a `dict` with at minimum keys `"ssh_host"` and `"proxy_script"`.
- `glossary_ctx` is a `str` (may be empty).

**Postconditions**:
- Returns a `str`.
- **Never raises.** Any exception from subprocess, OpenRouter, or guard logic is caught and the original `text` is returned (the empty-text early return returns `""`).
- `config` and `glossary_ctx` are not mutated.

**Config keys consumed**:

| Key | Type | Default | Purpose |
|---|---|---|---|
| `ssh_host` | str | (required) | SSH target |
| `proxy_script` | str | (required) | Remote script path (e.g. `~/anthropic_proxy.py`) |
| `model` | str | `"claude-haiku-4-5-20251001"` | Anthropic model |
| `timeout` | int | `15` | SSH subprocess timeout (seconds) |
| `min_text_len` | int | `15` | Short-circuit threshold |
| `system_prompt_file` | str | absent → use `system_prompt` | path relative to `VOICE_INPUT_DATA_DIR` |
| `system_prompt` | str | `""` | inline fallback |
| `user_prompt_template_file` | str | absent → use `user_prompt_template` | path relative to `VOICE_INPUT_DATA_DIR` |
| `user_prompt_template` | str | absent → use raw `text` | format with `{text}` |

**SSH command** (exact argv list, no shell interpolation):

```python
cmd = [
    "ssh", "-o", "ConnectTimeout=5",
    config["ssh_host"],
    "python3", config["proxy_script"],
]
```

**JSON stdin payload** (sent via `_run_vertex_proxy(cmd, stdin_data, timeout, fallback_model=None)`):

```python
user_input = user_prompt_template.format(text=text) if user_prompt_template else text
max_tokens = min(8192, max(512, len(user_input)))
stdin_data = json.dumps({
    "system_prompt": system_prompt + ("\n\n" + glossary_ctx if glossary_ctx else ""),
    "user_input":    user_input,
    "model":         config.get("model", "claude-haiku-4-5-20251001"),
    "max_tokens":    max_tokens,
}, ensure_ascii=False)
```

**Note**: the JSON key is `max_tokens` (NOT `max_output_tokens` — that key is Gemini-specific).

**Fallback chain**:
1. `_run_vertex_proxy(cmd, stdin_data, timeout, fallback_model=None)`. (The 429 retry branch is effectively a no-op for Anthropic because Anthropic stderr does not contain the substrings `'429'` / `'RESOURCE_EXHAUSTED'`. The retry MUST NOT be removed or specialized; reuse the existing function as-is.)
2. If returncode != 0 OR `subprocess.TimeoutExpired` raised: try `openrouter_client.call_openrouter(system_prompt, user_input, timeout=timeout)`.
3. If OpenRouter returns `None`: call `voice_input.notify("Votype", "Anthropic + OpenRouter both failed", urgency="low")` and return `text`.

**Guards** (applied AFTER a successful output is obtained from either Anthropic or OpenRouter):
- Hallucination guard: `if len(output) > len(text) * 2: return text`
- Question guard:      `if '？' in text and '？' not in output and '?' not in output: return text`

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: success | `text="你好呃这是测试文本"`, valid config | `'你好这是测试文本'` (stdout from proxy, stripped) | subprocess.run called once; `[SSH] anthropic-fix success: N→M chars` logged at INFO |
| 2 | Edge: empty text | `text=""` | `""` | NO subprocess.run call; returns immediately |
| 3 | Edge: text shorter than min_text_len | `text="abc"`, default `min_text_len=15` | `"abc"` (returns input) | NO subprocess.run call; INFO log `"Text length 3 below min_text_len 15, skipping SSH"` |
| 4 | Edge: glossary_ctx appended | valid config, `glossary_ctx="Commonly used terms: A, B"` | proxy output | stdin JSON `system_prompt` ends with `"\n\nCommonly used terms: A, B"` |
| 5 | Edge: long text → max_tokens cap | `text="x" * 20000` | proxy output | stdin JSON contains `"max_tokens": 8192` |
| 6 | Edge: short text → max_tokens floor | `text` short enough that `len(user_input) < 512` but `>= min_text_len` | proxy output | stdin JSON contains `"max_tokens": 512` |
| 7 | Edge: medium text → linear max_tokens | `len(user_input) == 1500` | proxy output | stdin JSON contains `"max_tokens": 1500` |
| 8 | Failure: subprocess rc != 0, OpenRouter succeeds | proxy exits 1; OR returns `'recovered text'` | `'recovered text'` | INFO log `'[OPENROUTER] fallback success for anthropic-fix: N chars'` |
| 9 | Failure: subprocess rc != 0, OpenRouter returns None | proxy exits 1; OR returns None | `text` (unchanged) | `notify("Votype", "Anthropic + OpenRouter both failed", urgency="low")` called once |
| 10 | Failure: subprocess.TimeoutExpired, OpenRouter succeeds | proxy timeout; OR returns `'fast'` | `'fast'` | WARNING log `'Anthropic timed out after 15s'` (or equivalent); fallback succeeds |
| 11 | Failure: timeout + OpenRouter None | both fail | `text` | notify with `'Anthropic + OpenRouter both failed'` |
| 12 | Guard: hallucination | input `text` len=20; proxy returns 50-char output | `text` (unchanged) | WARNING log `'LLM output too long'` |
| 13 | Guard: question mark dropped | `text` contains `'？'`; proxy output contains neither `'？'` nor `'?'` | `text` | WARNING log `'LLM dropped question marks'` |
| 14 | Guard: question mark preserved as ASCII `'?'` | `text` has `'？'`; output has `'?'` | output (passes guard) | normal log |
| 15 | Robustness: missing model key | config lacks `"model"` | proxy called with default `"claude-haiku-4-5-20251001"` | normal |
| 16 | Robustness: prompt file load failure | `system_prompt_file` points to nonexistent path | raises propagates → **NEVER**: function must NOT raise; catch and return `text` | WARNING log; returns input |

**Note on row 16**: If the prompt file read raises (FileNotFoundError / OSError / UnicodeDecodeError), the function must catch and return `text` to preserve the "Never raises" contract. The Build agent should wrap the prompt-loading section in a try/except that logs at WARNING level and returns `text`. This mirrors the de-facto behavior of `process_with_vertex_ai` (where a missing file would also bubble up — but for the new functions we make the catch explicit because the spec contract says "Never raises").

**Data Flow**:
```
text, config, glossary_ctx
  → empty-text / short-text guards
  → load system_prompt (file or inline) + append glossary_ctx
  → load user_prompt_template (file or inline)
  → user_input = template.format(text=text) or text
  → max_tokens = min(8192, max(512, len(user_input)))
  → stdin_data = json.dumps({system_prompt, user_input, model, max_tokens}, ensure_ascii=False)
  → cmd = [ssh, -o, ConnectTimeout=5, ssh_host, python3, proxy_script]
  → _run_vertex_proxy(cmd, stdin_data, timeout, fallback_model=None)
      rc==0 → output = result.stdout.strip()
      else  → fallthrough to OpenRouter
      TimeoutExpired → catch → fallthrough to OpenRouter
  → if not output: call_openrouter(system_prompt, user_input, timeout)
      None → notify + return text
      str  → output = result
  → hallucination guard → return text if tripped
  → question guard → return text if tripped
  → INFO log success; return output
```

**Performance**: O(network round-trip). Default timeout 15 s. p50 ≈ 1.5 s, p95 ≈ 3 s.

**Log lines** (Unit-Test agent will assert at least one of these per code path with `caplog`):
- `'[PROMPT] loaded system prompt: <path>'` — INFO, if `system_prompt_file` used
- `'[PROMPT] loaded user template: <path>'` — INFO, if `user_prompt_template_file` used
- `'[OPENROUTER] fallback success for anthropic-fix: <N> chars'` — INFO, fallback wins
- `'[SSH] anthropic-fix success: <N>→<M> chars'` — INFO, primary path wins
- `'LLM output too long (... possible hallucination ...)'` — WARNING, hallucination guard
- `'LLM dropped question marks'` — WARNING, question guard

---

### `post_processor_configs.process_with_anthropic_merge(primary_text: str, secondary_text: Optional[str], config: dict, glossary_ctx: str = "") -> str`

**Purpose**: Merge dual-ASR transcriptions (SenseVoice primary + faster-whisper secondary) via Claude Haiku → polished text. Mirror of `process_with_gemini_merge` with Anthropic transport.

**Preconditions**:
- `primary_text` is a `str` (may be empty).
- `secondary_text` is `Optional[str]`.
- `config` is a `dict` with at minimum `"ssh_host"` and `"proxy_script"`.
- `glossary_ctx` is a `str`.

**Postconditions**:
- Returns a `str`.
- **Never raises.**
- `config` is not mutated.

**Special early-return rules** (must occur BEFORE any SSH call, identical semantics to `process_with_gemini_merge`):

```python
min_text_len = config.get("min_text_len", 15)
if not primary_text:                              # empty primary
    if secondary_text and len(secondary_text) >= 1:
        return secondary_text
    return ""
if len(primary_text) < min_text_len:              # short primary
    if secondary_text and len(secondary_text) > len(primary_text):
        return secondary_text
    return primary_text
```

**User-input construction**:

```python
if secondary_text is not None:
    user_input = f"Chinese ASR: {primary_text}\nEnglish ASR: {secondary_text}"
else:
    user_input = f"Chinese ASR: {primary_text}"
```

**JSON stdin payload, SSH cmd, max_tokens formula, fallback chain, guards**:
Identical to `process_with_anthropic`, except:
- Model default: `"claude-haiku-4-5-20251001"` (same).
- There is NO `user_prompt_template*` config key consumed (the user_input is built in code above).
- Hallucination/question guards compare against `primary_text` (not `text`).
- "Both failed" notify text: `"Anthropic merge + OpenRouter both failed"`.

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: dual ASR merge | `primary="今天天气很好"`, `secondary="It's sunny today"`, valid config; proxy returns `'今天天气很好。It is sunny today.'` | `'今天天气很好。It is sunny today.'` | stdin JSON `user_input == "Chinese ASR: 今天天气很好\nEnglish ASR: It's sunny today"`; INFO log `'[SSH] anthropic-merge success: primary=N, secondary=M → K chars'` |
| 2 | Normal: secondary is None | `primary="hello world这是测试文本"`, `secondary=None` | proxy output | stdin JSON `user_input == "Chinese ASR: hello world这是测试文本"`; INFO `'[MERGE] primary=N chars, secondary=None'` |
| 3 | Edge: empty primary, has secondary | `primary=""`, `secondary="fallback text"` | `'fallback text'` | NO subprocess call; INFO log `'Primary empty, falling back to secondary (13 chars)'` (or similar) |
| 4 | Edge: empty primary, no secondary | `primary=""`, `secondary=None` | `""` | NO subprocess call |
| 5 | Edge: empty primary, empty secondary | `primary=""`, `secondary=""` | `""` | NO subprocess call (secondary falsy → second branch) |
| 6 | Edge: short primary, secondary longer | `primary="hi"` (len 2), `secondary="this is longer"` | `'this is longer'` | NO subprocess call |
| 7 | Edge: short primary, secondary shorter or None | `primary="hi"`, `secondary=None` | `'hi'` | NO subprocess call |
| 8 | Edge: short primary, secondary equal-or-shorter | `primary="hi"`, `secondary="x"` | `'hi'` | NO subprocess call |
| 9 | Edge: max_tokens cap | `primary` such that `len(user_input) > 8192` | proxy output | stdin JSON `max_tokens == 8192` |
| 10 | Edge: max_tokens floor | `primary` such that `len(user_input) < 512` but `len(primary) >= min_text_len` | proxy output | stdin JSON `max_tokens == 512` |
| 11 | Edge: glossary_ctx appended to system | valid input, `glossary_ctx="terms: A,B"` | proxy output | system_prompt sent ends with `"\n\nterms: A,B"` |
| 12 | Failure: subprocess rc != 0, OpenRouter succeeds | proxy exits 1; OR returns `'recovered'` | `'recovered'` | INFO log `'[OPENROUTER] fallback success for anthropic-merge: N chars'` |
| 13 | Failure: rc != 0, OpenRouter None | both fail | `primary_text` | `notify("Votype", "Anthropic merge + OpenRouter both failed", urgency="low")` called once |
| 14 | Failure: TimeoutExpired, OpenRouter succeeds | proxy timeout; OR succeeds | OR output | WARNING log `'Anthropic merge timed out after 15s'` (or equivalent) |
| 15 | Failure: TimeoutExpired, OpenRouter None | both fail | `primary_text` | notify with `'Anthropic merge + OpenRouter both failed'` |
| 16 | Guard: hallucination | proxy returns text > 2 × len(primary_text) | `primary_text` | WARNING log `'LLM output too long'` |
| 17 | Guard: question dropped | `'？' in primary_text` but neither `'？'` nor `'?'` in output | `primary_text` | WARNING log `'LLM dropped question marks'` |
| 18 | Guard: question preserved as ASCII '?' | `'？' in primary`, `'?'` in output | output | passes guard |
| 19 | Robustness: missing model key | config lacks `"model"` | proxy called with default model | normal |
| 20 | Robustness: prompt file load failure | `system_prompt_file` nonexistent | returns `primary_text` (Never raises) | WARNING log |

**Data Flow**:
```
primary_text, secondary_text, config, glossary_ctx
  → empty/short-primary guards (may early-return secondary or primary)
  → load system_prompt (file or inline) + append glossary_ctx
  → user_input = merge-format (with-secondary or single-line)
  → max_tokens = min(8192, max(512, len(user_input)))
  → stdin_data = json.dumps({system_prompt, user_input, model, max_tokens}, ensure_ascii=False)
  → cmd = [ssh, -o, ConnectTimeout=5, ssh_host, python3, proxy_script]
  → _run_vertex_proxy(cmd, stdin_data, timeout, fallback_model=None)
      rc==0 → output = result.stdout.strip()
      else  → fallthrough
      TimeoutExpired → catch → fallthrough
  → if not output: call_openrouter(system_prompt, user_input, timeout)
      None → notify + return primary_text
      str  → output = result
  → hallucination guard vs primary_text → return primary_text if tripped
  → question guard vs primary_text → return primary_text if tripped
  → INFO log success; return output
```

**Performance**: same as `process_with_anthropic`.

**Log lines** (assertable):
- `'[MERGE] primary=<N> chars, secondary=<M|None> chars'` — INFO
- `'[PROMPT] loaded system prompt: <path>'` — INFO (if file loader used)
- `'[SSH] anthropic-merge success: primary=<N>, secondary=<M|None> → <K> chars'` — INFO
- `'[OPENROUTER] fallback success for anthropic-merge: <K> chars'` — INFO
- `'Primary empty, falling back to secondary (<N> chars)'` — INFO, empty-primary path
- `'Primary too short (<N> chars), falling back to secondary (<M> chars)'` — INFO, short-primary path
- `'LLM output too long'` — WARNING
- `'LLM dropped question marks'` — WARNING

---

## Module C — `post_processor_presets.py` (MODIFY)

No function added. Only data structure mutation.

### `POST_PROCESSOR_PRESETS` (dict mutation contract)

**Purpose**: Add `claude-fix` and `claude-merge` entries; do not alter any existing key.

**Postconditions** (testable invariants):

1. `POST_PROCESSOR_PRESETS["claude-fix"]` is exactly:
   ```python
   {
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
   ```

2. `POST_PROCESSOR_PRESETS["claude-merge"]` is exactly:
   ```python
   {
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
       },
   }
   ```
   Note: NO `user_prompt_template_file` / `user_prompt_template` / `fallback_model` keys in `claude-merge.config`.

3. `DEFAULT_POST_PROCESSOR == "claude-merge"`.

4. **All eight existing presets are bit-for-bit identical** to the pre-change state. Specifically (deep-equal comparison must hold):
   - `"none"` (regex framework, no config block — present with just `name`/`description`/`framework`).
   - `"chinese-text-correction"`, `"qwen3-0.6b"`, `"minicpm4-0.5b"` (llama-cpp).
   - `"haiku-fix"` (ssh-claude, with `claude_path`, `model`, `system_prompt_file="prompts/haiku-fix-system.txt"`, `user_prompt_template_file="prompts/haiku-fix-user.txt"`, `timeout=15`, `min_text_len=15`, `vocab_min_count=3`).
   - `"haiku-expand"` (ssh-claude with empty `config: {}`).
   - `"gemini-fix"` (vertex-ai, with `proxy_script="~/vertex_proxy.py"`, `fallback_model="gemini-2.5-flash-lite"`, etc. — unchanged).
   - `"gemini-merge"` (vertex-ai-merge, with `proxy_script="~/vertex_proxy.py"`, `fallback_model="gemini-2.5-flash-lite"`, `system_prompt_file="prompts/gemini-merge-system.txt"` — unchanged).

**Behavior Table** (for the imported module):

| # | Scenario | Check | Expected |
|---|----------|-------|----------|
| 1 | Normal: import | `from post_processor_presets import POST_PROCESSOR_PRESETS, DEFAULT_POST_PROCESSOR` | both bindings exist |
| 2 | New key present | `"claude-fix" in POST_PROCESSOR_PRESETS` | `True` |
| 3 | New key present | `"claude-merge" in POST_PROCESSOR_PRESETS` | `True` |
| 4 | Default updated | `DEFAULT_POST_PROCESSOR` | `"claude-merge"` |
| 5 | Default points to a real preset | `DEFAULT_POST_PROCESSOR in POST_PROCESSOR_PRESETS` | `True` |
| 6 | Default's framework | `POST_PROCESSOR_PRESETS[DEFAULT_POST_PROCESSOR]["framework"]` | `"anthropic-merge"` |
| 7 | claude-fix framework | `POST_PROCESSOR_PRESETS["claude-fix"]["framework"]` | `"anthropic"` |
| 8 | claude-merge has no user template | `"user_prompt_template_file" in POST_PROCESSOR_PRESETS["claude-merge"]["config"]` | `False` |
| 9 | claude-fix prompt files reused | `POST_PROCESSOR_PRESETS["claude-fix"]["config"]["system_prompt_file"]` | `"prompts/gemini-fix-system.txt"` |
| 10 | All eight legacy presets present | `{"none","chinese-text-correction","qwen3-0.6b","minicpm4-0.5b","haiku-fix","haiku-expand","gemini-fix","gemini-merge"} <= set(POST_PROCESSOR_PRESETS)` | `True` |
| 11 | gemini-merge unchanged | `POST_PROCESSOR_PRESETS["gemini-merge"]["framework"]` | `"vertex-ai-merge"` (unchanged) |
| 12 | Total preset count | `len(POST_PROCESSOR_PRESETS)` | `10` (8 existing + 2 new) |
| 13 | Module exports | `VOICE_INPUT_DATA_DIR` and `MODELS_DIR` still importable | both `Path` instances |

**Data Flow**: static — module-level dict literal.

**Performance**: O(1) at import.

---

## Module D — `voice_input.py` (MODIFY)

Only two methods on `ASRDaemon` change. No public signature change.

---

### `ASRDaemon.load_post_processor(self, preset_id: Optional[str] = None) -> None`

**Purpose**: Load (or reload) a post-processor preset; load/unload the secondary ASR model based on whether the new framework is a merge framework. Updates persistent state. **Critical contract for US-004**: switching between two merge frameworks (e.g. `vertex-ai-merge` ↔ `anthropic-merge`) MUST NOT cause a `_unload_secondary_model()` → `_load_secondary_model()` cycle; the secondary model must remain loaded across the switch.

**Preconditions**:
- `self` is a fully-initialized `ASRDaemon` with `current_post_processor_id` set.
- `preset_id` is either `None` (reload current) or a key in `POST_PROCESSOR_PRESETS`.

**Postconditions** (on the success path):
- `self.current_post_processor_id == preset_id` (or its previous value if `preset_id is None`).
- `self.post_processor_framework == POST_PROCESSOR_PRESETS[preset_id]["framework"]`.
- `update_state(STATE_DB_PATH, post_processor=preset_id)` has been called once.
- If `self.post_processor_framework ∈ {"ssh-claude","vertex-ai","vertex-ai-merge","anthropic","anthropic-merge"}`: `self._vocab` is the dict returned by `load_vocab()` (possibly `{}`).
- If `self.post_processor_framework ∈ {"vertex-ai-merge","anthropic-merge"}` (merge set):
  - `self._secondary_model` is non-`None` after the call.
  - If `self._secondary_model` was already non-`None` before the call (idempotency contract): `self._load_secondary_model` is NOT called a second time. **Equivalent test assertion**: after sequentially calling `load_post_processor("gemini-merge")` then `load_post_processor("claude-merge")`, `_load_secondary_model.call_count == 1` (load called once, on the first invocation only).
- If `self.post_processor_framework ∉ merge set`: `self._unload_secondary_model()` is called.
- `self._unload_secondary_model()` is NOT called when switching from one merge framework to another.

**Side-effect order** (test-observable via mocks; this is the contract the Build agent must satisfy):
1. `PostProcessorLoader.load_post_processor(preset_id)` → `self.post_processor_model`
2. `self.current_post_processor_id = preset_id`
3. `update_state(STATE_DB_PATH, post_processor=preset_id)`
4. `self.post_processor_framework = preset["framework"]`
5. If framework ∈ vocab-set: `self._vocab = load_vocab()`
6. If framework ∈ merge-set AND `self._secondary_model is None`: `self._load_secondary_model()`
   Else if framework ∉ merge-set: `self._unload_secondary_model()`
   Else (merge-set AND already loaded): no-op.

**Vocab set**: `{"ssh-claude", "vertex-ai", "vertex-ai-merge", "anthropic", "anthropic-merge"}`.
**Merge set**: `{"vertex-ai-merge", "anthropic-merge"}`.

**Failure recovery contract** (existing behavior, must be preserved):
- If any step 1–6 raises `Exception` (other than `RuntimeError` / `ValueError` listed below), it is caught internally:
  - `self.post_processor_model = None`
  - `self.current_post_processor_id = "none"`
  - `self.post_processor_framework = "regex"`
  - `self._unload_secondary_model()` is called (best-effort).

**Behavior Table**:

| # | Scenario | Preconditions (mocked) | Input | Side effects observable on `self` and mocks |
|---|----------|------------------------|-------|---------------------------------------------|
| 1 | Normal: load `claude-fix` (fix framework) | `self._secondary_model = None` | `load_post_processor("claude-fix")` | `current_post_processor_id == "claude-fix"`, `post_processor_framework == "anthropic"`, `_vocab` is a `dict` (loaded), `_load_secondary_model` NOT called, `_unload_secondary_model` called once |
| 2 | Normal: load `claude-merge` from cold | `self._secondary_model = None`, `current_post_processor_id = "none"` | `load_post_processor("claude-merge")` | `current_post_processor_id == "claude-merge"`, `post_processor_framework == "anthropic-merge"`, `_load_secondary_model` called once, `_unload_secondary_model` NOT called |
| 3 | **US-004 critical**: switch `gemini-merge` → `claude-merge` (warm) | After step 1 of: `load_post_processor("gemini-merge")`; then call again | sequential calls | `_load_secondary_model` total call count == `1` (NOT 2); `_unload_secondary_model` NOT called between the two; `current_post_processor_id == "claude-merge"`, `post_processor_framework == "anthropic-merge"` |
| 4 | **US-004 critical**: switch `claude-merge` → `gemini-merge` (warm) | Same idempotency expectation, opposite direction | sequential | `_load_secondary_model` total call count == `1`; `current_post_processor_id == "gemini-merge"` |
| 5 | Switch merge → fix unloads secondary | After `claude-merge` warm | `load_post_processor("claude-fix")` | `_unload_secondary_model` called once; `_load_secondary_model` not called this turn; `post_processor_framework == "anthropic"` |
| 6 | Switch fix → merge loads secondary | `self._secondary_model is None` | `load_post_processor("claude-merge")` | `_load_secondary_model` called once |
| 7 | Switch `vertex-ai` → `anthropic` (both fix) | warm | sequential | neither `_load_secondary_model` nor `_unload_secondary_model` causes a secondary load; `_unload_secondary_model` called once on the second call (no-op since model already absent) |
| 8 | Reload current preset | `preset_id=None`, `current_post_processor_id="claude-merge"` | `load_post_processor(None)` | resolves to `"claude-merge"`; same side-effects as steps 4–6 of the original load |
| 9 | Vocab loaded for anthropic frameworks | `load_vocab` mocked to return `{"foo": {...}}` | `load_post_processor("claude-fix")` | `self._vocab == {"foo": {...}}` |
| 10 | Vocab loaded for anthropic-merge | `load_vocab` mocked | `load_post_processor("claude-merge")` | `self._vocab == <mock return>` |
| 11 | Error: unknown preset | `preset_id="nonexistent"` | call | raises `RuntimeError("Unknown post-processor: nonexistent")` (existing behavior, propagated) |
| 12 | Error: haiku-expand placeholder | `preset_id="haiku-expand"` | call | calls `notify(...)` then raises `ValueError("Haiku Expand is not yet implemented")` (existing behavior, propagated) |
| 13 | Failure recovery: model loader raises | `PostProcessorLoader.load_post_processor` mock raises `RuntimeError("boom")` | `load_post_processor("claude-merge")` | function returns normally (no exception escapes); `current_post_processor_id == "none"`, `post_processor_framework == "regex"`, `post_processor_model is None`, `_unload_secondary_model` called |
| 14 | `update_state` called with new id | normal | `load_post_processor("claude-merge")` | `update_state` called with `post_processor="claude-merge"` exactly once (on the success path, BEFORE the framework field is reassigned) |

**Data Flow**:
```
preset_id (or current_post_processor_id if None)
  → validate in POST_PROCESSOR_PRESETS (raise RuntimeError if missing)
  → check haiku-expand placeholder (raise ValueError if so)
  → try:
        post_processor_model = PostProcessorLoader.load_post_processor(preset_id)
        current_post_processor_id = preset_id
        update_state(STATE_DB_PATH, post_processor=preset_id)
        post_processor_framework = preset["framework"]
        if framework in {ssh-claude, vertex-ai, vertex-ai-merge, anthropic, anthropic-merge}:
            self._vocab = load_vocab()
        if framework in {vertex-ai-merge, anthropic-merge}:
            if self._secondary_model is None:
                self._load_secondary_model()
            # else: idempotent no-op, do NOT unload-then-reload
        else:
            self._unload_secondary_model()
    except Exception:
        fallback to none/regex (see Failure recovery contract above)
```

**Performance**: O(model load) on cold path; O(1) on idempotent warm switches between merge frameworks.

---

### `ASRDaemon._post_process(self, text: str) -> str`

**Purpose**: Apply post-processing to transcribed text. Adds dispatch for `anthropic` and `anthropic-merge` frameworks alongside the existing `ssh-claude`, `vertex-ai`, `vertex-ai-merge`.

**Preconditions**:
- `self.current_post_processor_id` is a key in `POST_PROCESSOR_PRESETS`.
- `self.post_processor_framework` is the framework string for that preset.
- `text` is a `str`.

**Postconditions**:
- Returns a `str`.
- On any internal failure inside the SSH-dispatch block (process_with_* never raises per Module B), the function returns the pre-LLM `result` (same fall-through behavior as today).
- Calls `update_state(status="polishing")` exactly once, BEFORE the LLM call, when the framework is in the SSH-framework set.

**SSH-framework set (gate)** — extended to:
```python
{"ssh-claude", "vertex-ai", "vertex-ai-merge", "anthropic", "anthropic-merge"}
```

**Merge-dispatch set** — extended to:
```python
{"vertex-ai-merge", "anthropic-merge"}
```

**Fix-dispatch dict** — extended to:
```python
{
    "ssh-claude": process_with_ssh_claude,
    "vertex-ai":  process_with_vertex_ai,
    "anthropic":  process_with_anthropic,
}
```

**Merge-dispatch dict**:
```python
{
    "vertex-ai-merge": process_with_gemini_merge,
    "anthropic-merge": process_with_anthropic_merge,
}
```

**Behavior Table**:

| # | Scenario | `current_post_processor_id` / framework | Input | Expected behavior (mocked process_with_*) |
|---|----------|------------------------------------------|-------|--------------------------------------------|
| 1 | Normal: claude-fix dispatch | `claude-fix` / `anthropic` | `text="呃这是测试文本输入A"` (≥ min_text_len) | `process_with_anthropic(result, config, glossary_ctx)` is called exactly once; `process_with_anthropic_merge` NOT called; returns the mock's return value |
| 2 | Normal: claude-merge dispatch | `claude-merge` / `anthropic-merge` | `text="这是测试文本输入"` | `process_with_anthropic_merge(result, secondary, config, glossary_ctx)` called with `secondary = self._last_secondary_text`; `process_with_anthropic` NOT called |
| 3 | Existing path unchanged: vertex-ai-merge | `gemini-merge` / `vertex-ai-merge` | `text="..."` | `process_with_gemini_merge` called; anthropic functions NOT called |
| 4 | Existing path unchanged: vertex-ai | `gemini-fix` / `vertex-ai` | `text="..."` | `process_with_vertex_ai` called; anthropic functions NOT called |
| 5 | Existing path unchanged: ssh-claude | `haiku-fix` / `ssh-claude` | `text="..."` | `process_with_ssh_claude` called |
| 6 | Polishing status set | `claude-fix` / `anthropic` | normal | `update_state(status="polishing")` called once BEFORE `process_with_anthropic` |
| 7 | apply_vocab + glossary still applied | `claude-merge` / `anthropic-merge`, `self._vocab` non-empty | normal | `apply_vocab(result, self._vocab, min_count)` called once before the merge dispatch; `glossary_context(self._vocab)` called once |
| 8 | Vocab diff accumulation | `claude-fix`, mock returns text different from input | normal | `diff_to_vocab(before_polish, result, self._vocab)` called; `save_vocab(self._vocab)` called once |
| 9 | Vocab not saved if LLM is a no-op | `claude-fix`, mock returns input unchanged | normal | `diff_to_vocab` / `save_vocab` NOT called |
| 10 | Empty input | any framework | `text=""` | returns `""`; no SSH-dispatch block entered |
| 11 | Regex-only framework | `none` / `regex` | `text="呃测试"` | returns filler-removed text; anthropic functions NOT called; `update_state(status="polishing")` NOT called |
| 12 | llama-cpp framework | e.g. `qwen3-0.6b` / `llama-cpp` | normal | falls through to Step 4 LLM block; anthropic functions NOT called |
| 13 | `process_with_anthropic` returns input on failure | `claude-fix`, mock returns the input unchanged (simulating fallback-to-input) | normal | function returns input; no exception escapes; `save_vocab` NOT called |
| 14 | Secondary text propagation | `claude-merge`, `self._last_secondary_text = "English text"` | normal | the third positional arg to `process_with_anthropic_merge` is `"English text"` |
| 15 | Secondary text None | `claude-merge`, `self._last_secondary_text = None` | normal | `process_with_anthropic_merge` called with `secondary=None` |

**Data Flow** (SSH-dispatch branch only — existing pre-steps preserved):
```
text → remove_fillers → (optional) firered_punc → result
if self.post_processor_framework in SSH-framework set:
    preset = POST_PROCESSOR_PRESETS[self.current_post_processor_id]
    config = preset["config"]
    min_count = config.get("vocab_min_count", 3)
    result = apply_vocab(result, self._vocab, min_count)
    glossary_ctx = glossary_context(self._vocab)
    before_polish = result
    update_state(status="polishing")
    if framework in {vertex-ai-merge, anthropic-merge}:
        secondary = getattr(self, '_last_secondary_text', None)
        merge_fn = {"vertex-ai-merge": process_with_gemini_merge,
                    "anthropic-merge": process_with_anthropic_merge}[framework]
        result = merge_fn(result, secondary, config, glossary_ctx)
    else:
        fix_fn = {"ssh-claude": process_with_ssh_claude,
                  "vertex-ai":  process_with_vertex_ai,
                  "anthropic":  process_with_anthropic}[framework]
        result = fix_fn(result, config, glossary_ctx)
    if before_polish != result:
        self._vocab = diff_to_vocab(before_polish, result, self._vocab)
        save_vocab(self._vocab)
        self._vocab = load_vocab()
# Step 4: llama-cpp path unchanged
return result
```

**Performance**: dominated by Anthropic round-trip via `process_with_anthropic[_merge]`.

**Invariant**:
- The dispatch dicts `{ssh-claude, vertex-ai, anthropic}` and `{vertex-ai-merge, anthropic-merge}` are the **single source of truth** for framework→function routing. No alternate registry. A unit test reading the source AST (or runtime introspection) should find these exact keys.

---

## Module E — `state_db.py` (MODIFY)

Module-level constants change; public function signatures unchanged.

---

### Module-level constants (testable post-edit)

**Postconditions** (assertable by import):

```python
from state_db import _DEPRECATED_PP, _SAFE_DEFAULT, _CREATE_TABLE_SQL

assert _DEPRECATED_PP == {
    "firered-punc": "claude-merge",   # bumped from "gemini-merge"
    "gemini-merge": "claude-merge",   # NEW
}
assert "gemini-fix" not in _DEPRECATED_PP
assert "haiku-fix" not in _DEPRECATED_PP
assert _SAFE_DEFAULT["post_processor"] == "claude-merge"
assert "DEFAULT 'claude-merge'" in _CREATE_TABLE_SQL
assert "DEFAULT 'gemini-merge'" not in _CREATE_TABLE_SQL
```

---

### `state_db.init_db(db_path: Optional[Path] = None) -> None`

**Purpose**: Initialize the SQLite state database (unchanged behavior; only the CREATE TABLE DEFAULT clause has changed).

**Preconditions**:
- `db_path` is `None` (use `DEFAULT_DB_PATH`) or a `Path` whose parent is creatable.

**Postconditions**:
- File at `db_path` exists with table `daemon_state` having a single row `id=1`.
- WAL journal mode is enabled.
- The `post_processor` column DEFAULT in the schema is `'claude-merge'`.
- The default-inserted row has `post_processor == 'claude-merge'` (SQLite applies DEFAULT on INSERT when column is omitted, as in `INSERT OR IGNORE INTO daemon_state (id) VALUES (1)`).
- Legacy file `current_post_processor.txt` (if present) is migrated to DB and unlinked.
- Never raises (errors are logged at WARNING).

**Behavior Table**:

| # | Scenario | Input | Side effects | Post-state |
|---|----------|-------|--------------|------------|
| 1 | Normal: fresh init | `db_path = tmp_path / "state.db"` (does not exist) | file created, table created, row `id=1` inserted | `get_state(db_path)["post_processor"] == "claude-merge"` |
| 2 | Idempotent re-init | call twice on same path | second call is a no-op (INSERT OR IGNORE) | row count == 1; `post_processor` unchanged from first init |
| 3 | Legacy file migration | `db_path.parent / "current_post_processor.txt"` contains `"gemini-fix"` | `update_state(post_processor="gemini-fix")` called; legacy file deleted | `get_state(db_path)["post_processor"] == "gemini-fix"` (gemini-fix is NOT in _DEPRECATED_PP, so it is preserved) |
| 4 | Legacy file with deprecated value | legacy file contains `"gemini-merge"` | `update_state(post_processor="gemini-merge")` called, legacy file deleted, then on first `get_state` the migration triggers | first `get_state` returns `"claude-merge"` and DB is rewritten |
| 5 | Failure: unwritable parent | `db_path = "/proc/foo/state.db"` | logs WARNING "init_db failed: ..."; raises nothing | function returns; subsequent `get_state` returns `_SAFE_DEFAULT` |

**Data Flow**: unchanged from existing implementation; only the embedded SQL DEFAULT string changes.

**Performance**: O(1) sqlite ops.

---

### `state_db.get_state(db_path: Optional[Path] = None) -> dict[str, object]`

**Purpose**: Read singleton row; **auto-migrate** any deprecated `post_processor` value to its replacement, writing back to the DB.

**Preconditions**:
- None (function is self-initializing — if table missing, calls `init_db` and retries).

**Postconditions**:
- Returns a `dict[str, object]` with the six daemon_state columns (`id`, `status`, `daemon_pid`, `recording_pid`, `recording_path`, `post_processor`, `updated_at`).
- If the stored `post_processor` value is a key in `_DEPRECATED_PP`:
  - returned `state["post_processor"]` is the mapped replacement.
  - a synchronous `UPDATE daemon_state SET post_processor=? WHERE id=1` writes back the new value.
  - **A log line is emitted at INFO level** in the format `"[STATE-DB] migrated post_processor: '<old>' → '<new>'"`.
- If the stored value is NOT in `_DEPRECATED_PP`, no UPDATE is issued and no migration log line is emitted.
- Never raises. On error returns a copy of `_SAFE_DEFAULT` and logs at WARNING.

**Migration matrix** (this table is the test contract — Unit-Test agent will iterate over it):

| # | Initial DB value | `get_state()` returns | DB after call | Migration log emitted? |
|---|------------------|------------------------|---------------|------------------------|
| 1 | `"firered-punc"` | `"claude-merge"` | `"claude-merge"` | YES (INFO) |
| 2 | `"gemini-merge"` | `"claude-merge"` | `"claude-merge"` | YES (INFO) |
| 3 | `"gemini-fix"` | `"gemini-fix"` | `"gemini-fix"` (unchanged) | NO |
| 4 | `"haiku-fix"` | `"haiku-fix"` | `"haiku-fix"` (unchanged) | NO |
| 5 | `"none"` | `"none"` | `"none"` (unchanged) | NO |
| 6 | `"claude-merge"` | `"claude-merge"` | `"claude-merge"` (unchanged) | NO (idempotent — no-op) |
| 7 | `"claude-fix"` | `"claude-fix"` | `"claude-fix"` (unchanged) | NO |
| 8 | `"unknown-x"` | `"unknown-x"` | `"unknown-x"` (unchanged) | NO |
| 9 | (table missing → init_db triggers) | `"claude-merge"` (from new DEFAULT) | `"claude-merge"` | NO (fresh row, not migration) |
| 10 | (DB unreadable / sqlite error) | `_SAFE_DEFAULT` copy (post_processor=`"claude-merge"`) | unchanged | NO; WARNING log `"get_state failed: ..."` |

**Behavior Table**:

| # | Scenario | Input (DB state) | Expected returned dict | DB write |
|---|----------|------------------|------------------------|----------|
| 1 | Migration from `gemini-merge` | row with `post_processor="gemini-merge"` | dict with `post_processor="claude-merge"` | UPDATE issued; INFO log emitted |
| 2 | Migration from `firered-punc` | row with `post_processor="firered-punc"` | dict with `post_processor="claude-merge"` | UPDATE issued; INFO log emitted |
| 3 | No migration for explicit choice | row with `post_processor="gemini-fix"` | dict with `post_processor="gemini-fix"` | NO UPDATE; no migration log |
| 4 | Edge: fresh DB | empty table, then `init_db` runs | dict with `post_processor="claude-merge"` (from CREATE TABLE DEFAULT) | INSERT happened in init_db; no migration log |
| 5 | Error: corrupt DB | sqlite3 raises any error | copy of `_SAFE_DEFAULT` (with `post_processor="claude-merge"`) | NO write; WARNING log |
| 6 | Idempotence | call twice on migrated DB | both calls return `"claude-merge"`; only the first emits the migration log | only first emits UPDATE |

**Data Flow**:
```
connect → SELECT * FROM daemon_state WHERE id=1
  if table missing → init_db + reconnect + reselect
  if row is None → return _SAFE_DEFAULT copy
  state = dict(row)
  if state["post_processor"] in _DEPRECATED_PP:
      new = _DEPRECATED_PP[state["post_processor"]]
      log INFO "[STATE-DB] migrated post_processor: '<old>' → '<new>'"
      try UPDATE daemon_state SET post_processor=? WHERE id=1, (new,)
      state["post_processor"] = new
  return state
```

**Performance**: O(1) sqlite queries; one read + possibly one write.

---

### `state_db.update_state(db_path: Optional[Path] = None, **kwargs) -> None`

**Purpose**: UNCHANGED. Listed here for completeness of the interface contract.

**Preconditions**:
- All `kwargs` keys are in `{"status","daemon_pid","recording_pid","recording_path","post_processor","updated_at"}`.

**Postconditions**:
- Specified columns updated atomically; `updated_at` auto-set unless explicitly provided.
- Raises `ValueError` if any key is not in the valid column set (message format: `f"Invalid column: {bad_keys_sorted}. Valid columns: {valid_sorted}"`).

**Behavior Table**:

| # | Scenario | Input | Expected | Side effects |
|---|----------|-------|----------|--------------|
| 1 | Normal: post_processor update | `update_state(db_path, post_processor="claude-fix")` | returns None | row's `post_processor=claude-fix`, `updated_at` updated |
| 2 | Normal: multiple columns | `update_state(db_path, status="idle", daemon_pid=1234)` | returns None | both columns set, `updated_at` updated |
| 3 | Error: invalid column | `update_state(db_path, foobar="x")` | raises `ValueError("Invalid column: foobar. Valid columns: daemon_pid, post_processor, recording_path, recording_pid, status, updated_at")` | no DB write |
| 4 | Edge: no kwargs | `update_state(db_path)` | returns None | no write |

**Data Flow**: unchanged.

**Performance**: O(1).

---

## Module F — `openrouter_client.py` (UNCHANGED — interface restated)

### `openrouter_client.call_openrouter(system_prompt: str, user_input: str, timeout: int = 15) -> Optional[str]`

**Purpose**: UNCHANGED. The two new functions in Module B invoke it identically to the existing Vertex AI helpers.

**Preconditions**:
- `system_prompt`, `user_input` are strings.

**Postconditions**:
- Returns polished text (`str`) on success.
- Returns `None` on any failure (no API key, network error, HTTP non-2xx, JSON parse error, empty content).
- Never raises.

**Behavior Table** (existing, restated for traceability):

| # | Scenario | Input | Output |
|---|----------|-------|--------|
| 1 | Normal | valid system + user + timeout | polished `str` |
| 2 | No API key file | `API_KEY_PATH` missing | `None` |
| 3 | HTTP error | OpenRouter returns 500 | `None` |
| 4 | Timeout | request exceeds `timeout` | `None` |

**Performance**: O(network).

---

## Cross-cutting Contracts (summary)

| Caller | Callee | Failure behavior |
|---|---|---|
| `_post_process` | `process_with_anthropic` / `process_with_anthropic_merge` | Callee NEVER raises → `_post_process` always proceeds |
| `process_with_anthropic{,_merge}` | `_run_vertex_proxy` | `subprocess.TimeoutExpired` propagates; caller catches → OpenRouter fallback |
| `process_with_anthropic{,_merge}` | `_run_vertex_proxy` | non-zero rc → caller falls through to OpenRouter |
| `process_with_anthropic{,_merge}` | `call_openrouter` | `None` → caller calls `notify(...)` and returns input |
| `process_with_anthropic{,_merge}` | `voice_input.notify` (lazy import) | exception swallowed; logged at WARNING (must NOT escape) |
| `anthropic_proxy.main` | Anthropic SDK | exception → stderr error + exit 1 |
| `anthropic_proxy.main` | filesystem (key file) | exception → stderr error + exit 1 |
| `load_post_processor` | `_load_secondary_model` / `_unload_secondary_model` | Both idempotent (existing). New invariant: `_load_secondary_model` skipped when already loaded. |
| `get_state` | `_DEPRECATED_PP` lookup | KeyError impossible (membership tested with `in`); UPDATE failure → caught silently (existing) but migration log MUST still be emitted before the UPDATE attempt |

---

## Test Coverage Map (which spec backs which US)

| US | Spec sections that back acceptance criteria |
|---|---|
| US-001 | Module A — all six functions (`_trace`, `_import_anthropic`, `_read_api_key`, `print_help`, `run_test`, `main`) and the AST-level invariants (no project imports). |
| US-002 | Module B — `process_with_anthropic` (16-row table), `process_with_anthropic_merge` (20-row table), plus the cross-cutting fallback chain. |
| US-003 | Module C — invariant snapshot of 8 existing presets, 2 new presets schema, `DEFAULT_POST_PROCESSOR == "claude-merge"`. |
| US-004 | Module D — `load_post_processor` rows 1–14 (especially #3, #4 for the idempotency contract) and `_post_process` rows 1–15. |
| US-005 | Module E — constants block, `init_db` table 1–5, `get_state` matrix rows 1–10. |

<promise>COMPLETE</promise>
