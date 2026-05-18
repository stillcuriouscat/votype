# High-Level Design: Anthropic Claude Fallback

Generated: 2026-05-18 12:10 CST
PRD: prd.json — ralph/anthropic-fallback, 5 stories (all passes: false)

## 1. System Context

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Local Machine (dev) — /home/dev/code/voice_input/                       │
│                                                                          │
│  ┌──────────────┐    ┌─────────────────────────────┐  ┌───────────────┐  │
│  │ voice_input. │───▶│ post_processor_configs.py    │─▶│  notify log    │  │
│  │ py (daemon)  │    │  process_with_vertex_ai()    │  │  _log()        │  │
│  │              │    │  process_with_gemini_merge() │  └───────────────┘  │
│  │ _post_       │    │  process_with_anthropic()       (US-002 NEW)       │  │
│  │  process()   │    │  process_with_anthropic_merge() (US-002 NEW)       │  │
│  │ load_post_   │    │                              │                     │  │
│  │  processor() │    │ _run_vertex_proxy()  ◀── reused as-is              │  │
│  │  (US-004)    │    │  (cmd-agnostic subprocess     │                    │  │
│  │              │    │   runner with 429 retry +     │                    │  │
│  │              │    │   fallback_model)             │                    │  │
│  │              │    └───────────────┬───────────────┘                    │  │
│  │              │                    │                                    │  │
│  │              │                    │ subprocess.run(ssh cmd, stdin=JSON)│  │
│  │              ▼                    ▼                                    │  │
│  │  state_db.py     ┌─────────────────────────────────┐                   │  │
│  │  _DEPRECATED_PP  │ ssh -o ConnectTimeout=5         │                   │  │
│  │  (US-005)        │   oracle-cloud python3 <proxy>  │                   │  │
│  │   gemini-merge   └────────────────┬────────────────┘                   │  │
│  │   → claude-merge                  │ SSH tunnel                         │  │
│  └──────────────────────────────────┼─────────────────────────────────────┘  │
│                                     │                                        │
└─────────────────────────────────────┼────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────┼────────────────────────────────────────┐
│  Oracle Cloud (oracle-cloud)        │                                        │
│                                     ▼                                        │
│  ┌──────────────────────────┐  ┌──────────────────────────────────┐          │
│  │ ~/vertex_proxy.py         │  │ ~/anthropic_proxy.py  (US-001)   │          │
│  │  (unchanged)              │  │  NEW — mirrors vertex_proxy.py    │          │
│  │  google-genai SDK         │  │  anthropic SDK                    │          │
│  │  stdin: JSON              │  │  stdin: {system_prompt,           │          │
│  │  stdout: text             │  │          user_input, model,       │          │
│  │  reads ADC creds          │  │          max_tokens}              │          │
│  └──────────────┬───────────┘  │  stdout: text                     │          │
│                 │              │  reads ~/.config/claude.secret    │          │
│                 │ HTTPS        └────────────────┬──────────────────┘          │
│                 ▼                               │ HTTPS                       │
│  ┌──────────────────────────┐  ┌────────────────▼──────────────────┐         │
│  │ Vertex AI Gemini 2.5      │  │ Anthropic Claude Haiku 4.5         │         │
│  │ (existing path, kept)     │  │ claude-haiku-4-5-20251001          │         │
│  └──────────────────────────┘  └────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────────────────┘

Final fallback path (unchanged): OpenRouter (openrouter_client.py) — direct HTTPS
                                  from local machine, no SSH, no Oracle.
```

Data flow on a typical recording:
```
User hotkey → ASR (SenseVoice + faster-whisper) → _post_process()
  → load preset 'claude-merge' (framework='anthropic-merge')
  → process_with_anthropic_merge(primary, secondary, config, glossary)
  → ssh oracle-cloud python3 ~/anthropic_proxy.py  (JSON stdin)
    success → polished text → typed via xdotool
    failure → openrouter_client.call_openrouter(...)
              success → polished text
              failure → notify "both failed" + return primary_text
```

## 2. Module Decomposition

### Module A: `anthropic_proxy.py` (NEW, repo root + deployed to oracle-cloud:~/)

- **Responsibility**: Self-contained Anthropic Messages API proxy. Reads JSON from stdin, calls Anthropic, writes text to stdout. Strict mirror of `vertex_proxy.py` shape so the local caller code is interchangeable.
- **Public interface**: CLI script.
  - Stdin JSON schema: `{system_prompt: str, user_input: str, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1024}`
  - Stdout: `response.content[0].text.strip()` on success.
  - Exit 0 = success, exit 1 = failure (stderr carries the message).
  - Flags: `--help`, `--test` (verifies SDK import + API key file readable).
- **Dependencies**: `anthropic` SDK (PyPI), stdlib only otherwise. **No** imports from the voice_input project.
- **Owns**: Anthropic API call shape (model, max_tokens, system, messages), API-key file read (`~/.config/claude.secret`).

### Module B: `post_processor_configs.py` (MODIFY, local)

- **Responsibility**: Post-processing orchestration — builds prompts, dispatches to remote proxies via SSH, applies hallucination/question guards, performs OpenRouter fallback.
- **Public interface** (unchanged + 2 new functions):
  - existing: `process_with_ssh_claude`, `process_with_vertex_ai`, `process_with_gemini_merge`, `_run_vertex_proxy`, `apply_vocab`, `glossary_context`, `load_vocab`, `save_vocab`, `diff_to_vocab`.
  - **NEW US-002**: `process_with_anthropic(text, config, glossary_ctx="")` → polished text or original on failure.
  - **NEW US-002**: `process_with_anthropic_merge(primary_text, secondary_text, config, glossary_ctx="")` → merged text or `primary_text` on failure.
- **Dependencies**: `post_processor_presets.py`, `openrouter_client.py` (fallback), `voice_input.notify` (lazy).
- **Owns**: JSON stdin construction for proxy scripts, the `max_tokens = min(8192, max(512, len(user_input)))` formula, fallback chain.
- **Reuse note**: `_run_vertex_proxy()` takes a `cmd` list and `stdin_data` string. It is already proxy-script-agnostic — the new functions pass a different `cmd` (pointing to `~/anthropic_proxy.py`) but reuse the same retry/trace logic. The 429-specific retry branch is a no-op for Anthropic (Anthropic returns different error shapes); behavior degrades to single-attempt + OpenRouter fallback, which is acceptable.

### Module C: `post_processor_presets.py` (MODIFY, local)

- **Responsibility**: Static preset registry — declares user-selectable post-processors and their config dicts. Defines `DEFAULT_POST_PROCESSOR`.
- **Public interface**: `POST_PROCESSOR_PRESETS: dict[str, dict]`, `DEFAULT_POST_PROCESSOR: str`.
- **Changes (US-003)**:
  - Add `"claude-fix"` entry with `framework="anthropic"`, mirroring `gemini-fix` shape.
  - Add `"claude-merge"` entry with `framework="anthropic-merge"`, mirroring `gemini-merge` shape.
  - Both set `ssh_host="oracle-cloud"`, `proxy_script="~/anthropic_proxy.py"`, `model="claude-haiku-4-5-20251001"`, `timeout=15`, `min_text_len=15`, `vocab_min_count=3`.
  - `claude-fix.config.system_prompt_file = "prompts/gemini-fix-system.txt"`; `user_prompt_template_file = "prompts/haiku-fix-user.txt"`.
  - `claude-merge.config.system_prompt_file = "prompts/gemini-merge-system.txt"` (no user template; merge user_input is built in code).
  - `DEFAULT_POST_PROCESSOR = "claude-merge"`.
- **Invariant**: Existing presets (`none`, `chinese-text-correction`, `qwen3-0.6b`, `minicpm4-0.5b`, `haiku-fix`, `haiku-expand`, `gemini-fix`, `gemini-merge`) remain bit-for-bit unchanged.

### Module D: `voice_input.py` — `ASRDaemon._post_process` + `load_post_processor` (MODIFY, local)

- **Responsibility**: Daemon-side dispatch. Routes `(framework, text, ...)` to the correct processing function and decides whether the secondary ASR model must be loaded.
- **Public interface**: `_post_process(text)`, `load_post_processor(preset_id=None)`. Both already exist.
- **Changes (US-004)**:
  - `load_post_processor`: extend the "vocab needed" set and "secondary model needed" check.
    - Vocab loaded for frameworks: `{"ssh-claude", "vertex-ai", "vertex-ai-merge", "anthropic", "anthropic-merge"}`.
    - Secondary ASR loaded for frameworks: `{"vertex-ai-merge", "anthropic-merge"}`.
    - **Switching between `vertex-ai-merge` and `anthropic-merge` must not unload+reload the secondary model**: guard the load/unload with an `is_merge` boolean and only call `_load_secondary_model()` if not already loaded.
  - `_post_process`: in the SSH-dispatch block (currently L1071+):
    - Extend framework gate from `("ssh-claude", "vertex-ai", "vertex-ai-merge")` to also include `("anthropic", "anthropic-merge")`.
    - Import `process_with_anthropic, process_with_anthropic_merge`.
    - Extend the merge-vs-fix branch:
      - merge frameworks (`vertex-ai-merge`, `anthropic-merge`): call the corresponding merge function with `(result, secondary, config, glossary_ctx)`.
      - fix frameworks (`ssh-claude`, `vertex-ai`, `anthropic`): dispatch via a dict `{"ssh-claude": process_with_ssh_claude, "vertex-ai": process_with_vertex_ai, "anthropic": process_with_anthropic}`.
- **Owns**: Framework→function dispatch dicts (single source of truth, no separate registry module).

### Module E: `state_db.py` — `_DEPRECATED_PP` (MODIFY, local)

- **Responsibility**: SQLite-backed state; auto-migrates deprecated `post_processor` values on read.
- **Public interface**: `get_state`, `update_state`, `init_db` (all unchanged).
- **Changes (US-005)**: extend `_DEPRECATED_PP` from `{"firered-punc": "gemini-merge"}` to:
  ```python
  _DEPRECATED_PP = {
      "firered-punc": "claude-merge",   # bumped from gemini-merge (new default)
      "gemini-merge": "claude-merge",   # NEW: shift merge-mode users to Claude
  }
  ```
  Also update `_SAFE_DEFAULT["post_processor"]` and the `CREATE TABLE` `DEFAULT` clause from `"gemini-merge"` to `"claude-merge"` (so fresh installs match the new default).
- **Invariant**: `gemini-fix` and `haiku-fix` are **not** in the dict — users who explicitly chose a fix-mode keep it.

### Module F: `openrouter_client.py` (UNCHANGED)

- Already the final fallback. Both new functions (`process_with_anthropic`, `process_with_anthropic_merge`) call `call_openrouter()` on failure, identical to the existing Vertex AI pattern.

## 3. Data Flow

### 3.1 Single-ASR polish (framework = `anthropic`, preset = `claude-fix`)

```
voice_input._post_process(raw_text)
  ├─ remove_fillers(raw_text)                                  → text1
  ├─ process_with_firered_punc(...) (if punc_model loaded)     → text2 (rare for sensevoice path)
  ├─ apply_vocab(text2, self._vocab, min_count)                → text3
  ├─ glossary_context(self._vocab)                             → "Commonly used terms: ..."
  ├─ update_state(status="polishing")                          → blue icon
  ├─ process_with_anthropic(text3, config, glossary_ctx):
  │     load system_prompt from prompts/gemini-fix-system.txt
  │     append glossary_ctx
  │     load user template from prompts/haiku-fix-user.txt
  │     user_input = template.format(text=text3)
  │     max_tokens = min(8192, max(512, len(user_input)))
  │     stdin_json = {system_prompt, user_input,
  │                   model="claude-haiku-4-5-20251001", max_tokens}
  │     cmd = ["ssh", "-o", "ConnectTimeout=5", "oracle-cloud",
  │            "python3", "~/anthropic_proxy.py"]
  │     _run_vertex_proxy(cmd, stdin_json, timeout=15)
  │       success: stdout → polished
  │       failure: → openrouter fallback (same system_prompt, user_input)
  │     hallucination guard: len(polished) > 2 × len(text3) → return text3
  │     question guard: '？' in text3 but not in polished → return text3
  │     return polished
  ├─ diff_to_vocab + save_vocab (if polished != text3)
  └─ return polished                                           → typed via xdotool
```

### 3.2 Dual-ASR merge (framework = `anthropic-merge`, preset = `claude-merge`)

```
voice_input._post_process(primary_processed_text):
  ├─ ... same pre-steps through apply_vocab ...
  ├─ secondary = self._last_secondary_text   (faster-whisper raw)
  ├─ process_with_anthropic_merge(primary, secondary, config, glossary_ctx):
  │     short-text guard: if len(primary) < min_text_len:
  │         if secondary is longer → return secondary
  │         else → return primary
  │     load system_prompt from prompts/gemini-merge-system.txt
  │     append glossary_ctx
  │     user_input = "Chinese ASR: {primary}\nEnglish ASR: {secondary}"
  │                 (or "Chinese ASR: {primary}" if secondary is None)
  │     max_tokens = min(8192, max(512, len(user_input)))
  │     stdin_json = {..., model="claude-haiku-4-5-20251001", max_tokens}
  │     cmd = ssh + ~/anthropic_proxy.py
  │     _run_vertex_proxy(cmd, stdin_json, timeout=15)
  │       success → merged text
  │       failure → openrouter fallback (same system_prompt, user_input)
  │       both fail → notify "Anthropic merge + OpenRouter both failed"
  │                   return primary
  │     hallucination & question guards (same as 3.1)
  │     return merged
  └─ ...
```

### 3.3 DB auto-migration on daemon startup (US-005)

```
ASRDaemon.__init__:
  saved_state = state_db.get_state()
    ├─ SELECT post_processor FROM daemon_state WHERE id=1
    ├─ if value in _DEPRECATED_PP:
    │     UPDATE daemon_state SET post_processor='claude-merge' WHERE id=1
    │     log a notice (info level, not error)
    └─ return state with migrated value
  self.current_post_processor_id = saved if saved in PRESETS else DEFAULT_POST_PROCESSOR
  → load_post_processor() then dispatches to anthropic-merge path on first recording
```

### 3.4 Error propagation summary

| Failure point | Behavior |
|---|---|
| `anthropic_proxy.py` import fails on Oracle | exit 1, stderr → local `_run_vertex_proxy` sees `rc!=0` → OpenRouter fallback |
| Anthropic API 5xx / network | proxy catches exception → exit 1 → OpenRouter fallback |
| Anthropic API rate-limit (429) | proxy exits 1 with stderr containing "429"; existing 429 retry branch in `_run_vertex_proxy` retries once with same model, then falls through to OpenRouter (Anthropic has no `fallback_model` in preset, so retry path stops there) |
| SSH connection timeout (>5s) | `subprocess.TimeoutExpired` caught locally → OpenRouter fallback |
| OpenRouter also fails | notify "X + OpenRouter both failed" → return original text |
| Hallucination/question guard triggers | log warning, return original text (no notify) |

## 4. Technology Decisions

| Decision | Choice | Rationale | Alternatives Considered |
|----------|--------|-----------|------------------------|
| Anthropic transport | New `anthropic_proxy.py` invoked over SSH, mirroring `vertex_proxy.py` | Reuses the exact proven plumbing (`_run_vertex_proxy`, `ssh -o ConnectTimeout=5`, JSON stdin, OpenRouter fallback). Zero new infrastructure. | Direct local HTTPS to api.anthropic.com — would bypass Oracle, requires key on local machine and bypasses the existing trace/retry/timeout machinery. Rejected per PRD instruction to "mirror the existing pattern". |
| Anthropic SDK call shape | `client.messages.create(model, max_tokens, system, messages=[{role:'user', content:user_input}])` | Standard Anthropic Messages API. `system` as a string is the documented shape for system prompts. | `system` as a list of blocks — over-engineered for single-text system prompt. |
| API-key location on Oracle | `~/.config/claude.secret` (file read in proxy, never sent over the wire) | PRD-mandated; matches existing convention of keeping secrets on Oracle, never on the local dev box. | Env var injection via SSH `SendEnv` — fragile, requires sshd config on Oracle. |
| `max_tokens` formula | `min(8192, max(512, len(user_input)))` | Identical to the formula already proven in `process_with_vertex_ai` — output length ≈ input length for editing tasks; 512 floor for short text; 8192 cap for cost control. Anthropic Haiku 4.5 supports up to 8192 output tokens. | Hard-coded 1024 — would re-introduce the Gemini truncation bug for long recordings. |
| Preset-key naming | `claude-fix` / `claude-merge` | Symmetric with `gemini-fix` / `gemini-merge`; "claude" is the vendor-recognizable name (the existing `haiku-fix` already uses Anthropic, but predates this pattern). | `anthropic-fix` / `anthropic-merge` — vendor name not user-friendly; mismatches `gemini-*` symmetry. |
| Prompt-file reuse | Reuse `prompts/gemini-fix-system.txt`, `prompts/gemini-merge-system.txt`, `prompts/haiku-fix-user.txt` as-is | Prompts are task-specific, not LLM-specific. Both Gemini and Claude accept the same editor-style system prompt. Avoids duplicate-prompt maintenance burden. | Fork prompts into `prompts/claude-*.txt` — duplicates content, drifts over time. Per CLAUDE.md "同一数据只保存一份". |
| DB migration mechanism | Extend `_DEPRECATED_PP` dict in `state_db.py` (existing migration hook) | Already a single-row, idempotent, log-on-write pattern. Migration runs lazily on first `get_state()` call (no startup migration script). | One-shot migration script — extra deployment step; lazy migration is self-healing if users skip a release. |
| Migration policy: `gemini-fix` and `haiku-fix` NOT migrated | Explicit per PRD: only `gemini-merge` (the previous default) migrates | Users who deliberately chose a fix-mode have a non-default preference; respect it. | Migrate everything Gemini-based — overrides user choice. |
| Default change: `gemini-merge` → `claude-merge` | Set `DEFAULT_POST_PROCESSOR = "claude-merge"` in `post_processor_presets.py` AND `state_db._SAFE_DEFAULT["post_processor"]` AND the `CREATE TABLE` `DEFAULT` clause | Three places must agree — fresh installs (CREATE TABLE), DB read errors (SAFE_DEFAULT), and Python preset dispatch. Any divergence creates state-vs-config drift. | Change only one — guaranteed bug on fresh install or DB-read failure. |
| Reuse `_run_vertex_proxy` | Keep its name; do not rename to a generic name like `_run_proxy` | Already proven and tested. Renaming touches imports across files for zero behavior change. The function is structurally proxy-agnostic; the misleading name is acceptable tech-debt for this PR. | Rename to `_run_remote_proxy` — pure-cosmetic refactor with merge-conflict risk. |

## 5. Non-Functional Requirements

- **Latency (target)**: SSH round-trip ~50ms + Anthropic Haiku 4.5 API ~500–2000ms for ≤2k-char inputs → end-to-end p50 ≈ 1.5s, p95 ≈ 3s. Comparable to or better than the recent Gemini Vertex AI path (which has been degrading per PRD motivation). `timeout=15` matches existing presets.
- **Throughput**: Single-user manual hotkey workload (<1 call per 30s). No concurrency design needed; subprocess + SSH per call is fine.
- **Cost**: Claude Haiku 4.5 pricing ~$1/M input + $5/M output. Typical 35s recording: ~5k chars input × 2 (system + user) ≈ 3k tokens input + ~2k tokens output → ≈ $0.013 per recording. 50 recordings/day → ~$20/month worst case. Mitigated by 8192 output cap and OpenRouter fallback (free model).
- **Reliability**: Three-tier fallback preserved: Anthropic → OpenRouter → original text. Adds **one** new failure surface (anthropic_proxy.py on Oracle) but does not remove any existing path — Gemini presets remain selectable.
- **Storage**: No DB schema change; one new column-value mapping in `_DEPRECATED_PP`. No persistent state added.
- **Security**: API key stored on Oracle in `~/.config/claude.secret`, never transmitted from local. Key file must be `chmod 600`. Local code never sees the key. SSH command construction uses a fixed argv list — no shell interpolation, no injection surface.
- **Backward compatibility**:
  - Existing `gemini-fix`, `gemini-merge`, `haiku-fix` presets continue to work unchanged.
  - Users on `gemini-merge` are auto-migrated to `claude-merge` on next daemon start.
  - Users on `gemini-fix` keep `gemini-fix` (no surprise switch).
  - `vertex_proxy.py` on Oracle is untouched.
- **Deployment surface**: 1 new local file (`anthropic_proxy.py`), 1 new remote file (`oracle-cloud:~/anthropic_proxy.py` via scp), modifications to 4 existing local files. No new local dependencies (`anthropic` SDK is only needed on Oracle).

## 6. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | `anthropic_proxy.py` not deployed to Oracle before local code switches default to `claude-merge` | M | H | Deploy script: `scp anthropic_proxy.py oracle-cloud:~/anthropic_proxy.py` AND `ssh oracle-cloud "pip install --user anthropic && python3 ~/anthropic_proxy.py --test"` MUST pass before merging US-003 (which flips the default). OpenRouter fallback masks the failure but degrades quality. |
| 2 | `anthropic` SDK not installed on Oracle | M | H | `--test` flag in `anthropic_proxy.py` verifies SDK import + key file readable. Deploy-time check + clear `ImportError` message → operator runs `pip install --user anthropic` on Oracle. |
| 3 | `~/.config/claude.secret` missing or wrong permissions on Oracle | L | H | `--test` flag verifies file exists and is readable. Proxy returns exit 1 with explicit message; OpenRouter fallback prevents user-visible breakage. |
| 4 | `max_tokens` semantics differ between Anthropic and Gemini — Anthropic counts only output, Gemini counts thinking+output | L | M | Anthropic Haiku 4.5 has no implicit thinking budget; `max_tokens=8192` is hard output cap. Same formula is safe. Document in proxy docstring. |
| 5 | Loading secondary ASR (faster-whisper) on `anthropic-merge` switch duplicates work if user toggles between `gemini-merge` and `claude-merge` | M | M | Guard in `load_post_processor`: only call `_load_secondary_model()` if `self._secondary_model is None`; only call `_unload_secondary_model()` if the new framework is non-merge. PRD US-004 acceptance criterion explicitly requires this. |
| 6 | DB migration runs even for users who never used `gemini-merge` (no-op but logs a notice) | H | L | `_DEPRECATED_PP` lookup is keyed on actual stored value; migration only fires on `pp in _DEPRECATED_PP`. No-op for users on other presets. Log at info level, not error. |
| 7 | `_run_vertex_proxy`'s 429 retry branch checks stderr for "429"/"RESOURCE_EXHAUSTED" (Gemini-specific) — won't match Anthropic 429 (`rate_limit_error` JSON) | M | L | Acceptable: Anthropic failure falls through to OpenRouter on first attempt, which is the desired safety net. No silent failure. |
| 8 | `fallback_model` field in `_run_vertex_proxy` is unused by Anthropic presets (no Anthropic fallback model defined) | H | L | Pass `fallback_model=None` (default). The fallback branch is gated on `if fallback_model:` so it's a no-op. |
| 9 | Hallucination guard tuned for Chinese editing might mis-trigger if Claude phrases differently from Gemini | L | M | Same guard (`2× input length`) is already proven for both Gemini and SSH-Claude (which is also Anthropic). Keep identical for symmetry; revisit if real-world traces show false positives. |
| 10 | Prompt files (`gemini-fix-system.txt`, `gemini-merge-system.txt`) shared across vendors — a future Gemini-specific tweak could degrade Claude output | M | M | Document in preset config that prompts are vendor-agnostic; review prompt changes against both vendors. Alternative (forking) was rejected — accept this coupling. |
| 11 | First-run on a fresh machine: `state.db` `CREATE TABLE` DEFAULT must match preset DEFAULT, else first-call dispatch hits an unknown preset | L | H | US-005 design explicitly updates BOTH locations (`CREATE TABLE` clause AND `_SAFE_DEFAULT` AND `DEFAULT_POST_PROCESSOR`). Integration test must cover fresh-install path. |
| 12 | Test environment writes to live `state.db` (known existing bug per CLAUDE.md "Tests 污染生产 DB") could corrupt user state during CI run of new tests | M | M | New tests for US-005 MUST use `tmp_path / "state.db"` (pytest fixture) and never default to `~/.config/voice-input/state.db`. Pre-existing risk; do not worsen it. |

## 7. User Story Mapping

| Story ID | Module(s) | Key Interfaces | Notes |
|----------|-----------|----------------|-------|
| US-001 | A (`anthropic_proxy.py`) | `main()`: parse stdin JSON, call `client.messages.create(...)`, print `response.content[0].text.strip()`. `run_test()`: SDK import + key file readable. `print_help()`. | ~120 LOC, self-contained, no project imports. Deploy via `scp` to Oracle. |
| US-002 | B (`post_processor_configs.py`) | `process_with_anthropic(text, config, glossary_ctx="")`; `process_with_anthropic_merge(primary_text, secondary_text, config, glossary_ctx="")`. Both reuse `_run_vertex_proxy`, OpenRouter fallback, hallucination/question guards. JSON stdin uses `max_tokens` (not `max_output_tokens`). | Two new functions, ~70 LOC each, structurally identical to their Vertex AI counterparts. |
| US-003 | C (`post_processor_presets.py`) | Add `"claude-fix"` and `"claude-merge"` entries to `POST_PROCESSOR_PRESETS`; set `DEFAULT_POST_PROCESSOR = "claude-merge"`. | ~25 LOC added. Other presets untouched. |
| US-004 | D (`voice_input.py` — `_post_process`, `load_post_processor`) | Extend framework set in dispatch; add `"anthropic"` and `"anthropic-merge"` branches; guard secondary-model load/unload to be a no-op when switching between merge frameworks. | ~20 LOC modified in two methods. |
| US-005 | E (`state_db.py`) | `_DEPRECATED_PP["gemini-merge"] = "claude-merge"`; bump `_SAFE_DEFAULT["post_processor"]` to `"claude-merge"`; bump `CREATE TABLE … DEFAULT` to `'claude-merge'`. Migration logs at info level. | ~3 lines + 1 ALTER-equivalent (new DBs only). No data backfill needed — migration is lazy. |

### Dependency Order

```
US-001 (anthropic_proxy.py on Oracle)
   ↓
US-002 (post_processor_configs.py: process_with_anthropic[_merge])
   │  depends on US-001 deployment for E2E test
   ↓
US-003 (post_processor_presets.py: claude-fix, claude-merge presets + DEFAULT)
   │  depends on US-002 functions existing
   ↓
US-004 (voice_input.py dispatch)
   │  depends on US-002 + US-003 (needs functions and preset keys)
   ↓
US-005 (state_db.py migration)
   │  depends on US-003 (target preset must exist before users are migrated to it)
```

US-001 can be built in parallel with US-002/003/004 (only deployment to Oracle is blocking, not the local code). US-005 is the smallest change and should land last so that user state migration only happens after every other piece is in place.

### E2E acceptance gate (project-mandatory per CLAUDE.md)

Every story's `passes: true` requires:
1. Unit tests for the new function/preset/migration.
2. Integration test invoking `anthropic_proxy.py` against a stub (or live Oracle if available).
3. `voice-e2e-test` skill run end-to-end with `current_post_processor = claude-merge` succeeding on a real recording.

<promise>COMPLETE</promise>
