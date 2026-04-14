# High-Level Design: Gemini Output Truncation Fix

Generated: 2026-04-14 15:05 CST
PRD: prd.json — ralph/gemini-output-truncation-fix, 3 stories (all passes: false)

## 1. System Context

```
┌──────────────────────────────────────────────────────────────────────┐
│  Local Machine (dev)                                                 │
│                                                                      │
│  ┌──────────────┐    ┌───────────────────────┐    ┌───────────────┐  │
│  │ voice_input.py│───▶│post_processor_configs.py│──▶│ _log() notify │  │
│  │ (daemon)      │    │ process_with_vertex_ai()│   │ log file      │  │
│  │               │    │ process_with_gemini_   │   └───────────────┘  │
│  │ _post_process │    │   merge()              │                      │
│  │ _handle_      │    │                        │                      │
│  │  transcribe   │    │ Builds JSON stdin:     │                      │
│  └──────────────┘    │ {system_prompt,         │                      │
│                       │  user_input,            │                      │
│                       │  model, region}         │  ◀── US-002: add    │
│                       │                         │      max_output_    │
│                       │                         │      tokens here    │
│                       └───────────┬─────────────┘                     │
│                                   │ SSH + JSON stdin                  │
│                                   ▼                                   │
│                       ┌───────────────────────┐                       │
│                       │ ssh oracle-cloud       │                       │
│                       │   python3              │                       │
│                       │   ~/vertex_proxy.py    │                       │
│                       └───────────┬────────────┘                      │
└───────────────────────────────────┼───────────────────────────────────┘
                                    │ SSH tunnel
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Oracle Cloud (oracle-cloud)                                         │
│                                                                      │
│  ┌────────────────────────────────────────┐                          │
│  │ ~/vertex_proxy.py                       │                          │
│  │                                         │                          │
│  │ Reads JSON stdin ──▶ calls Gemini API   │                          │
│  │                                         │                          │
│  │ GenerateContentConfig(                  │                          │
│  │   max_output_tokens=512  ◀── US-001:    │                          │
│  │ )                           read from   │                          │
│  │                             stdin JSON  │                          │
│  │                                         │                          │
│  │ stdout: corrected text                  │                          │
│  └────────────────────┬───────────────────┘                          │
│                       │ HTTPS                                         │
│                       ▼                                               │
│  ┌────────────────────────────────────────┐                          │
│  │ Vertex AI Gemini 2.5 Flash (global)     │                          │
│  │ (or gemini-2.5-flash-lite fallback)     │                          │
│  └─────────────────────────────────────────┘                          │
└──────────────────────────────────────────────────────────────────────┘

Data flow for truncation bug:
  User speaks 35s audio
    → SenseVoice ASR: ~2250 chars primary
    → faster-whisper ASR: ~2373 chars secondary
    → user_input to Gemini: ~4637 chars
    → max_output_tokens=512 ← TRUNCATES to ~1164 chars
    → Fix: max_output_tokens=min(8192, max(512, len(user_input)))=4637
```

## 2. Module Decomposition

### Module A: `vertex_proxy.py` (Oracle Cloud: `~/vertex_proxy.py`)

- **Responsibility**: Standalone Gemini API proxy on Oracle Cloud. Reads JSON from stdin, calls Gemini, writes result to stdout.
- **Public interface**: CLI script — stdin JSON → stdout text. Exit 0 = success, exit 1 = failure.
- **Dependencies**: `google-genai` SDK (external), no voice_input project imports.
- **Owns**: Gemini API call configuration (`GenerateContentConfig`), including `max_output_tokens`.
- **Current state**: `max_output_tokens` hardcoded to 512 at line 150. JSON stdin schema: `{system_prompt, user_input, model, region}`. No `max_output_tokens` field recognized.

### Module B: `post_processor_configs.py` (Local: `~/code/voice_input/post_processor_configs.py`)

- **Responsibility**: Post-processing orchestration — builds prompts, sends to vertex_proxy.py via SSH, applies guards.
- **Public interface**:
  - `process_with_vertex_ai(text, config, glossary_ctx)` → polished text
  - `process_with_gemini_merge(primary_text, secondary_text, config, glossary_ctx)` → merged text
- **Dependencies**: `post_processor_presets.py` (preset configs), `openrouter_client.py` (fallback), `voice_input.py` (notify, lazy import).
- **Owns**: JSON stdin construction for vertex_proxy.py, hallucination/question guards, retry/fallback logic.
- **Current state**: JSON stdin at lines 393-398 and 515-520 does NOT include `max_output_tokens`. Formula `min(8192, max(512, len(user_input)))` must be computed here.

### Module C: `voice_input.py` (Local: `~/code/voice_input/voice_input.py`)

- **Responsibility**: Main daemon — audio recording, ASR, post-processing orchestration, GTK tray icon, CLI.
- **Public interface**: `_post_process(text)`, `_handle_transcribe(msg)`, `_log(tag, message)`.
- **Dependencies**: `post_processor_configs.py`, `post_processor_presets.py`, `state_db.py`, `model_configs.py`.
- **Owns**: Notify log file writes via `_log()`, training data CSV via `_log_csv()`.
- **Current state**: Five `[:120]` truncation sites in `_log()` calls (lines 1054, 1066, 1128, 1219, 1225). US-003 targets 4 of these (lines 1054, 1128, 1219, 1225). Line 1066 (PUNC) is not in the PRD but is the same pattern.

## 3. Data Flow

### US-001: vertex_proxy.py accepts max_output_tokens

```
stdin JSON (current):
  {"system_prompt": "...", "user_input": "...", "model": "gemini-2.5-flash", "region": "global"}

stdin JSON (after US-001):
  {"system_prompt": "...", "user_input": "...", "model": "gemini-2.5-flash", "region": "global",
   "max_output_tokens": 4637}

vertex_proxy.py main():
  data = json.loads(stdin)
  max_tokens = data.get("max_output_tokens", 512)  ← NEW (default=512 for backward compat)
  ...
  config = GenerateContentConfig(
      ...,
      max_output_tokens=max_tokens,                 ← CHANGED from hardcoded 512
  )
```

Error path: If `max_output_tokens` is absent, uses 512 (identical to current behavior). If present but non-integer, Gemini SDK raises TypeError → caught by existing try/except → exit 1 → caller falls back.

### US-002: post_processor_configs.py computes and passes max_output_tokens

```
process_with_vertex_ai(text, config, glossary_ctx):
  ...builds user_input from text + prompt template...
  max_output_tokens = min(8192, max(512, len(user_input)))   ← NEW
  stdin_data = json.dumps({
      "system_prompt": ...,
      "user_input": ...,
      "model": ...,
      "region": ...,
      "max_output_tokens": max_output_tokens,                ← NEW
  })

process_with_gemini_merge(primary_text, secondary_text, config, glossary_ctx):
  ...builds user_input as "Chinese ASR: {p}\nEnglish ASR: {s}"...
  max_output_tokens = min(8192, max(512, len(user_input)))   ← NEW
  stdin_data = json.dumps({
      ...,
      "max_output_tokens": max_output_tokens,                ← NEW
  })
```

Formula rationale:
- Floor 512: short text doesn't need more (backward compat)
- Ceiling 8192: prevents runaway cost on extremely long input
- Middle: len(user_input) — output should be roughly same length as input for an editing task

Real incident case: user_input ~4637 chars → max_output_tokens=4637 → no truncation.

### US-003: Remove log truncation

```
Before: _log("PP", f"input (...): {text[:120]}")
After:  _log("PP", f"input (...): {text}")

4 lines changed in voice_input.py:
  L1054: PP input log      — text[:120]  → text
  L1128: PP output log     — result[:120] → result
  L1219: ASR-2 secondary   — self._last_secondary_text[:120] → self._last_secondary_text
  L1225: ASR raw primary   — raw_primary[:120] → raw_primary
```

Note: Line 1066 (PUNC punctuation log) also uses `[:120]` but is NOT in the PRD scope. The `_log()` function writes to a file (not terminal), so length is not a concern.

## 4. Technology Decisions

| Decision | Choice | Rationale | Alternatives Considered |
|----------|--------|-----------|------------------------|
| max_output_tokens source | stdin JSON field, default 512 | Backward compatible — existing callers without the field behave identically | Hardcode higher value (e.g., 4096) — simpler but wastes tokens on short text |
| Token budget formula | `min(8192, max(512, len(user_input)))` | Output length ≈ input length for editing tasks; 512 floor preserves current behavior; 8192 cap prevents cost explosion | `len(user_input) * 1.5` — overestimates; fixed 4096 — still truncates 5000+ char inputs |
| Formula placement | Computed in `post_processor_configs.py`, not `vertex_proxy.py` | Caller knows the semantic context (editing task → output ≈ input); proxy stays generic | Compute in proxy from user_input length — couples proxy to editing assumption |
| Log truncation removal | Remove `[:120]` entirely, no replacement | `_log()` writes to file, not terminal; full text needed for debugging (this truncation caused the original misdiagnosis) | Replace with `[:500]` — arbitrary limit; log rotation — separate concern |
| vertex_proxy.py deployment | scp updated file to Oracle Cloud | Existing deployment pattern (file is already at `~/vertex_proxy.py` on oracle-cloud) | Git clone on Oracle — overkill for a single file |

## 5. Non-Functional Requirements

- **Latency**: No latency change. Adding one JSON field adds negligible parsing overhead. `max_output_tokens` increase may slightly increase Gemini response time for long text (more tokens to generate), but the current 512 limit is the bug — the extra generation time is desired behavior.
- **Cost**: Gemini 2.5 Flash output pricing is per-token. Worst case: 8192 output tokens instead of 512 = 16x cost increase for that single call. Mitigated by: (a) 8192 cap, (b) real-world long recordings are infrequent, (c) Gemini Flash is cheap (~$0.15/M output tokens on Vertex AI).
- **Disk**: Log files grow larger without `[:120]` truncation. Mitigated by: (a) recordings are infrequent (manual hotkey trigger), (b) notify log file is append-only and small, (c) log rotation is a separate concern.
- **Reliability**: No change to retry/fallback chain. `_run_vertex_proxy()` retry on 429 + fallback to flash-lite + fallback to OpenRouter all preserved.
- **Backward compatibility**: `vertex_proxy.py` defaults `max_output_tokens` to 512 when field is absent. Existing callers (if any) that don't include the field see zero behavior change.
- **Security**: No new attack surface. `max_output_tokens` is an integer consumed only by the Gemini SDK. No user-facing input reaches this parameter.

## 6. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | vertex_proxy.py on Oracle is out of sync with local copy | M | H | scp the updated file to Oracle after local edit; verify with `ssh oracle-cloud "python3 ~/vertex_proxy.py --test"` |
| 2 | Gemini returns garbage when max_output_tokens is very large | L | M | 8192 cap + existing hallucination guard (2x input length) + question guard |
| 3 | Non-integer max_output_tokens in JSON crashes vertex_proxy.py | L | L | Gemini SDK validates the type; proxy's try/except catches and exits 1; caller falls back to original text |
| 4 | Log file size grows unbounded after removing [:120] | L | L | User's recording frequency is low (manual hotkey); log rotation is a separate concern; can add rotation later |
| 5 | Fallback model (flash-lite) also needs increased max_output_tokens | M | M | `_run_vertex_proxy()` replaces only `model` in the JSON payload (line 321); `max_output_tokens` is preserved in the payload, so fallback inherits the dynamic value automatically |
| 6 | OpenRouter fallback path doesn't benefit from this fix | L | L | OpenRouter's `call_openrouter()` uses its own max_tokens (not configurable from vertex_proxy.py). But OpenRouter is only hit when Vertex AI fails entirely, not when output is truncated — different failure mode |
| 7 | `[:120]` removal misses line 1066 (PUNC log) | L | L | PRD scope is 4 lines; line 1066 is punctuation output log, not ASR/PP output. Can be addressed separately if needed |

## 7. User Story Mapping

| Story ID | Module(s) | Key Interfaces | Notes |
|----------|-----------|---------------|-------|
| US-001 | vertex_proxy.py (Module A) | `main()`: parse `data.get("max_output_tokens", 512)`, pass to `GenerateContentConfig(max_output_tokens=...)` | ~5 lines changed: add data.get(), update config constructor, update print_help() and docstring |
| US-002 | post_processor_configs.py (Module B) | `process_with_vertex_ai()`: compute formula, add to stdin JSON (line ~397). `process_with_gemini_merge()`: same formula, add to stdin JSON (line ~519) | 2 call sites, identical formula: `min(8192, max(512, len(user_input)))` |
| US-003 | voice_input.py (Module C) | `_post_process()`: lines 1054, 1128. `_handle_transcribe()`: lines 1219, 1225 | Pure deletion of `[:120]` — 4 lines, mechanical change |

### Dependency Order

```
US-001 (vertex_proxy.py accepts field)
  ↓
US-002 (post_processor_configs.py sends field)  ← depends on US-001 being deployed
  ↓
US-003 (log truncation removal)  ← independent, can be done in parallel with US-001/002
```

US-003 is fully independent — it touches `voice_input.py` which neither US-001 nor US-002 modify. US-001 must be deployed to Oracle before US-002 can be tested end-to-end, but the code changes can be written in parallel.
