# Function Specification (CLEAN-ROOM DOCUMENT)
> This document is shared between the Build agent and the Unit-Test agent.
> The Build agent implements these specs. The Unit-Test agent writes tests against them.
> Neither agent should read the other's output.

Generated: 2026-03-12T15:00:00+08:00
PRD: prd.json — branch `ralph/dual-asr-fusion`, 7 stories, dual ASR fusion (FireRedASR + faster-whisper → Gemini merge)
Architecture: HIGH_LEVEL_DESIGN.md, LOW_LEVEL_DESIGN.md

---

## Module 1: model_configs.py — ASR Framework Dispatch

> US-001 (already implemented). Documented for contract reference and test coverage.

---

### `ModelLoader.load_faster_whisper_model(config: Dict[str, Any], device: str = "cpu") -> WhisperModel`

**Purpose**: Load a faster-whisper model with CTranslate2 backend for CPU-only inference.

**Preconditions**:
- `config` is a dict (may be empty — defaults used for missing keys)
- `device` is a string (typically `"cpu"`)

**Postconditions**:
- Returns a `WhisperModel` instance ready for transcription
- Model is loaded with the specified `model_size` and `compute_type`

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: default config | `config={"model_size": "large-v3-turbo", "compute_type": "int8"}, device="cpu"` | `WhisperModel` instance | `WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")` called |
| 2 | Normal: custom model_size | `config={"model_size": "medium", "compute_type": "float16"}, device="cpu"` | `WhisperModel` instance | `WhisperModel("medium", device="cpu", compute_type="float16")` called |
| 3 | Edge: empty config uses defaults | `config={}, device="cpu"` | `WhisperModel` instance | `WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")` called |
| 4 | Error: faster-whisper not installed | `config={"model_size": "large-v3-turbo"}, device="cpu"` | raises `ImportError("faster-whisper is not installed. Install it with: pip install faster-whisper")` | None |

**Data Flow**: config → extract model_size/compute_type → `WhisperModel(model_size, device, compute_type)` → model

**Performance**: O(1) function call; model download on first use (~1.5GB, cached by HuggingFace Hub)

---

### `ModelLoader.load_model(model_id: str, device: str = DEVICE) -> tuple`

**Purpose**: Unified model loader dispatch. For `model_id="faster-whisper"`, delegates to `load_faster_whisper_model`.

**Preconditions**:
- `model_id` exists in `MODEL_PRESETS`

**Postconditions**:
- Returns `(model, framework, extra_data)` tuple
- For faster-whisper: `extra_data` is `None`

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: faster-whisper preset | `model_id="faster-whisper"` | `(WhisperModel, "faster-whisper", None)` | `load_faster_whisper_model` called with `device="cpu"` (force_cpu=True) |
| 2 | Normal: firered-asr preset | `model_id="firered-asr"` | `(model, "fireredasr", {"use_gpu": bool})` | Unchanged existing behavior |
| 3 | Edge: force_cpu overrides device | `model_id="faster-whisper", device="cuda:0"` | Model loaded with `device="cpu"` | Logged "forced to use CPU mode" |
| 4 | Error: unknown model_id | `model_id="nonexistent"` | raises `ValueError("Unknown model: nonexistent")` | None |

**Data Flow**: model_id → `MODEL_PRESETS[model_id]` → extract framework → dispatch to `load_faster_whisper_model` → `(model, "faster-whisper", None)`

**Performance**: O(1) dispatch + model load time

---

### `ModelInference.transcribe_faster_whisper(model: WhisperModel, audio_path: str) -> str`

**Purpose**: Transcribe audio using faster-whisper. Language auto-detected (not forced) for mixed Chinese-English.

**Preconditions**:
- `model` is a valid `WhisperModel` instance
- `audio_path` points to a valid WAV file

**Postconditions**:
- Returns concatenated text from all segments with no separator between them
- Language is auto-detected

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: multiple segments | `model.transcribe()` returns segments with `.text` = [" Hello ", "world"] | `" Hello world"` | None |
| 2 | Normal: single segment | `model.transcribe()` returns one segment with `.text` = "test output" | `"test output"` | None |
| 3 | Edge: empty segments (silence) | `model.transcribe()` returns empty iterable | `""` | None |
| 4 | Error: model.transcribe fails | model raises `RuntimeError` | raises `RuntimeError` (not caught here) | None |

**Data Flow**: `model.transcribe(audio_path)` → `(segments, info)` → unpack → `"".join(segment.text for segment in segments)` → text

**Performance**: O(audio_duration); ~0.1-0.3s for short audio on CPU

---

### `ModelInference.transcribe(model, audio_path, model_id, framework, extra_data, hotwords) -> str`

**Purpose**: Unified transcription interface. For `framework="faster-whisper"`, delegates to `transcribe_faster_whisper`.

**Preconditions**:
- `framework` is one of: `"funasr"`, `"transformers"`, `"glmasr"`, `"fireredasr"`, `"faster-whisper"`

**Postconditions**:
- Returns transcribed text string
- Leading clipping trimmed via `_trim_leading_clipping` before transcription

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: faster-whisper dispatch | `framework="faster-whisper"` | Text from `transcribe_faster_whisper` | `_trim_leading_clipping` called first; temp file cleaned up |
| 2 | Normal: fireredasr dispatch | `framework="fireredasr"` | Text from `transcribe_fireredasr` | Unchanged existing behavior |
| 3 | Edge: extra_data ignored for faster-whisper | `framework="faster-whisper", extra_data={"foo": 1}` | Text from `transcribe_faster_whisper` | extra_data not used |
| 4 | Error: unknown framework | `framework="unknown"` | raises `ValueError("Unknown framework: unknown")` | None |

**Data Flow**: `audio_path` → `_trim_leading_clipping` → `transcribe_faster_whisper(model, trimmed_path)` → text

**Performance**: O(audio_duration)

---

## Module 2: model_presets.py — ASR Model Configuration

> US-002 (already implemented). Documented for contract reference.

---

### `MODEL_PRESETS["faster-whisper"]` (data)

**Purpose**: Define static configuration for faster-whisper as a primary ASR model.

**Behavior Table**:

| # | Scenario | Field | Expected Value | Notes |
|---|----------|-------|---------------|-------|
| 1 | Normal: name | `name` | `"Faster-Whisper"` | Display name |
| 2 | Normal: description | `description` | Contains "English" and "CPU" | Mentions English strength and CPU-only |
| 3 | Normal: framework | `framework` | `"faster-whisper"` | Dispatch key |
| 4 | Normal: punctuation | `punctuation` | `"builtin"` | Whisper has built-in punctuation |
| 5 | Normal: force_cpu | `force_cpu` | `True` | Must not compete with FireRedASR for VRAM |
| 6 | Normal: model_size | `config["model_size"]` | `"large-v3-turbo"` | Best accuracy/speed tradeoff |
| 7 | Normal: compute_type | `config["compute_type"]` | `"int8"` | ~1.5GB RAM |

---

## Module 3: post_processor_configs.py — Post-Processing Logic

---

### `process_with_gemini_merge(primary_text: str, secondary_text: Optional[str], config: dict, glossary_ctx: str = "") -> str`

**Purpose**: Merge two ASR transcriptions via Vertex AI Gemini SSH proxy, or polish single text in fallback mode when secondary_text is None.

**Preconditions**:
- `config` contains keys: `ssh_host`, `proxy_script`, `model`, `vertex_region`, `timeout`, `min_text_len`, `max_text_len`
- `config` contains either `system_prompt_file` (path relative to `VOICE_INPUT_DATA_DIR`) or `system_prompt` (inline string)
- `primary_text` is a string (may be empty)
- `secondary_text` is a string or `None`

**Postconditions**:
- On success: returns merged/polished text from Gemini
- On any failure: returns `primary_text` (never raises, never returns empty unless `primary_text` is empty)
- SSH subprocess cleaned up

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: dual merge success | `primary_text="a"*50, secondary_text="b"*50, config=MERGE_CONFIG` | Stripped stdout from SSH subprocess | SSH called with JSON stdin containing `user_input="Chinese ASR: {primary}\nEnglish ASR: {secondary}"` |
| 2 | Normal: single fallback (secondary=None) | `primary_text="a"*50, secondary_text=None, config=MERGE_CONFIG` | Stripped stdout from SSH subprocess | SSH called with JSON stdin containing `user_input="Chinese ASR: {primary}"` (no English ASR line) |
| 3 | Normal: glossary appended | `glossary_ctx="Commonly used terms: Claude"` | Output text | `system_prompt` in JSON stdin ends with `"\n\nCommonly used terms: Claude"` |
| 4 | Normal: no glossary | `glossary_ctx=""` | Output text | `system_prompt` in JSON stdin has no glossary suffix |
| 5 | Edge: empty primary | `primary_text="", secondary_text="anything"` | `""` | SSH NOT called |
| 6 | Edge: primary below min_text_len | `primary_text="short" (len<45), config with min_text_len=45` | `"short"` (original primary) | SSH NOT called; logged "below min_text_len" |
| 7 | Edge: primary above max_text_len | `primary_text="a"*201, config with max_text_len=200` | `"a"*201` (original primary) | SSH NOT called; logged "exceeds max_text_len" |
| 8 | Edge: primary at exactly max_text_len | `primary_text="a"*200, config with max_text_len=200` | Output from SSH | SSH IS called (200 is not > 200) |
| 9 | Edge: primary at exactly min_text_len | `primary_text="a"*45, config with min_text_len=45` | Output from SSH | SSH IS called (45 is not < 45) |
| 10 | Error: SSH timeout | SSH takes longer than `config["timeout"]` seconds | `primary_text` (original) | `subprocess.TimeoutExpired` caught; `notify("Votype", "Gemini merge timed out after {timeout}s", urgency="low")` called |
| 11 | Error: SSH non-zero exit | SSH returns `returncode != 0` | `primary_text` (original) | Logged error with stderr; `notify("Votype", "Gemini merge error: {stderr[:100]}", urgency="low")` called |
| 12 | Error: hallucination guard | `len(output) > len(primary_text) * 2` | `primary_text` (original) | Logged warning "possible hallucination" |
| 13 | Edge: output at exactly 2x primary | `len(output) == len(primary_text) * 2` | output (accepted) | No guard triggered |
| 14 | Error: question guard | `'？' in primary_text` and `'？' not in output` and `'?' not in output` | `primary_text` (original) | Logged warning "dropped question marks" |
| 15 | Edge: question guard with ? (ASCII) | `'？' in primary_text` and `'?' in output` (ASCII question mark) | output (accepted) | No guard triggered — ASCII `?` satisfies guard |

**Data Flow**:
`primary_text` → length guards → load system_prompt (file or inline) → append glossary → construct user_input → JSON stdin → `subprocess.run(ssh cmd)` → strip stdout → hallucination guard → question guard → output

**SSH Command Construction**:
```
["ssh", "-o", "ConnectTimeout=5", config["ssh_host"], "python3", config["proxy_script"]]
```

**JSON stdin payload**:
```json
{
  "system_prompt": "<loaded prompt + optional glossary>",
  "user_input": "Chinese ASR: {primary_text}\nEnglish ASR: {secondary_text}",
  "model": "<config['model'] or 'gemini-2.5-flash'>",
  "region": "<config['vertex_region'] or 'us-central1'>"
}
```
When `secondary_text is None`, `user_input` is `"Chinese ASR: {primary_text}"` only.

**Performance**: ~5.5-6.5s (network-bound: SSH + Vertex AI Gemini inference)

---

### `PostProcessorLoader.load_post_processor(preset_id: str) -> Optional[Any]`

**Purpose**: Load a post-processor by preset ID. Extended to handle `framework="vertex-ai-merge"`.

**Preconditions**:
- `preset_id` is a string

**Postconditions**:
- Returns loaded model, or `None` for frameworks that don't need a local model
- For `vertex-ai-merge`: returns `None`

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: gemini-merge | `preset_id="gemini-merge"` | `None` | None |
| 2 | Normal: gemini-fix | `preset_id="gemini-fix"` | `None` | None |
| 3 | Normal: none | `preset_id="none"` | `None` | None |
| 4 | Normal: haiku-fix | `preset_id="haiku-fix"` | `None` | None |
| 5 | Normal: llama-cpp model | `preset_id="chinese-text-correction"` | `Llama` instance | Model loaded from GGUF file |
| 6 | Error: unknown preset | `preset_id="nonexistent"` | raises `ValueError("Unknown post-processor: nonexistent")` | None |
| 7 | Error: unknown framework | Preset with `framework="unknown"` | raises `ValueError("Unknown post-processor framework: unknown")` | None |

**Data Flow**: `preset_id` → lookup in `POST_PROCESSOR_PRESETS` → extract `framework` → dispatch → return model or None

**Performance**: O(1) for None-returning frameworks

---

## Module 4: post_processor_presets.py — Post-Processor Configuration

> US-005 (already implemented). Documented for contract reference.

---

### `POST_PROCESSOR_PRESETS["gemini-merge"]` (data)

**Purpose**: Define static configuration for the gemini-merge post-processor preset.

**Behavior Table**:

| # | Scenario | Field | Expected Value | Notes |
|---|----------|-------|---------------|-------|
| 1 | Normal: name | `name` | `"Gemini Merge (Dual ASR)"` | Display name |
| 2 | Normal: description | `description` | Contains "FireRedASR" and "faster-whisper" | Describes merge behavior |
| 3 | Normal: framework | `framework` | `"vertex-ai-merge"` | Distinct from `"vertex-ai"` |
| 4 | Normal: ssh_host | `config["ssh_host"]` | `"oracle-cloud"` | Same as gemini-fix |
| 5 | Normal: proxy_script | `config["proxy_script"]` | `"~/vertex_proxy.py"` | Same as gemini-fix |
| 6 | Normal: model | `config["model"]` | `"gemini-2.5-flash"` | Same as gemini-fix |
| 7 | Normal: vertex_region | `config["vertex_region"]` | `"us-central1"` | Same as gemini-fix |
| 8 | Normal: timeout | `config["timeout"]` | `15` | Seconds |
| 9 | Normal: min_text_len | `config["min_text_len"]` | `45` | Same as gemini-fix |
| 10 | Normal: max_text_len | `config["max_text_len"]` | `200` | Same as gemini-fix |
| 11 | Normal: vocab_min_count | `config["vocab_min_count"]` | `3` | Same as gemini-fix |
| 12 | Normal: system_prompt_file | `config["system_prompt_file"]` | `"prompts/gemini-merge-system.txt"` | Merge-specific prompt |
| 13 | Edge: no user_prompt_template_file | `config` | Key absent | Merge constructs user_input directly, unlike gemini-fix |

---

## Module 5: voice_input.py :: ASRDaemon — Daemon Core

---

### `ASRDaemon.__init__(self, model_id: Optional[str] = None) -> None`

**Purpose**: Initialize daemon state, including secondary model attributes.

**Preconditions**:
- None (constructor)

**Postconditions**:
- `self._secondary_model` is `None`
- `self._last_secondary_text` is `None`
- All other instance attributes initialized as per existing behavior

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: default init | `model_id=None` | Instance with all attrs set | `_secondary_model=None, _last_secondary_text=None` |
| 2 | Normal: explicit model | `model_id="firered-asr"` | Instance with `current_model_id="firered-asr"` | Same secondary attrs |

**Data Flow**: `model_id` → set defaults → `_restore_post_processor_id()` → instance ready

**Performance**: O(1)

---

### `ASRDaemon._load_secondary_model(self) -> None`

**Purpose**: Load faster-whisper as secondary ASR model for dual fusion. Non-fatal: all failures caught.

**Preconditions**:
- Called from `load_post_processor` when framework is `"vertex-ai-merge"`

**Postconditions**:
- `self._secondary_model` is either a `WhisperModel` instance or `None`
- Never raises an exception

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: successful load | faster-whisper installed, model available | `self._secondary_model = WhisperModel(...)` | Logged info "faster-whisper loaded successfully"; `WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")` called |
| 2 | Error: package not installed | `import faster_whisper` raises `ImportError` | `self._secondary_model = None` | Logged warning containing "faster-whisper not installed" and "pip install faster-whisper" |
| 3 | Error: model load fails | `WhisperModel()` constructor raises any `Exception` | `self._secondary_model = None` | Logged warning "Failed to load secondary ASR model: {e}" |

**Data Flow**: `import WhisperModel` → `WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")` → assign to `self._secondary_model`

**Performance**: First call downloads ~1.5GB model; subsequent calls use cached model

---

### `ASRDaemon._unload_secondary_model(self) -> None`

**Purpose**: Unload secondary ASR model to free ~1.5GB RAM.

**Preconditions**:
- None (safe to call even if no secondary model loaded)

**Postconditions**:
- `self._secondary_model` is `None`
- `self._last_secondary_text` is `None`

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: model loaded | `self._secondary_model` is a WhisperModel | None returned | `_secondary_model=None, _last_secondary_text=None`; logged "unloaded secondary ASR model" |
| 2 | Edge: no model loaded | `self._secondary_model` is `None` | None returned | No-op (no log, no state change) |
| 3 | Edge: attr missing (__new__ construction) | `self` has no `_secondary_model` attr | None returned | No-op — uses `getattr(self, '_secondary_model', None)` |

**Data Flow**: check `_secondary_model is not None` → set both to `None` → log

**Performance**: O(1)

---

### `ASRDaemon.load_post_processor(self, preset_id: Optional[str] = None) -> None`

**Purpose**: Load a post-processor. Manages secondary model lifecycle: loads for `vertex-ai-merge`, unloads otherwise.

**Preconditions**:
- `preset_id` exists in `POST_PROCESSOR_PRESETS` (or is `None` to use current)

**Postconditions**:
- `self.current_post_processor_id` updated
- `self.post_processor_framework` updated
- Post-processor ID persisted to state file
- If framework is `"vertex-ai-merge"`: vocab loaded AND `_load_secondary_model()` called
- If framework is NOT `"vertex-ai-merge"`: `_unload_secondary_model()` called

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: switch to gemini-merge | `preset_id="gemini-merge"` | None | `PostProcessorLoader.load_post_processor("gemini-merge")` called; vocab loaded; `_load_secondary_model()` called; ID persisted |
| 2 | Normal: switch to gemini-fix | `preset_id="gemini-fix"` | None | `_unload_secondary_model()` called; vocab loaded |
| 3 | Normal: switch to none | `preset_id="none"` | None | `_unload_secondary_model()` called; no vocab loaded |
| 4 | Normal: switch from gemini-merge to haiku-fix | Previous framework was `vertex-ai-merge` | None | `_unload_secondary_model()` called; secondary model freed |
| 5 | Edge: load gemini-merge but faster-whisper unavailable | ImportError in `_load_secondary_model` | None | `_secondary_model=None`; fusion silently disabled; still uses gemini-merge preset |
| 6 | Error: unknown preset | `preset_id="nonexistent"` | raises `RuntimeError("Unknown post-processor: nonexistent")` | No state changed |
| 7 | Error: loader exception | `PostProcessorLoader.load_post_processor` raises | None (caught) | Falls back to regex-only: `current_post_processor_id="none"`, `post_processor_framework="regex"`; `_unload_secondary_model()` called |

**Data Flow**: `preset_id` → validate → `PostProcessorLoader.load_post_processor()` → set framework → conditionally load vocab → conditionally load/unload secondary model → persist ID

**Performance**: O(1) for non-model frameworks; secondary model load adds ~5-30s on first use

---

### `ASRDaemon._handle_transcribe(self, msg: Dict[str, str]) -> Dict[str, str]`

**Purpose**: Handle transcription request. Runs primary ASR, then optionally runs secondary ASR when model is available.

**Preconditions**:
- `msg` is a dict with `"data"` key containing audio file path
- Primary model is loaded (`self.model is not None`)

**Postconditions**:
- `self._last_secondary_text` is updated: set to transcription result, or `None` on failure/unavailability
- Returns `{"text": str}` on success or `{"error": str}` on primary failure
- Status set to `"processing"` at start

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: primary only (no secondary model) | `msg={"data": "/tmp/audio.wav"}, self._secondary_model=None` | `{"text": "post-processed text"}` | `_last_secondary_text` reset to `None`; `_post_process()` called |
| 2 | Normal: dual ASR (secondary available) | `msg={"data": "/tmp/audio.wav"}, self._secondary_model=WhisperModel` | `{"text": "post-processed text"}` | Secondary starts in background Thread before primary; result collected after join(timeout=30); `_last_secondary_text` set to secondary transcription; `_post_process()` called |
| 3 | Edge: secondary transcription fails | `_run_secondary` thread catches Exception | `{"text": "post-processed text"}` (primary still works) | `result["error"]` set in thread; `_last_secondary_text=None`; warning logged "Secondary ASR failed: {e}" |
| 4 | Error: primary ASR fails | `self.transcribe()` returns `{"error": "..."}` | `{"error": "..."}` | Secondary may have started in parallel; result discarded; `_last_secondary_text = None` |
| 5 | Edge: secondary transcription produces text | Secondary returns segments with text | `{"text": ...}` | `_last_secondary_text` = joined segment text; logged "secondary: {text[:120]}" |
| 6 | Edge: secondary thread timeout (>30s) | `secondary_thread.is_alive()` after `join(timeout=30)` | `{"text": "post-processed text"}` | `_last_secondary_text = None`; warning logged "Secondary ASR timed out after 30s" |
| 7 | Edge: stale data reset | Any call, even without secondary model | `{"text": ...}` or `{"error": ...}` | `_last_secondary_text = None` at start of every call |

**Data Flow**:
```
set_status("processing")
  → _last_secondary_text = None (stale data reset)
  → if _secondary_model exists: start background Thread(_run_secondary)
  → self.transcribe(audio_path) → primary response  (main thread, GPU)
  → if secondary_thread: join(timeout=30), collect result
      → primary empty/failed → discard secondary
      → thread alive → timeout warning
      → result["error"] → secondary failed
      → result["text"] → assign _last_secondary_text
  → _post_process(primary_text) → final text
  → {"text": final_text}
```

**Secondary ASR call pattern** (in background thread `_run_secondary`, NOT via `ModelInference.transcribe_faster_whisper`):
```python
def _run_secondary(model, path, result):
    segments, _info = model.transcribe(path)
    result["text"] = "".join(seg.text for seg in segments)
```

**Performance**: max(Primary ASR, ~0.1-0.3s secondary ASR) + post-process time (parallel reduces wall-clock by 0.1-2s)

---

### `ASRDaemon._post_process(self, text: str) -> str`

**Purpose**: Multi-stage post-processing pipeline. Extended with `vertex-ai-merge` dispatch for dual ASR fusion.

**Preconditions**:
- `text` is a raw ASR transcription string
- `self.post_processor_framework` is set
- For `vertex-ai-merge`: `self._last_secondary_text` has been set by `_handle_transcribe`

**Postconditions**:
- Returns fully processed text
- Vocab updated if LLM changed the text (for ssh-claude, vertex-ai, vertex-ai-merge)

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: vertex-ai-merge with secondary text | `text="raw", framework="vertex-ai-merge", _last_secondary_text="whisper output"` | Merged text from Gemini | `process_with_gemini_merge(vocab_applied, "whisper output", config, glossary_ctx)` called |
| 2 | Normal: vertex-ai-merge without secondary (None) | `text="raw", framework="vertex-ai-merge", _last_secondary_text=None` | Polished text from Gemini (fallback) | `process_with_gemini_merge(vocab_applied, None, config, glossary_ctx)` called |
| 3 | Normal: vertex-ai (gemini-fix) | `text="raw", framework="vertex-ai"` | Polished text | `process_with_vertex_ai(vocab_applied, config, glossary_ctx)` called |
| 4 | Normal: ssh-claude (haiku-fix) | `text="raw", framework="ssh-claude"` | Polished text | `process_with_ssh_claude(vocab_applied, config, glossary_ctx)` called |
| 5 | Normal: regex only | `text="raw", framework="regex"` | Filler-removed text | Only `remove_fillers` applied (+ punc if punc_model) |
| 6 | Edge: LLM changed text → vocab updated | Before-polish differs from result | Processed text | `diff_to_vocab()` called; `save_vocab()` called; `load_vocab()` called to refresh |
| 7 | Edge: LLM returned same text → no vocab update | Before-polish equals result | Same text | `diff_to_vocab` NOT called |

**Pipeline for `vertex-ai-merge`**:
```
text
  → remove_fillers(text) → defillered
  → process_with_firered_punc(punc_model, defillered) → punctuated  (if punc_model)
  → apply_vocab(punctuated, vocab, min_count) → vocab_applied
  → process_with_gemini_merge(
        primary_text=vocab_applied,
        secondary_text=self._last_secondary_text,
        config=preset["config"],
        glossary_ctx=glossary_context(vocab)
    ) → merged
  → diff_to_vocab(vocab_applied, merged) → updated vocab  (if changed)
  → save_vocab(updated_vocab)  (if changed)
  → load_vocab() → refresh self._vocab  (if changed)
  → return merged
```

**Key difference from vertex-ai/ssh-claude dispatch**:
- `vertex-ai-merge` is dispatched via a separate `elif` branch (not the existing dispatch dict) because it has a 4-argument signature vs the 3-argument signature of `process_with_vertex_ai`/`process_with_ssh_claude`
- The secondary text is retrieved from `self._last_secondary_text` using `getattr(self, '_last_secondary_text', None)` for safety

**Performance**: Filler removal O(n), punc O(n), vocab O(n*v), Gemini ~5.5-6.5s, vocab diff O(n)

---

## Module 6: prompts/gemini-merge-system.txt — Merge Prompt

> US-004 (already implemented). Documented for contract reference.

---

### Prompt Content Contract

**Purpose**: Dual-purpose system prompt for Gemini: merge two ASR transcriptions or polish a single transcription.

**Behavior Table**:

| # | Scenario | Input Format | Expected Behavior | Notes |
|---|----------|-------------|-------------------|-------|
| 1 | Dual input: merge | `"Chinese ASR: ...\nEnglish ASR: ..."` | Merge Chinese text structure with English proper nouns/terms | Chinese ASR authoritative for Chinese, English ASR authoritative for English terms |
| 2 | Single input: polish | `"Chinese ASR: ..."` only | Clean up text as editor (remove fillers, fix errors, punctuate) | Matches gemini-fix behavior |
| 3 | CS domain context | Any input | Prioritize software engineering terminology | Same domain as gemini-fix-system.txt |
| 4 | Editor identity | Question-like input | Edit the text, do NOT answer the question | "你是编辑器，不是助手" |
| 5 | Output format | Any input | Clean text only, no labels, no explanations | No "Chinese ASR:" prefix in output |
| 6 | Filler removal | Input with 呃/嗯/就是说 | Remove fillers | Same rules as gemini-fix |
| 7 | CJK-English spacing | Mixed text | Add space between CJK and English/numbers | "使用Python" → "使用 Python" |
| 8 | Official capitalization | English terms | Strict official case | "github" → "GitHub", "vscode" → "VS Code" |

---

## Existing Functions (referenced, specs for test contract)

These existing functions are unchanged but referenced in the pipeline. Specs included for Unit-Test agent to verify integration.

---

### `PostProcessorInference.remove_fillers(text: str) -> str`

**Purpose**: Remove Chinese and English filler words via regex.

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: Chinese fillers | `"呃嗯就是说你好"` | `"你好"` | None |
| 2 | Normal: English fillers | `"Um like hello"` | `"hello"` | None |
| 3 | Edge: empty text | `""` | `""` | None |
| 4 | Edge: no fillers | `"纯净文本"` | `"纯净文本"` | None |

---

### `apply_vocab(text: str, vocab: dict, min_count: int) -> str`

**Purpose**: Replace known ASR error variants with correct terms.

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: Chinese variant | `text="克劳的", vocab={"Claude": {"variants": {"克劳的": 5}}}, min_count=3` | `"Claude"` | None |
| 2 | Edge: count below threshold | `text="克劳的", vocab={"Claude": {"variants": {"克劳的": 2}}}, min_count=3` | `"克劳的"` (unchanged) | None |
| 3 | Edge: empty vocab | `text="anything", vocab={}, min_count=3` | `"anything"` | None |
| 4 | Edge: empty text | `text="", vocab={"a": {"variants": {"b": 5}}}, min_count=1` | `""` | None |

---

### `glossary_context(vocab: dict) -> str`

**Purpose**: Generate glossary context string for LLM prompts.

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: vocab with terms | `{"Claude": {...}, "Python": {...}}` | `"Commonly used terms: Claude, Python"` | None |
| 2 | Edge: empty vocab | `{}` | `""` | None |

---

### `diff_to_vocab(original: str, polished: str, vocab: dict) -> dict`

**Purpose**: Extract word-level replacements and accumulate in vocab (immutable — returns new dict).

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: replacement found | `original="克劳的很好", polished="Claude很好"` | New vocab with `"Claude": {"variants": {"克劳的": 1}}` | None |
| 2 | Edge: no change | `original="same", polished="same"` | Deep copy of input vocab | None |
| 3 | Edge: single-char replacement skipped | Single CJK char replaced | Input vocab unchanged (skip) | None |

---

### `save_vocab(vocab: dict, vocab_path: Optional[str] = None) -> None`

**Purpose**: Save vocab atomically, merging with on-disk data.

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: merge and save | Vocab dict | None | Reads disk vocab, merges (max counts), writes via .tmp rename |
| 2 | Edge: no existing file | Vocab dict, path doesn't exist | None | Creates new file |

---

### `load_vocab(vocab_path: Optional[str] = None) -> dict`

**Purpose**: Load glossary vocab from JSON file.

**Behavior Table**:

| # | Scenario | Input | Expected Output | Side Effects |
|---|----------|-------|-----------------|--------------|
| 1 | Normal: valid file | Existing vocab.json | Parsed dict | None |
| 2 | Edge: missing file | Non-existent path | `{}` | None |
| 3 | Edge: invalid JSON | Malformed file | `{}` | None |

---

## Error Message Contract

These exact error messages are used in the codebase. Unit tests MUST assert on these strings.

| Function | Error Condition | Error Class | Exact Message Pattern |
|----------|----------------|-------------|----------------------|
| `ModelLoader.load_faster_whisper_model` | Package not installed | `ImportError` | `"faster-whisper is not installed. Install it with: pip install faster-whisper"` |
| `ModelLoader.load_model` | Unknown model_id | `ValueError` | `"Unknown model: {model_id}"` |
| `ModelLoader.load_model` | Unknown framework | `ValueError` | `"Unknown framework: {framework}"` |
| `PostProcessorLoader.load_post_processor` | Unknown preset_id | `ValueError` | `"Unknown post-processor: {preset_id}"` |
| `PostProcessorLoader.load_post_processor` | Unknown framework | `ValueError` | `"Unknown post-processor framework: {framework}"` |
| `ASRDaemon.load_post_processor` | Unknown preset | `RuntimeError` | `"Unknown post-processor: {preset_id}"` |

---

## Log Message Contract

These log messages are used in the codebase for observable side effects in tests.

| Function | Level | Message Pattern |
|----------|-------|----------------|
| `_load_secondary_model` (import fail) | `WARNING` | `"faster-whisper not installed, secondary ASR unavailable..."` |
| `_load_secondary_model` (load fail) | `WARNING` | `"Failed to load secondary ASR model: {e}"` |
| `_load_secondary_model` (success) | `INFO` | `"Loading secondary ASR model (faster-whisper large-v3-turbo, CPU, int8)..."` |
| `_handle_transcribe` (secondary fail) | `WARNING` | `"Secondary ASR failed: {e}"` |
| `process_with_gemini_merge` (timeout) | `WARNING` | `"Gemini merge timed out after {timeout}s"` |
| `process_with_gemini_merge` (SSH fail) | `ERROR` | `"Gemini merge failed (exit {returncode}): {stderr}"` |
| `process_with_gemini_merge` (hallucination) | `WARNING` | `"Gemini merge output too long ({len_out} vs input {len_in}), possible hallucination, using original text"` |
| `process_with_gemini_merge` (question) | `WARNING` | `"Gemini merge dropped question marks — likely answered instead of editing, using original text"` |
| `process_with_gemini_merge` (min_len skip) | `INFO` | `"Text length {len} below min_text_len {min}, skipping merge"` |
| `process_with_gemini_merge` (max_len skip) | `INFO` | `"Text length {len} exceeds max_text_len {max}, skipping merge"` |

---

## Test Configuration Constants

Tests should use these constants to avoid dependency on real prompt files and SSH.

```python
# For process_with_gemini_merge tests
MERGE_CONFIG = {
    "ssh_host": "oracle-cloud",
    "proxy_script": "~/vertex_proxy.py",
    "model": "gemini-2.5-flash",
    "vertex_region": "us-central1",
    "timeout": 15,
    "min_text_len": 45,
    "max_text_len": 200,
    "vocab_min_count": 3,
    "system_prompt": "You are a merge editor.",  # Inline — avoids file I/O in tests
}

# For ASRDaemon tests (bypass __init__ pattern from test_vertex_ai.py)
def _make_daemon():
    """Create minimal ASRDaemon for testing."""
    with patch("voice_input.ModelLoader"), \
         patch("voice_input.get_current_model", return_value="firered-asr"):
        from voice_input import ASRDaemon
        daemon = ASRDaemon.__new__(ASRDaemon)
        daemon.model = None
        daemon.framework = None
        daemon.extra_data = None
        daemon.current_model_id = "firered-asr"
        daemon.running = False
        daemon.indicator = None
        daemon.gtk_thread = None
        daemon.post_processor_model = None
        daemon.current_post_processor_id = "none"
        daemon.post_processor_framework = "regex"
        daemon.punc_model = None
        daemon._vocab = {}
        daemon._secondary_model = None
        daemon._last_secondary_text = None
        return daemon
```

---

## Mock Patterns

### Mocking subprocess for gemini-merge
```python
@patch("post_processor_configs.subprocess.run")
def test_example(self, mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="merged text", stderr="")
    # ...
```

### Mocking notify (lazy import from voice_input)
```python
@patch("voice_input.notify")
@patch("post_processor_configs.subprocess.run")
def test_example(self, mock_run, mock_notify):
    # ...
```

### Mocking WhisperModel for secondary model
```python
@patch("voice_input.WhisperModel")  # NOT valid — import is inside method
# Instead, mock the import mechanism:
with patch.dict("sys.modules", {"faster_whisper": MagicMock()}):
    # or
with patch("builtins.__import__", side_effect=ImportError):
    # ...
```

Note: `_load_secondary_model` imports `WhisperModel` inside the method body (`from faster_whisper import WhisperModel`). To mock it, patch `faster_whisper.WhisperModel` after ensuring the module is in `sys.modules`, or use `patch.dict("sys.modules", ...)`.
