#!/usr/bin/env python3
"""
Post-processor loading and inference logic.
Configuration is imported from post_processor_presets.py.

Pipeline: regex filler removal (always) -> LLM refinement (optional)
"""

import copy
import difflib
import json
import re
import logging
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from post_processor_presets import POST_PROCESSOR_PRESETS, DEFAULT_POST_PROCESSOR, VOICE_INPUT_DATA_DIR


# Vocab file path — derived from shared VOICE_INPUT_DATA_DIR constant
VOCAB_PATH = VOICE_INPUT_DATA_DIR / "vocab.json"


# Regex patterns for filler words (Chinese + English)
# Chinese fillers don't have spaces, so no word boundary needed
CHINESE_FILLER_PATTERN = re.compile(
    r'(?:'
    r'那个那个|就是说|然后嘛|'  # multi-char fillers first (greedy)
    r'呃+|嗯+|额+|哦+'         # single-char fillers (repeated)
    r')'
    r'[，,、\s]*',              # trailing punctuation/space after filler
    re.UNICODE
)
# Chinese 啊 is tricky: only remove when standalone or sentence-initial, not in words like 天气啊
CHINESE_STANDALONE_AH_PATTERN = re.compile(
    r'(?:^|(?<=[，,。！？\s]))啊+[，,、\s]*',
    re.UNICODE
)
# English fillers need word boundaries
ENGLISH_FILLER_PATTERN = re.compile(
    r'(?:^|(?<=\s))'
    r'(?:[Uu]mm?|[Uu]hh?|[Ee]rr?|[Aa]hh?|[Ll]ike|[Yy]ou know)'
    r'[,\s]*',
    re.UNICODE
)

# Pattern for repeated punctuation cleanup
REPEATED_PUNCT_PATTERN = re.compile(r'[，,、]{2,}')
# Pattern for leading punctuation
LEADING_PUNCT_PATTERN = re.compile(r'^[，,、\s]+')


def load_vocab(vocab_path=None):
    """Load glossary vocab from JSON file.

    Vocab format: {"correct_term": {"variants": {"error_form": count, ...}}, ...}

    Args:
        vocab_path: Path to vocab.json. Defaults to VOCAB_PATH.

    Returns:
        dict: Vocab dictionary, or {} if file missing/invalid.
    """
    path = vocab_path or VOCAB_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        logging.info("[VOCAB] loaded %d entries from %s", len(data), path)
        return data
    except FileNotFoundError:
        logging.info("[VOCAB] no vocab file found, starting empty")
        return {}
    except json.JSONDecodeError as e:
        logging.info("[VOCAB] corrupt vocab file, starting empty: %s", e)
        return {}
    except OSError:
        return {}


def apply_vocab(text, vocab, min_count):
    """Replace known ASR error variants with correct terms.

    Collects all (variant, correct_term) pairs where variant count >= min_count,
    sorts by variant length descending (longer variants first to handle overlaps),
    then applies regex replacements.

    Chinese variants: no word boundary (direct substring replacement).
    English variants: word boundary + case-insensitive matching.

    Args:
        text: Input text to fix.
        vocab: Vocab dict {correct: {variants: {error: count}}}.
        min_count: Minimum variant count threshold.

    Returns:
        Text with known error variants replaced.
    """
    if not text or not vocab:
        return text

    # Collect all (variant, correct_term) pairs meeting min_count threshold
    pairs = []
    for correct_term, entry in vocab.items():
        variants = entry.get("variants", {})
        for variant, count in variants.items():
            if count >= min_count:
                pairs.append((variant, correct_term))

    if not pairs:
        return text

    # Sort by variant length descending — longer variants first (R2-M1)
    pairs.sort(key=lambda p: len(p[0]), reverse=True)

    replacement_count = 0
    result = text
    for variant, correct_term in pairs:
        # Detect if variant is Chinese (contains CJK characters)
        if re.search(r'[\u4e00-\u9fff]', variant):
            # Chinese: no word boundary
            pattern = re.escape(variant)
            new_result = re.sub(pattern, correct_term, result)
        else:
            # English: ASCII letter boundary + case-insensitive (R2-L1)
            # Use (?<![a-zA-Z]) and (?![a-zA-Z]) instead of \b because
            # Python treats CJK as \w, so \b won't match at CJK-English boundaries
            pattern = r'(?<![a-zA-Z])' + re.escape(variant) + r'(?![a-zA-Z])'
            new_result = re.sub(pattern, correct_term, result, flags=re.IGNORECASE)
        if new_result != result:
            replacement_count += 1
        result = new_result

    if replacement_count > 0:
        logging.info("[VOCAB] applied %d vocab replacements", replacement_count)
    return result


def glossary_context(vocab):
    """Generate glossary context string for Haiku prompt.

    Lists correct terms from vocab so Haiku can recognize them.

    Args:
        vocab: Vocab dict {correct: {variants: {error: count}}}.

    Returns:
        Context string like "Commonly used terms: Ralph, session, Claude Code",
        or empty string if vocab is empty.
    """
    if not vocab:
        return ""

    terms = list(vocab.keys())
    logging.info("[VOCAB] glossary context: %d terms", len(terms))
    return "Commonly used terms: " + ", ".join(terms)


def process_with_ssh_claude(text, config, glossary_ctx=""):
    """Call Claude CLI on remote server via SSH for text polishing.

    System prompt passed via --system-prompt flag, ASR text via stdin.
    Glossary context appended to system prompt at call time.

    Args:
        text: ASR transcription text to polish.
        config: Preset config dict with ssh_host, claude_path, model, timeout, etc.
        glossary_ctx: Glossary context string to append to system prompt.

    Returns:
        Polished text on success, original text on any failure.
    """
    # Empty text: return immediately without SSH call
    if not text:
        return ""

    # Text too short: skip SSH call (not worth the latency)
    min_text_len = config.get("min_text_len", 15)
    if len(text) < min_text_len:
        logging.info(f"Text length {len(text)} below min_text_len {min_text_len}, skipping SSH")
        return text


    # Load system prompt from file or inline config
    if "system_prompt_file" in config:
        prompt_path = VOICE_INPUT_DATA_DIR / config["system_prompt_file"]
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        logging.info("[PROMPT] loaded system prompt: %s", prompt_path)
    else:
        system_prompt = config.get("system_prompt", "")
    if glossary_ctx:
        system_prompt += "\n\n" + glossary_ctx

    # Load user prompt template from file or inline config
    if "user_prompt_template_file" in config:
        tpl_path = VOICE_INPUT_DATA_DIR / config["user_prompt_template_file"]
        user_prompt_template = tpl_path.read_text(encoding="utf-8").strip()
        logging.info("[PROMPT] loaded user template: %s", tpl_path)
    else:
        user_prompt_template = config.get("user_prompt_template")
    user_input = user_prompt_template.format(text=text) if user_prompt_template else text

    # Build SSH command (C3+C4+N1+R3-L1)
    cmd = [
        "ssh", "-o", "ConnectTimeout=5",
        config["ssh_host"],
        config["claude_path"],
        "--model", config["model"],
        "--system-prompt", shlex.quote(system_prompt),
        "-p",
    ]

    timeout = config.get("timeout", 15)

    try:
        result = subprocess.run(
            cmd, input=user_input, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        logging.warning(f"SSH Claude timed out after {timeout}s")
        # Lazy import to avoid circular dependency
        from voice_input import notify
        notify("Votype", f"SSH Claude timed out after {timeout}s", urgency="low")
        return text

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        logging.error(f"SSH Claude failed (exit {result.returncode}): {stderr}")
        from voice_input import notify
        notify("Votype", f"SSH Claude error: {stderr[:100]}", urgency="low")
        return text

    output = result.stdout.strip()

    # Hallucination guard: editor output should not be much longer than input.
    # Legitimate edits add punctuation/spaces but remove fillers → roughly same length.
    # If output > 2x input, LLM likely "replied" instead of editing.
    if len(output) > len(text) * 2:
        logging.warning(
            f"SSH Claude output too long ({len(output)} vs input {len(text)}), "
            "possible hallucination, using original text"
        )
        return text

    # Question guard: if input contains '？' but output doesn't,
    # the LLM likely answered the question instead of editing it.
    if '？' in text and '？' not in output and '?' not in output:
        logging.warning(
            "SSH Claude dropped question marks — likely answered instead of editing, "
            "using original text"
        )
        return text

    logging.info("[SSH] haiku-fix success: %d→%d chars", len(text), len(output))
    return output


def _run_vertex_proxy(cmd, stdin_data, timeout, max_retries=1):
    """Run vertex_proxy.py via subprocess with retry on 429 RESOURCE_EXHAUSTED.

    Args:
        cmd: Command list for subprocess.run.
        stdin_data: JSON string to send via stdin.
        timeout: Timeout in seconds for each attempt.
        max_retries: Number of retries on 429 errors (default 1).

    Returns:
        subprocess.CompletedProcess from the last attempt.

    Raises:
        subprocess.TimeoutExpired: If the subprocess times out (not retried).
    """
    t0 = time.time()
    result = subprocess.run(
        cmd, input=stdin_data, capture_output=True, text=True, timeout=timeout
    )
    elapsed = time.time() - t0

    # Log remote traces from vertex_proxy.py stderr (sdk_init, gemini_api)
    stderr = result.stderr or ""
    trace_lines = [l for l in stderr.splitlines() if l.startswith("[TRACE]")]
    remote_trace = ", ".join(l.replace("[TRACE] ", "") for l in trace_lines)
    logging.info(
        f"[TRACE] vertex_proxy round-trip: {elapsed:.2f}s (rc={result.returncode})"
        + (f" | remote: {remote_trace}" if remote_trace else "")
    )

    if result.returncode != 0 and max_retries > 0:
        non_trace_stderr = "\n".join(l for l in stderr.splitlines() if not l.startswith("[TRACE]"))
        if "429" in non_trace_stderr or "RESOURCE_EXHAUSTED" in non_trace_stderr:
            logging.info("Vertex AI 429, retrying in 2s...")
            time.sleep(2)
            t0 = time.time()
            result = subprocess.run(
                cmd, input=stdin_data, capture_output=True, text=True, timeout=timeout
            )
            elapsed = time.time() - t0
            stderr2 = result.stderr or ""
            trace_lines2 = [l for l in stderr2.splitlines() if l.startswith("[TRACE]")]
            remote_trace2 = ", ".join(l.replace("[TRACE] ", "") for l in trace_lines2)
            logging.info(
                f"[TRACE] vertex_proxy retry: {elapsed:.2f}s (rc={result.returncode})"
                + (f" | remote: {remote_trace2}" if remote_trace2 else "")
            )

    return result


def process_with_vertex_ai(text, config, glossary_ctx=""):
    """Call Vertex AI Gemini via SSH proxy on Oracle Cloud for text polishing.

    System prompt and user text sent as JSON via stdin to vertex_proxy.py.
    Glossary context appended to system prompt at call time.

    Args:
        text: ASR transcription text to polish.
        config: Preset config dict with ssh_host, proxy_script, model, vertex_region, etc.
        glossary_ctx: Glossary context string to append to system prompt.

    Returns:
        Polished text on success, original text on any failure.
    """
    # Empty text: return immediately without SSH call
    if not text:
        return ""

    # Text too short: skip SSH call (not worth the latency)
    min_text_len = config.get("min_text_len", 15)
    if len(text) < min_text_len:
        logging.info(f"Text length {len(text)} below min_text_len {min_text_len}, skipping SSH")
        return text


    # Load system prompt from file or inline config
    if "system_prompt_file" in config:
        prompt_path = VOICE_INPUT_DATA_DIR / config["system_prompt_file"]
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        logging.info("[PROMPT] loaded system prompt: %s", prompt_path)
    else:
        system_prompt = config.get("system_prompt", "")
    if glossary_ctx:
        system_prompt += "\n\n" + glossary_ctx

    # Load user prompt template from file or inline config
    if "user_prompt_template_file" in config:
        tpl_path = VOICE_INPUT_DATA_DIR / config["user_prompt_template_file"]
        user_prompt_template = tpl_path.read_text(encoding="utf-8").strip()
        logging.info("[PROMPT] loaded user template: %s", tpl_path)
    else:
        user_prompt_template = config.get("user_prompt_template")
    user_input = user_prompt_template.format(text=text) if user_prompt_template else text

    # Build SSH command to call vertex_proxy.py on Oracle
    cmd = [
        "ssh", "-o", "ConnectTimeout=5",
        config["ssh_host"],
        "python3", config["proxy_script"],
    ]

    # JSON stdin avoids all shell escaping issues (Chinese text + long prompts)
    stdin_data = json.dumps({
        "system_prompt": system_prompt,
        "user_input": user_input,
        "model": config.get("model", "gemini-2.5-flash"),
        "region": config.get("vertex_region", "us-central1"),
    }, ensure_ascii=False)

    timeout = config.get("timeout", 15)

    try:
        result = _run_vertex_proxy(cmd, stdin_data, timeout)
    except subprocess.TimeoutExpired:
        logging.warning(f"Vertex AI timed out after {timeout}s")
        from voice_input import notify
        notify("Votype", f"Vertex AI timed out after {timeout}s", urgency="low")
        return text

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        logging.error(f"Vertex AI failed (exit {result.returncode}): {stderr}")
        from voice_input import notify
        notify("Votype", f"Vertex AI error: {stderr[:100]}", urgency="low")
        return text

    output = result.stdout.strip()

    # Hallucination guard: editor output should not be much longer than input.
    # Legitimate edits add punctuation/spaces but remove fillers → roughly same length.
    # If output > 2x input, LLM likely "replied" instead of editing.
    if len(output) > len(text) * 2:
        logging.warning(
            f"Vertex AI output too long ({len(output)} vs input {len(text)}), "
            "possible hallucination, using original text"
        )
        return text

    # Question guard: if input contains '？' but output doesn't,
    # the LLM likely answered the question instead of editing it.
    if '？' in text and '？' not in output and '?' not in output:
        logging.warning(
            "Vertex AI dropped question marks — likely answered instead of editing, "
            "using original text"
        )
        return text

    logging.info("[SSH] gemini-fix success: %d→%d chars", len(text), len(output))
    return output


def process_with_gemini_merge(primary_text, secondary_text, config, glossary_ctx=""):
    """Merge two ASR transcriptions via Vertex AI Gemini.

    Sends primary (FireRedASR) and secondary (faster-whisper) transcriptions
    to Gemini for intelligent merging. Falls back to single-text polish
    when secondary_text is None.

    Uses the same SSH + vertex_proxy.py mechanism as process_with_vertex_ai.

    Args:
        primary_text: Primary ASR (SenseVoice) transcription.
        secondary_text: Secondary ASR (faster-whisper) transcription, or None.
        config: Preset config dict with ssh_host, proxy_script, model, etc.
        glossary_ctx: Glossary context string to append to system prompt.

    Returns:
        Merged/polished text on success, primary_text on any failure.
    """
    # Empty text: return immediately
    if not primary_text:
        return ""

    # Text too short: skip SSH call (not worth the latency)
    min_text_len = config.get("min_text_len", 15)
    if len(primary_text) < min_text_len:
        logging.info(f"Text length {len(primary_text)} below min_text_len {min_text_len}, skipping merge")
        return primary_text


    # Load system prompt from file
    if "system_prompt_file" in config:
        prompt_path = VOICE_INPUT_DATA_DIR / config["system_prompt_file"]
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        logging.info("[PROMPT] loaded system prompt: %s", prompt_path)
    else:
        system_prompt = config.get("system_prompt", "")
    if glossary_ctx:
        system_prompt += "\n\n" + glossary_ctx

    # Build user input: dual or single format
    if secondary_text is not None:
        user_input = f"Chinese ASR: {primary_text}\nEnglish ASR: {secondary_text}"
    else:
        user_input = f"Chinese ASR: {primary_text}"

    # Build SSH command to call vertex_proxy.py on Oracle
    cmd = [
        "ssh", "-o", "ConnectTimeout=5",
        config["ssh_host"],
        "python3", config["proxy_script"],
    ]

    # JSON stdin avoids all shell escaping issues
    stdin_data = json.dumps({
        "system_prompt": system_prompt,
        "user_input": user_input,
        "model": config.get("model", "gemini-2.5-flash"),
        "region": config.get("vertex_region", "us-central1"),
    }, ensure_ascii=False)

    timeout = config.get("timeout", 15)

    try:
        result = _run_vertex_proxy(cmd, stdin_data, timeout)
    except subprocess.TimeoutExpired:
        logging.warning(f"Gemini merge timed out after {timeout}s")
        from voice_input import notify
        notify("Votype", f"Gemini merge timed out after {timeout}s", urgency="low")
        return primary_text

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        logging.error(f"Gemini merge failed (exit {result.returncode}): {stderr}")
        from voice_input import notify
        notify("Votype", f"Gemini merge error: {stderr[:100]}", urgency="low")
        return primary_text

    output = result.stdout.strip()

    # Hallucination guard: output should not be much longer than input
    if len(output) > len(primary_text) * 2:
        logging.warning(
            f"Gemini merge output too long ({len(output)} vs input {len(primary_text)}), "
            "possible hallucination, using original text"
        )
        return primary_text

    # Question guard: if input contains '？' but output doesn't,
    # the LLM likely answered the question instead of editing it
    if '？' in primary_text and '？' not in output and '?' not in output:
        logging.warning(
            "Gemini merge dropped question marks — likely answered instead of editing, "
            "using original text"
        )
        return primary_text

    logging.info(
        "[SSH] gemini-merge success: primary=%d, secondary=%s → %d chars",
        len(primary_text),
        str(len(secondary_text)) if secondary_text else "None",
        len(output),
    )
    return output


def _tokenize_for_diff(text):
    """Tokenize text into individual Chinese characters and English words.

    Each CJK character is a separate token; consecutive ASCII letters form one token.
    Punctuation and whitespace are discarded.

    Args:
        text: Input text.

    Returns:
        List of tokens.
    """
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text)


def _join_tokens(tokens):
    """Join tokens back to text.

    Chinese chars joined without separator; English words joined with space.

    Args:
        tokens: List of tokens (Chinese chars or English words).

    Returns:
        Joined string.
    """
    if not tokens:
        return ""
    parts = [tokens[0]]
    for i in range(1, len(tokens)):
        is_eng = bool(re.match(r'[a-zA-Z]', tokens[i]))
        prev_eng = bool(re.match(r'[a-zA-Z]', tokens[i - 1]))
        if is_eng and prev_eng:
            parts.append(' ')
        parts.append(tokens[i])
    return ''.join(parts)


def diff_to_vocab(original, polished, vocab):
    """Extract word-level replacements from original vs polished and accumulate in vocab.

    Uses SequenceMatcher on tokenized text. Only processes 'replace' opcodes;
    ignores 'delete' and 'insert' opcodes.

    Args:
        original: Original ASR text.
        polished: Haiku-polished text.
        vocab: Current vocab dict.

    Returns:
        NEW vocab dict with accumulated replacements (input not mutated).
    """
    if original == polished:
        return copy.deepcopy(vocab)

    orig_tokens = _tokenize_for_diff(original)
    pol_tokens = _tokenize_for_diff(polished)

    new_vocab = copy.deepcopy(vocab)
    extracted_count = 0

    matcher = difflib.SequenceMatcher(None, orig_tokens, pol_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'replace':
            continue

        error = _join_tokens(orig_tokens[i1:i2])
        correct = _join_tokens(pol_tokens[j1:j2])

        if not error or not correct:
            continue

        # Skip single-char corrections — too unreliable for CJK
        if len(correct) <= 1 or len(error) <= 1:
            continue

        if correct in new_vocab:
            old_variants = new_vocab[correct]["variants"]
            new_vocab[correct] = {
                "variants": {**old_variants, error: old_variants.get(error, 0) + 1}
            }
        else:
            new_vocab[correct] = {"variants": {error: 1}}
        extracted_count += 1

    if extracted_count > 0:
        logging.info("[VOCAB] extracted %d correction pairs from diff", extracted_count)
    return new_vocab


def save_vocab(vocab, vocab_path=None):
    """Save vocab dict to JSON file atomically, merging with on-disk data.

    Reads current file first to preserve entries added externally (e.g. glossary
    terms added by scripts while daemon is running). Merges variant counts
    additively, then writes atomically via .tmp rename.

    Args:
        vocab: Vocab dict from daemon memory.
        vocab_path: Optional path override (for testing). Defaults to VOCAB_PATH.
    """
    path = Path(vocab_path) if vocab_path else VOCAB_PATH

    # Merge with on-disk vocab to preserve externally-added entries
    logging.info("[VOCAB] merging %d in-memory + on-disk entries", len(vocab))
    disk_vocab = load_vocab(str(path))
    merged = copy.deepcopy(disk_vocab)
    for term, data in vocab.items():
        if term in merged:
            # Merge variants: keep max count for each variant
            existing_variants = merged[term].get("variants", {})
            new_variants = data.get("variants", {})
            for variant, count in new_variants.items():
                existing_variants[variant] = max(existing_variants.get(variant, 0), count)
            merged[term] = {"variants": existing_variants}
        else:
            merged[term] = copy.deepcopy(data)

    tmp_path = path.with_suffix('.tmp')

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")

    tmp_path.rename(path)
    logging.info("[VOCAB] saved %d entries to %s", len(merged), path)


class PostProcessorLoader:
    """Load post-processor models (llama-cpp GGUF) and punctuation models (FireRedPunc BERT)."""

    @staticmethod
    def load_firered_punc(config: dict) -> Any:
        """Load a FireRedPunc BERT punctuation model.

        Args:
            config: Config dict with model_dir pointing to FireRedPunc model directory.

        Returns:
            FireRedPunc instance
        """
        from fireredasr2s.fireredpunc.punc import FireRedPunc, FireRedPuncConfig

        model_dir = config["model_dir"]
        logging.info(f"Loading FireRedPunc model: {model_dir}")

        punc_config = FireRedPuncConfig(use_gpu=False)
        model = FireRedPunc.from_pretrained(model_dir, punc_config)
        logging.info("[PUNC] FireRedPunc loaded successfully")
        return model

    @staticmethod
    def load_llama_model(config: dict) -> Any:
        """Load a GGUF model via llama-cpp-python.

        Args:
            config: Model config dict with model_path, n_ctx, n_gpu_layers, etc.

        Returns:
            llama_cpp.Llama instance
        """
        from llama_cpp import Llama

        model_path = config["model_path"]
        logging.info(f"Loading GGUF model: {model_path}")

        model = Llama(
            model_path=model_path,
            n_ctx=config.get("n_ctx", 2048),
            n_gpu_layers=config.get("n_gpu_layers", -1),
            verbose=False,
        )
        logging.info("[LLM] Llama model loaded: %s", model_path)
        return model

    @classmethod
    def load_post_processor(cls, preset_id: str) -> Optional[Any]:
        """Load a post-processor by preset ID.

        Args:
            preset_id: Key in POST_PROCESSOR_PRESETS

        Returns:
            Loaded model instance, or None for "none" preset.

        Raises:
            ValueError: If preset_id is unknown.
            RuntimeError: If model loading fails.
        """
        if preset_id not in POST_PROCESSOR_PRESETS:
            raise ValueError(f"Unknown post-processor: {preset_id}")

        preset = POST_PROCESSOR_PRESETS[preset_id]
        framework = preset["framework"]
        logging.info("[POST] loading post-processor %s (framework=%s)", preset_id, framework)

        if framework == "regex":
            return None  # No model needed

        if framework == "ssh-claude":
            return None  # No model needed; SSH calls handled in _post_process()

        if framework == "vertex-ai":
            return None  # No model needed; SSH+proxy calls handled in _post_process()

        if framework == "vertex-ai-merge":
            return None  # No model needed; SSH+proxy calls handled in _post_process()

        if framework == "llama-cpp":
            try:
                return cls.load_llama_model(preset["config"])
            except Exception as e:
                raise RuntimeError(f"Failed to load post-processor {preset_id}: {e}")

        raise ValueError(f"Unknown post-processor framework: {framework}")


class PostProcessorInference:
    """Post-processing inference engine."""

    @staticmethod
    def remove_fillers(text: str) -> str:
        """Remove filler words from text using regex.

        Always runs regardless of LLM post-processor selection.

        Args:
            text: Raw transcription text

        Returns:
            Text with filler words removed
        """
        if not text:
            return text

        result = CHINESE_FILLER_PATTERN.sub('', text)
        result = CHINESE_STANDALONE_AH_PATTERN.sub('', result)
        result = ENGLISH_FILLER_PATTERN.sub('', result)
        # Clean up repeated punctuation left by filler removal
        result = REPEATED_PUNCT_PATTERN.sub('，', result)
        # Remove leading punctuation
        result = LEADING_PUNCT_PATTERN.sub('', result)
        return result.strip()

    @staticmethod
    def process_with_firered_punc(model: Any, text: str) -> str:
        """Process text through FireRedPunc for punctuation restoration.

        Preserves original English case by lowercasing input for the model
        (prevents tokenizer [UNK] on uppercase) and restoring case after.

        Args:
            model: FireRedPunc instance
            text: Text to add punctuation to

        Returns:
            Punctuated text with original English case preserved
        """
        if not text:
            return text

        # Build case mapping: lowered -> original for English words
        orig_words = {w.lower(): w for w in re.findall(r'[a-zA-Z]+', text)}

        # FireRedPunc expects lowercase English to avoid [UNK] tokens
        lowered = text.lower() if orig_words else text

        # Run punctuation restoration (batch API: list in, list of dicts out)
        results = model.process([lowered])
        result = results[0]["punc_text"]

        # Restore original English case
        if orig_words:
            result = re.sub(
                r'[a-zA-Z]+',
                lambda m: orig_words.get(m.group(), m.group()),
                result,
            )

        return result

    @staticmethod
    def process_with_llm(model: Any, text: str, prompt_template: str) -> str:
        """Process text through an LLM for refinement.

        Args:
            model: llama_cpp.Llama instance
            text: Text to refine (after filler removal)
            prompt_template: Prompt template with {text} placeholder

        Returns:
            Refined text from LLM
        """
        prompt = prompt_template.format(text=text)

        output = model.create_completion(
            prompt,
            max_tokens=len(text) * 3,  # allow generous output length
            temperature=0.1,           # low temperature for deterministic correction
            stop=["\n\n"],             # stop at double newline
            echo=False,
        )

        result = output["choices"][0]["text"].strip()
        # Guard against empty or obviously hallucinated output
        if not result or len(result) > len(text) * 5:
            logging.warning("LLM post-processor returned suspicious output, using original text")
            return text
        return result

    @classmethod
    def process(cls, text: str, model: Optional[Any], preset_id: str) -> str:
        """Unified post-processing interface.

        Pipeline: regex filler removal -> optional LLM refinement.

        Args:
            text: Raw transcription text
            model: Loaded LLM model (None for regex-only mode)
            preset_id: Post-processor preset ID

        Returns:
            Processed text
        """
        if not text:
            return text

        # Step 1: Always remove fillers
        result = cls.remove_fillers(text)

        if not result:
            return result

        # Step 2: Optional model-based refinement
        preset = POST_PROCESSOR_PRESETS.get(preset_id, {})
        framework = preset.get("framework")

        if framework == "llama-cpp" and model is not None:
            prompt_template = preset.get("config", {}).get("prompt_template", "")
            if prompt_template:
                try:
                    result = cls.process_with_llm(model, result, prompt_template)
                except Exception as e:
                    logging.error(f"LLM post-processing failed, using regex-only result: {e}")

        return result
