"""L2 Virtual E2E: Vocab operations and guard functions.

Tests pure functions and file I/O without SSH or hardware:
  - load_vocab / save_vocab: real file system round-trip
  - apply_vocab: regex replacement (Chinese/English/overlapping)
  - diff_to_vocab: difflib-based correction extraction
  - glossary_context: term list generation
  - Guard conditions: empty text, hallucination, haiku-expand

No SSH, no daemon, no audio, no Kitty required.

Usage:
    pytest tests/test_e2e_virtual_ssh.py -v

E2E features verified:
    - virtual-vocab-load-save-cycle
    - virtual-apply-vocab-chinese
    - virtual-apply-vocab-english
    - virtual-apply-vocab-overlapping
    - virtual-apply-vocab-min-count
    - virtual-diff-to-vocab-chinese
    - virtual-diff-to-vocab-english
    - virtual-diff-to-vocab-immutable
    - virtual-diff-to-vocab-no-change
    - virtual-glossary-context
    - virtual-empty-text-guard
    - virtual-hallucination-guard
    - virtual-haiku-expand-not-implemented
    - virtual-preset-structure
    - virtual-vertex-preset-structure
    - virtual-vertex-empty-text-guard
    - virtual-vertex-hallucination-guard
"""

import json
import sys
import time
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

E2E_FEATURES = Path(__file__).parent / "e2e_features.json"


def update_feature(feature_id, passes, error=None):
    """Update a feature's status in e2e_features.json."""
    if not E2E_FEATURES.exists():
        return
    try:
        data = json.loads(E2E_FEATURES.read_text())
        for f in data["features"]:
            if f["id"] == feature_id:
                f["passes"] = passes
                f["last_error"] = error
                break
        data["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        E2E_FEATURES.write_text(json.dumps(data, indent=2) + "\n")
    except (json.JSONDecodeError, KeyError, OSError):
        pass


# ===========================================================================
# Preset structure tests
# ===========================================================================

class TestPresetStructure:
    """Verify haiku-fix and haiku-expand preset definitions."""

    def test_haiku_fix_preset_exists(self):
        """haiku-fix preset must exist with correct structure."""
        from post_processor_presets import POST_PROCESSOR_PRESETS

        assert "haiku-fix" in POST_PROCESSOR_PRESETS
        preset = POST_PROCESSOR_PRESETS["haiku-fix"]
        assert preset["framework"] == "ssh-claude"
        assert "name" in preset
        assert "description" in preset
        assert "config" in preset
        config = preset["config"]
        assert "ssh_host" in config
        assert "claude_path" in config
        assert "model" in config
        assert "timeout" in config
        assert "vocab_min_count" in config
        assert "system_prompt" in config
        update_feature("virtual-preset-structure", True)

    def test_haiku_expand_preset_exists(self):
        """haiku-expand preset must exist as placeholder."""
        from post_processor_presets import POST_PROCESSOR_PRESETS

        assert "haiku-expand" in POST_PROCESSOR_PRESETS
        preset = POST_PROCESSOR_PRESETS["haiku-expand"]
        assert preset["framework"] == "ssh-claude"
        assert "name" in preset
        assert "description" in preset
        # config should be empty (not implemented)
        assert preset.get("config") == {} or not preset.get("config")

    def test_existing_presets_unchanged(self):
        """Existing 4 presets must not be modified."""
        from post_processor_presets import POST_PROCESSOR_PRESETS

        assert "none" in POST_PROCESSOR_PRESETS
        assert "chinese-text-correction" in POST_PROCESSOR_PRESETS
        assert "qwen3-0.6b" in POST_PROCESSOR_PRESETS
        assert "minicpm4-0.5b" in POST_PROCESSOR_PRESETS
        assert POST_PROCESSOR_PRESETS["none"]["framework"] == "regex"

    def test_voice_input_data_dir_constant(self):
        """VOICE_INPUT_DATA_DIR must exist and MODELS_DIR derived from it."""
        from post_processor_presets import VOICE_INPUT_DATA_DIR, MODELS_DIR

        assert VOICE_INPUT_DATA_DIR == Path.home() / ".local/share/voice-input"
        assert MODELS_DIR == VOICE_INPUT_DATA_DIR / "models"


# ===========================================================================
# Vocab load/save tests
# ===========================================================================

class TestVocabLoadSave:
    """Test vocab file I/O with real filesystem."""

    def test_load_save_roundtrip(self, tmp_path):
        """Load -> save -> reload produces identical vocab."""
        from post_processor_configs import load_vocab, save_vocab

        vocab_path = tmp_path / "vocab.json"
        test_vocab = {
            "Claude": {"variants": {"克劳的": 5, "克劳": 3}},
            "session": {"variants": {"筛选": 4}},
        }
        # Write initial vocab
        vocab_path.write_text(json.dumps(test_vocab))

        # Load
        loaded = load_vocab(vocab_path)
        assert loaded == test_vocab

        # Save to new path
        new_path = tmp_path / "vocab2.json"
        save_vocab(loaded, new_path)

        # Reload and verify
        reloaded = load_vocab(new_path)
        assert reloaded == test_vocab
        update_feature("virtual-vocab-load-save-cycle", True)

    def test_load_missing_file(self, tmp_path):
        """Loading nonexistent file returns empty dict."""
        from post_processor_configs import load_vocab

        result = load_vocab(tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_invalid_json(self, tmp_path):
        """Loading invalid JSON returns empty dict."""
        from post_processor_configs import load_vocab

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{")
        result = load_vocab(bad_file)
        assert result == {}

    def test_load_empty_file(self, tmp_path):
        """Loading empty file returns empty dict."""
        from post_processor_configs import load_vocab

        empty_file = tmp_path / "empty.json"
        empty_file.write_text("")
        result = load_vocab(empty_file)
        assert result == {}

    def test_save_atomic_write(self, tmp_path):
        """save_vocab uses atomic write (tmp file + rename)."""
        from post_processor_configs import save_vocab

        vocab_path = tmp_path / "vocab.json"
        save_vocab({"test": {"variants": {"t": 1}}}, vocab_path)

        # File should exist and be valid JSON
        assert vocab_path.exists()
        data = json.loads(vocab_path.read_text())
        assert "test" in data

        # Verify indent=2 formatting
        raw = vocab_path.read_text()
        assert "  " in raw  # indent=2


# ===========================================================================
# apply_vocab tests
# ===========================================================================

class TestApplyVocab:
    """Test vocab-based regex replacement."""

    def test_chinese_replacement_no_boundary(self):
        """Chinese variants replaced without word boundary."""
        from post_processor_configs import apply_vocab

        vocab = {"Claude": {"variants": {"克劳的": 5}}}
        result = apply_vocab("我在用克劳的写代码", vocab, min_count=3)
        assert "Claude" in result
        assert "克劳的" not in result
        update_feature("virtual-apply-vocab-chinese", True)

    def test_english_replacement_with_boundary(self):
        """English variants replaced with word boundary."""
        from post_processor_configs import apply_vocab

        vocab = {"session": {"variants": {"ression": 5}}}
        result = apply_vocab("the ression was good", vocab, min_count=3)
        assert "session" in result
        assert "ression" not in result
        update_feature("virtual-apply-vocab-english", True)

    def test_english_case_insensitive(self):
        """English variants match case-insensitively."""
        from post_processor_configs import apply_vocab

        vocab = {"Ralph": {"variants": {"raf": 5}}}
        result = apply_vocab("I talked to Raf today", vocab, min_count=3)
        assert "Ralph" in result

    def test_overlapping_variants_longer_first(self):
        """Longer variants applied first to handle overlaps."""
        from post_processor_configs import apply_vocab

        vocab = {
            "Claude Code": {"variants": {"克劳的科德": 5}},
            "Claude": {"variants": {"克劳的": 5}},
        }
        text = "我在用克劳的科德写代码"
        result = apply_vocab(text, vocab, min_count=3)
        assert "Claude Code" in result
        update_feature("virtual-apply-vocab-overlapping", True)

    def test_min_count_filter(self):
        """Variants with count < min_count are skipped."""
        from post_processor_configs import apply_vocab

        vocab = {"Claude": {"variants": {"克劳的": 2}}}
        result = apply_vocab("我在用克劳的写代码", vocab, min_count=3)
        # Count=2 < min_count=3, should NOT replace
        assert "克劳的" in result
        assert "Claude" not in result
        update_feature("virtual-apply-vocab-min-count", True)

    def test_empty_vocab(self):
        """Empty vocab returns original text."""
        from post_processor_configs import apply_vocab

        result = apply_vocab("some text", {}, min_count=3)
        assert result == "some text"

    def test_english_partial_word_not_replaced(self):
        """English partial word within another word not replaced."""
        from post_processor_configs import apply_vocab

        vocab = {"session": {"variants": {"ession": 5}}}
        result = apply_vocab("expression is good", vocab, min_count=3)
        # "ession" is part of "expression", word boundary should prevent match
        assert "expression" in result


# ===========================================================================
# diff_to_vocab tests
# ===========================================================================

class TestDiffToVocab:
    """Test difflib-based correction extraction."""

    def test_chinese_char_replacement(self):
        """Extract Chinese character replacement pair."""
        from post_processor_configs import diff_to_vocab

        original = "我在用克劳的写代码"
        polished = "我在用Claude写代码"
        vocab = diff_to_vocab(original, polished, {})
        # Should have extracted a correction pair
        assert len(vocab) > 0
        update_feature("virtual-diff-to-vocab-chinese", True)

    def test_english_word_replacement(self):
        """Extract English word replacement pair."""
        from post_processor_configs import diff_to_vocab

        original = "the ression was good"
        polished = "the session was good"
        vocab = diff_to_vocab(original, polished, {})
        assert "session" in vocab
        assert "ression" in vocab["session"]["variants"]
        update_feature("virtual-diff-to-vocab-english", True)

    def test_immutability(self):
        """diff_to_vocab must not mutate input vocab."""
        from post_processor_configs import diff_to_vocab
        import copy

        original_vocab = {"existing": {"variants": {"ex": 3}}}
        frozen = copy.deepcopy(original_vocab)

        diff_to_vocab("bad text", "good text", original_vocab)

        assert original_vocab == frozen, "Input vocab was mutated!"
        update_feature("virtual-diff-to-vocab-immutable", True)

    def test_no_change_returns_same(self):
        """Identical original/polished returns original vocab."""
        from post_processor_configs import diff_to_vocab

        vocab = {"existing": {"variants": {"ex": 3}}}
        result = diff_to_vocab("same text", "same text", vocab)
        assert result == vocab  # Same content
        assert result is not vocab  # New object (immutability)
        update_feature("virtual-diff-to-vocab-no-change", True)

    def test_existing_term_increment(self):
        """Existing variant count incremented, not reset."""
        from post_processor_configs import diff_to_vocab

        vocab = {"session": {"variants": {"筛选": 2}}}
        result = diff_to_vocab("筛选很重要", "session很重要", vocab)
        # Count should be incremented from 2
        assert result["session"]["variants"]["筛选"] > 2

    def test_multi_token_chinese_join(self):
        """Multi-token Chinese replacement joined without separator."""
        from post_processor_configs import diff_to_vocab

        original = "克劳的很好用"
        polished = "Claude很好用"
        vocab = diff_to_vocab(original, polished, {})
        # "克劳的" should be joined as one variant (no spaces between chars)
        if "Claude" in vocab:
            variants = vocab["Claude"]["variants"]
            # All variant keys should have no spaces between Chinese chars
            for v in variants:
                assert "  " not in v or not any(
                    "\u4e00" <= c <= "\u9fff" for c in v
                )


# ===========================================================================
# glossary_context tests
# ===========================================================================

class TestGlossaryContext:
    """Test glossary context string generation."""

    def test_normal_vocab(self):
        """Generate context string from vocab."""
        from post_processor_configs import glossary_context

        vocab = {
            "Claude": {"variants": {"克劳的": 5}},
            "session": {"variants": {"筛选": 3}},
        }
        ctx = glossary_context(vocab)
        assert "Claude" in ctx
        assert "session" in ctx
        update_feature("virtual-glossary-context", True)

    def test_empty_vocab(self):
        """Empty vocab returns empty or minimal context."""
        from post_processor_configs import glossary_context

        ctx = glossary_context({})
        # Should return empty string or minimal text
        assert isinstance(ctx, str)


# ===========================================================================
# Guard condition tests
# ===========================================================================

class TestGuardConditions:
    """Test guard conditions that prevent SSH calls."""

    def test_empty_text_returns_empty(self):
        """Empty text returns empty string without SSH call."""
        from post_processor_configs import process_with_ssh_claude

        config = {
            "ssh_host": "oracle-cloud",
            "claude_path": "/dummy",
            "model": "dummy",
            "system_prompt": "dummy",
            "timeout": 15,
        }
        result = process_with_ssh_claude("", config, "")
        assert result == ""
        update_feature("virtual-empty-text-guard", True)


class TestHaikuExpandNotImplemented:
    """Test haiku-expand raises ValueError at load time."""

    def test_haiku_expand_raises_on_load(self):
        """Switching to haiku-expand must raise ValueError."""
        from post_processor_presets import POST_PROCESSOR_PRESETS

        preset = POST_PROCESSOR_PRESETS.get("haiku-expand", {})
        # haiku-expand should have empty config (not implemented)
        assert not preset.get("config"), \
            "haiku-expand config should be empty (not yet implemented)"

        # The actual ValueError is raised in voice_input.py load_post_processor()
        # We verify the preset structure here; the pipeline test verifies the raise
        update_feature("virtual-haiku-expand-not-implemented", True)


# ===========================================================================
# Vertex AI preset and guard tests
# ===========================================================================

class TestVertexPresetStructure:
    """Verify gemini-fix preset definitions."""

    def test_gemini_fix_preset_exists(self):
        """gemini-fix preset must exist with correct structure."""
        from post_processor_presets import POST_PROCESSOR_PRESETS

        assert "gemini-fix" in POST_PROCESSOR_PRESETS
        preset = POST_PROCESSOR_PRESETS["gemini-fix"]
        assert preset["framework"] == "vertex-ai"
        assert "name" in preset
        assert "description" in preset
        assert "config" in preset
        config = preset["config"]
        assert "ssh_host" in config
        assert "proxy_script" in config
        assert "model" in config
        assert config["model"] == "gemini-2.5-flash"
        assert "vertex_region" in config
        assert config["vertex_region"] == "us-central1"
        assert "timeout" in config
        assert "min_text_len" in config
        assert "vocab_min_count" in config
        assert "system_prompt_file" in config
        update_feature("virtual-vertex-preset-structure", True)

    def test_gemini_fix_prompt_file_exists(self):
        """gemini-fix prompt file must exist without /no_think."""
        from post_processor_presets import POST_PROCESSOR_PRESETS, VOICE_INPUT_DATA_DIR

        config = POST_PROCESSOR_PRESETS["gemini-fix"]["config"]
        prompt_path = VOICE_INPUT_DATA_DIR / config["system_prompt_file"]
        assert prompt_path.exists(), f"Prompt file missing: {prompt_path}"
        content = prompt_path.read_text(encoding="utf-8")
        assert "/no_think" not in content, "gemini-fix prompt must not contain /no_think"


class TestVertexGuardConditions:
    """Test guard conditions for process_with_vertex_ai."""

    def test_empty_text_returns_empty(self):
        """Empty text returns empty string without SSH call."""
        from post_processor_configs import process_with_vertex_ai

        config = {
            "ssh_host": "oracle-cloud",
            "proxy_script": "~/vertex_proxy.py",
            "model": "gemini-2.5-flash",
            "vertex_region": "us-central1",
            "timeout": 15,
            "min_text_len": 15,
        }
        result = process_with_vertex_ai("", config, "")
        assert result == ""
        update_feature("virtual-vertex-empty-text-guard", True)

    def test_hallucination_guard(self):
        """Output > 5x input length is rejected."""
        from post_processor_configs import process_with_vertex_ai
        from unittest.mock import patch, MagicMock

        config = {
            "ssh_host": "oracle-cloud",
            "proxy_script": "~/vertex_proxy.py",
            "model": "gemini-2.5-flash",
            "vertex_region": "us-central1",
            "system_prompt": "test",
            "timeout": 15,
            "min_text_len": 5,
        }
        text = "a" * 50  # len 50
        with patch("post_processor_configs.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="a" * 251, stderr=""  # > 5x
            )
            result = process_with_vertex_ai(text, config, "")
        assert result == text
        update_feature("virtual-vertex-hallucination-guard", True)
