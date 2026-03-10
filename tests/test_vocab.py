"""Tests for vocab functions in post_processor_configs.py — US-002."""

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from post_processor_configs import VOCAB_PATH, apply_vocab, glossary_context, load_vocab


class TestLoadVocab:
    """Tests for load_vocab()."""

    def test_load_valid_vocab(self, tmp_path):
        vocab_file = tmp_path / "vocab.json"
        data = {"Ralph": {"variants": {"Raf": 5, "Rough": 3}}}
        vocab_file.write_text(json.dumps(data), encoding="utf-8")

        result = load_vocab(vocab_file)
        assert result == data

    def test_file_not_found_returns_empty(self, tmp_path):
        result = load_vocab(tmp_path / "nonexistent.json")
        assert result == {}

    def test_invalid_json_returns_empty(self, tmp_path):
        vocab_file = tmp_path / "vocab.json"
        vocab_file.write_text("not valid json {{{", encoding="utf-8")

        result = load_vocab(vocab_file)
        assert result == {}

    def test_empty_file_returns_empty(self, tmp_path):
        vocab_file = tmp_path / "vocab.json"
        vocab_file.write_text("", encoding="utf-8")

        result = load_vocab(vocab_file)
        assert result == {}

    def test_non_dict_json_returns_empty(self, tmp_path):
        vocab_file = tmp_path / "vocab.json"
        vocab_file.write_text('["a", "b"]', encoding="utf-8")

        result = load_vocab(vocab_file)
        assert result == {}

    def test_default_path_is_vocab_path(self):
        """Default vocab_path should use VOCAB_PATH constant."""
        from post_processor_presets import VOICE_INPUT_DATA_DIR

        assert VOCAB_PATH == VOICE_INPUT_DATA_DIR / "vocab.json"

    def test_empty_dict_json(self, tmp_path):
        vocab_file = tmp_path / "vocab.json"
        vocab_file.write_text("{}", encoding="utf-8")

        result = load_vocab(vocab_file)
        assert result == {}


class TestApplyVocab:
    """Tests for apply_vocab()."""

    def test_chinese_replacement_no_boundary(self):
        """Chinese variants replaced without word boundary."""
        vocab = {"克劳德": {"variants": {"克老德": 5}}}
        result = apply_vocab("今天用克老德写代码", vocab, min_count=3)
        assert result == "今天用克劳德写代码"

    def test_english_replacement_with_boundary(self):
        """English variants use word boundaries to avoid partial matches."""
        vocab = {"Ralph": {"variants": {"Raf": 5}}}
        result = apply_vocab("Hello Raf how are you", vocab, min_count=3)
        assert result == "Hello Ralph how are you"

    def test_english_no_partial_word_match(self):
        """English replacement must not match inside words."""
        vocab = {"Ralph": {"variants": {"Raf": 5}}}
        result = apply_vocab("Rafting is fun", vocab, min_count=3)
        assert result == "Rafting is fun"

    def test_english_case_insensitive(self):
        """R2-L1: English variant matching is case-insensitive."""
        vocab = {"Ralph": {"variants": {"raf": 5}}}
        result = apply_vocab("Hello Raf today", vocab, min_count=3)
        assert result == "Hello Ralph today"

    def test_count_below_threshold_skipped(self):
        """Variants with count < min_count are not applied."""
        vocab = {"Ralph": {"variants": {"Raf": 2}}}
        result = apply_vocab("Hello Raf", vocab, min_count=3)
        assert result == "Hello Raf"

    def test_count_at_threshold_applied(self):
        """Variants with count == min_count are applied."""
        vocab = {"Ralph": {"variants": {"Raf": 3}}}
        result = apply_vocab("Hello Raf", vocab, min_count=3)
        assert result == "Hello Ralph"

    def test_empty_vocab_returns_original(self):
        vocab = {}
        text = "Hello world"
        result = apply_vocab(text, vocab, min_count=3)
        assert result == text

    def test_empty_text_returns_empty(self):
        vocab = {"Ralph": {"variants": {"Raf": 5}}}
        result = apply_vocab("", vocab, min_count=3)
        assert result == ""

    def test_mixed_chinese_english(self):
        """Mixed text with both Chinese and English replacements."""
        vocab = {
            "Claude": {"variants": {"cloud": 5}},
            "代码": {"variants": {"带马": 4}},
        }
        result = apply_vocab("用cloud写带马", vocab, min_count=3)
        assert result == "用Claude写代码"

    def test_overlapping_variants_longer_first(self):
        """R2-M1: Longer variants applied first to handle overlaps."""
        vocab = {
            "克劳德代码": {"variants": {"克劳的代马": 5}},
            "克劳德": {"variants": {"克劳的": 4}},
        }
        # The longer variant "克劳的代马" should match first
        result = apply_vocab("使用克劳的代马开发", vocab, min_count=3)
        assert result == "使用克劳德代码开发"

    def test_overlapping_shorter_still_works(self):
        """When only the shorter variant is present, it still matches."""
        vocab = {
            "克劳德代码": {"variants": {"克劳的代马": 5}},
            "克劳德": {"variants": {"克劳的": 4}},
        }
        result = apply_vocab("使用克劳的开发", vocab, min_count=3)
        assert result == "使用克劳德开发"

    def test_multiple_variants_for_same_term(self):
        """Multiple error variants for the same correct term."""
        vocab = {"Ralph": {"variants": {"Raf": 5, "Rough": 3}}}
        result = apply_vocab("Hello Raf and Rough", vocab, min_count=3)
        assert result == "Hello Ralph and Ralph"

    def test_no_variants_above_threshold(self):
        """All variants below threshold — text unchanged."""
        vocab = {"Ralph": {"variants": {"Raf": 1, "Rough": 2}}}
        result = apply_vocab("Hello Raf and Rough", vocab, min_count=3)
        assert result == "Hello Raf and Rough"

    def test_none_text_returns_none(self):
        """None/falsy text returns as-is."""
        vocab = {"Ralph": {"variants": {"Raf": 5}}}
        result = apply_vocab(None, vocab, min_count=3)
        assert result is None


class TestGlossaryContext:
    """Tests for glossary_context()."""

    def test_normal_vocab(self):
        vocab = {
            "Ralph": {"variants": {"Raf": 5}},
            "session": {"variants": {"section": 3}},
            "Claude Code": {"variants": {"cloud code": 4}},
        }
        result = glossary_context(vocab)
        assert result == "Commonly used terms: Ralph, session, Claude Code"

    def test_empty_vocab(self):
        result = glossary_context({})
        assert result == ""

    def test_single_term(self):
        vocab = {"Ralph": {"variants": {"Raf": 5}}}
        result = glossary_context(vocab)
        assert result == "Commonly used terms: Ralph"

    def test_none_vocab(self):
        """None vocab returns empty string."""
        result = glossary_context(None)
        assert result == ""
