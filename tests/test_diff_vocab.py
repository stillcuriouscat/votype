"""Tests for diff_to_vocab, save_vocab, and supporting tokenization functions."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from post_processor_configs import (
    _tokenize_for_diff,
    _join_tokens,
    diff_to_vocab,
    save_vocab,
)


class TestTokenizeForDiff:
    """Tests for _tokenize_for_diff."""

    def test_chinese_individual_chars(self):
        assert _tokenize_for_diff("你好世界") == ["你", "好", "世", "界"]

    def test_english_words(self):
        assert _tokenize_for_diff("hello world") == ["hello", "world"]

    def test_mixed_chinese_english(self):
        assert _tokenize_for_diff("你好hello世界world") == [
            "你", "好", "hello", "世", "界", "world"
        ]

    def test_punctuation_stripped(self):
        assert _tokenize_for_diff("你好，世界！") == ["你", "好", "世", "界"]

    def test_empty_string(self):
        assert _tokenize_for_diff("") == []

    def test_only_punctuation(self):
        assert _tokenize_for_diff("，。！？") == []

    def test_numbers_stripped(self):
        assert _tokenize_for_diff("hello 123 world") == ["hello", "world"]


class TestJoinTokens:
    """Tests for _join_tokens."""

    def test_chinese_no_separator(self):
        assert _join_tokens(["你", "好"]) == "你好"

    def test_english_with_space(self):
        assert _join_tokens(["hello", "world"]) == "hello world"

    def test_mixed_chinese_then_english(self):
        assert _join_tokens(["你", "好", "hello"]) == "你好hello"

    def test_mixed_english_then_chinese(self):
        assert _join_tokens(["hello", "你", "好"]) == "hello你好"

    def test_empty_list(self):
        assert _join_tokens([]) == ""

    def test_single_chinese(self):
        assert _join_tokens(["好"]) == "好"

    def test_single_english(self):
        assert _join_tokens(["hello"]) == "hello"

    def test_multiple_english_words(self):
        assert _join_tokens(["Claude", "Code", "rocks"]) == "Claude Code rocks"


class TestDiffToVocab:
    """Tests for diff_to_vocab."""

    def test_no_changes_returns_copy(self):
        """Same text returns copy of vocab, no additions."""
        vocab = {"Ralph": {"variants": {"Raf": 5}}}
        result = diff_to_vocab("hello world", "hello world", vocab)
        assert result == vocab
        assert result is not vocab  # must be a new object

    def test_simple_english_replacement(self):
        """Single word replacement creates new vocab entry."""
        result = diff_to_vocab("Raf is here", "Ralph is here", {})
        assert "Ralph" in result
        assert result["Ralph"]["variants"]["Raf"] == 1

    def test_chinese_char_replacement(self):
        """Chinese character replacement detected."""
        result = diff_to_vocab("他说的时后", "他说的时候", {})
        assert "候" in result
        assert result["候"]["variants"]["后"] == 1

    def test_accumulate_new_variant(self):
        """Replacement for existing correct term adds new variant."""
        vocab = {"Ralph": {"variants": {"Raf": 3}}}
        result = diff_to_vocab("Rough is here", "Ralph is here", vocab)
        assert result["Ralph"]["variants"]["Rough"] == 1
        assert result["Ralph"]["variants"]["Raf"] == 3  # unchanged

    def test_increment_existing_variant(self):
        """Repeated replacement increments variant count."""
        vocab = {"Ralph": {"variants": {"Raf": 2}}}
        result = diff_to_vocab("Raf is here", "Ralph is here", vocab)
        assert result["Ralph"]["variants"]["Raf"] == 3

    def test_does_not_mutate_input(self):
        """Input vocab dict must not be modified."""
        vocab = {"Ralph": {"variants": {"Raf": 2}}}
        original_str = json.dumps(vocab, sort_keys=True)
        diff_to_vocab("Raf is here", "Ralph is here", vocab)
        assert json.dumps(vocab, sort_keys=True) == original_str

    def test_ignores_delete_opcodes(self):
        """Deleted tokens should not create vocab entries."""
        # "hello extra world" -> "hello world": "extra" is deleted
        result = diff_to_vocab("hello extra world", "hello world", {})
        # No entry for a pure deletion
        for correct, entry in result.items():
            for variant in entry["variants"]:
                assert variant != "extra"

    def test_ignores_insert_opcodes(self):
        """Inserted tokens should not create vocab entries."""
        result = diff_to_vocab("hello world", "hello new world", {})
        # "new" is an insert, not a replace — should produce no entries
        assert len(result) == 0

    def test_empty_vocab(self):
        """Works with empty vocab dict."""
        result = diff_to_vocab("Raf", "Ralph", {})
        assert "Ralph" in result
        assert result["Ralph"]["variants"]["Raf"] == 1

    def test_multi_token_chinese_replace(self):
        """Multi-character Chinese replacement joined without separator."""
        result = diff_to_vocab("克劳的", "克劳德", {})
        # Single char replace: "的" -> "德"
        assert "德" in result
        assert result["德"]["variants"]["的"] == 1

    def test_multi_token_english_replace(self):
        """Multi-word English replacement joined with space."""
        result = diff_to_vocab("Cloud Code is great", "Claude Code is great", {})
        assert "Claude" in result
        assert result["Claude"]["variants"]["Cloud"] == 1

    def test_mixed_replacement(self):
        """Mixed Chinese+English replacement works."""
        result = diff_to_vocab("他用Cloud做开发", "他用Claude做开发", {})
        assert "Claude" in result
        assert result["Claude"]["variants"]["Cloud"] == 1

    def test_punctuation_only_difference(self):
        """Strings differing only in punctuation create no vocab entries."""
        result = diff_to_vocab("你好！", "你好。", {})
        assert result == {}

    def test_multiple_replacements(self):
        """Multiple replacements in one diff all accumulate."""
        result = diff_to_vocab(
            "Raf用Cloud写代码",
            "Ralph用Claude写代码",
            {},
        )
        assert "Ralph" in result
        assert "Claude" in result

    def test_preserves_unrelated_vocab(self):
        """Existing vocab entries unrelated to diff are preserved."""
        vocab = {"session": {"variants": {"seshion": 2}}}
        result = diff_to_vocab("Raf here", "Ralph here", vocab)
        assert result["session"] == {"variants": {"seshion": 2}}
        assert "Ralph" in result


class TestSaveVocab:
    """Tests for save_vocab."""

    def test_writes_valid_json(self):
        """Saved file is valid JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            vocab = {"Ralph": {"variants": {"Raf": 3}}}
            save_vocab(vocab, path)
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded == vocab
        finally:
            os.unlink(path)

    def test_human_readable_indent(self):
        """JSON output has indent=2 for readability."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            vocab = {"Ralph": {"variants": {"Raf": 3}}}
            save_vocab(vocab, path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert '  "Ralph"' in content
        finally:
            os.unlink(path)

    def test_unicode_preserved(self):
        """Chinese characters preserved (ensure_ascii=False)."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            vocab = {"时候": {"variants": {"时后": 1}}}
            save_vocab(vocab, path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "时候" in content  # not escaped as \\uXXXX
        finally:
            os.unlink(path)

    def test_no_tmp_file_remains(self):
        """After save, no .tmp file should remain."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_vocab({"test": {"variants": {}}}, path)
            tmp_path = Path(path).with_suffix('.tmp')
            assert not tmp_path.exists()
        finally:
            os.unlink(path)

    def test_trailing_newline(self):
        """JSON file ends with newline."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_vocab({"test": {"variants": {}}}, path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content.endswith("\n")
        finally:
            os.unlink(path)

    def test_overwrites_existing_file(self):
        """Overwriting an existing file works correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_vocab({"old": {"variants": {}}}, path)
            save_vocab({"new": {"variants": {}}}, path)
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert "new" in loaded
            assert "old" not in loaded
        finally:
            os.unlink(path)
