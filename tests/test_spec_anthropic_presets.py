"""Clean-room unit tests for US-003: claude-fix and claude-merge presets.

Derived from FUNCTION_SPEC.md Module C invariants.
"""

import pytest

from post_processor_presets import (
    POST_PROCESSOR_PRESETS,
    DEFAULT_POST_PROCESSOR,
    VOICE_INPUT_DATA_DIR,
    MODELS_DIR,
)
from pathlib import Path


EXISTING_PRESETS = {
    "none", "chinese-text-correction", "qwen3-0.6b", "minicpm4-0.5b",
    "haiku-fix", "haiku-expand", "gemini-fix", "gemini-merge",
}


class TestImports:
    def test_default_post_processor_exported(self):
        assert DEFAULT_POST_PROCESSOR == "claude-merge"

    def test_default_points_to_real_preset(self):
        assert DEFAULT_POST_PROCESSOR in POST_PROCESSOR_PRESETS

    def test_default_framework(self):
        assert POST_PROCESSOR_PRESETS[DEFAULT_POST_PROCESSOR]["framework"] == "anthropic-merge"

    def test_voice_input_data_dir_is_path(self):
        assert isinstance(VOICE_INPUT_DATA_DIR, Path)

    def test_models_dir_is_path(self):
        assert isinstance(MODELS_DIR, Path)


class TestClaudeFixPreset:
    def test_present(self):
        assert "claude-fix" in POST_PROCESSOR_PRESETS

    def test_structure(self):
        preset = POST_PROCESSOR_PRESETS["claude-fix"]
        assert preset["name"] == "Claude Fix (Anthropic)"
        assert preset["description"] == "ASR error correction via Claude Haiku 4.5 (Anthropic) over SSH"
        assert preset["framework"] == "anthropic"

    def test_config_keys(self):
        cfg = POST_PROCESSOR_PRESETS["claude-fix"]["config"]
        assert cfg["ssh_host"] == "oracle-cloud"
        assert cfg["proxy_script"] == "~/anthropic_proxy.py"
        assert cfg["model"] == "claude-haiku-4-5-20251001"
        assert cfg["timeout"] == 15
        assert cfg["min_text_len"] == 15
        assert cfg["vocab_min_count"] == 3
        assert cfg["system_prompt_file"] == "prompts/gemini-fix-system.txt"
        assert cfg["user_prompt_template_file"] == "prompts/haiku-fix-user.txt"


class TestClaudeMergePreset:
    def test_present(self):
        assert "claude-merge" in POST_PROCESSOR_PRESETS

    def test_structure(self):
        preset = POST_PROCESSOR_PRESETS["claude-merge"]
        assert preset["name"] == "Claude Merge (Dual ASR)"
        assert preset["description"] == "Merge SenseVoice + faster-whisper via Claude Haiku 4.5"
        assert preset["framework"] == "anthropic-merge"

    def test_config_keys(self):
        cfg = POST_PROCESSOR_PRESETS["claude-merge"]["config"]
        assert cfg["ssh_host"] == "oracle-cloud"
        assert cfg["proxy_script"] == "~/anthropic_proxy.py"
        assert cfg["model"] == "claude-haiku-4-5-20251001"
        assert cfg["timeout"] == 15
        assert cfg["min_text_len"] == 15
        assert cfg["vocab_min_count"] == 3
        assert cfg["system_prompt_file"] == "prompts/gemini-merge-system.txt"

    def test_no_user_template_file(self):
        cfg = POST_PROCESSOR_PRESETS["claude-merge"]["config"]
        assert "user_prompt_template_file" not in cfg
        assert "user_prompt_template" not in cfg
        assert "fallback_model" not in cfg


class TestExistingPresetsPreserved:
    def test_all_eight_present(self):
        for k in EXISTING_PRESETS:
            assert k in POST_PROCESSOR_PRESETS, f"missing preset: {k}"

    def test_none_unchanged(self):
        preset = POST_PROCESSOR_PRESETS["none"]
        assert preset["framework"] == "regex"

    def test_haiku_fix_unchanged(self):
        cfg = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]
        assert POST_PROCESSOR_PRESETS["haiku-fix"]["framework"] == "ssh-claude"
        assert cfg["model"] == "claude-haiku-4-5-20251001"
        assert cfg["timeout"] == 15
        assert cfg["min_text_len"] == 15
        assert cfg["vocab_min_count"] == 3
        assert cfg["system_prompt_file"] == "prompts/haiku-fix-system.txt"
        assert cfg["user_prompt_template_file"] == "prompts/haiku-fix-user.txt"

    def test_haiku_expand_unchanged(self):
        preset = POST_PROCESSOR_PRESETS["haiku-expand"]
        assert preset["framework"] == "ssh-claude"
        assert preset["config"] == {}

    def test_gemini_fix_unchanged(self):
        preset = POST_PROCESSOR_PRESETS["gemini-fix"]
        assert preset["framework"] == "vertex-ai"
        cfg = preset["config"]
        assert cfg["proxy_script"] == "~/vertex_proxy.py"
        assert cfg["fallback_model"] == "gemini-2.5-flash-lite"

    def test_gemini_merge_unchanged(self):
        preset = POST_PROCESSOR_PRESETS["gemini-merge"]
        assert preset["framework"] == "vertex-ai-merge"
        cfg = preset["config"]
        assert cfg["proxy_script"] == "~/vertex_proxy.py"
        assert cfg["fallback_model"] == "gemini-2.5-flash-lite"
        assert cfg["system_prompt_file"] == "prompts/gemini-merge-system.txt"


class TestPresetCount:
    def test_total_count_is_ten(self):
        assert len(POST_PROCESSOR_PRESETS) == 10
