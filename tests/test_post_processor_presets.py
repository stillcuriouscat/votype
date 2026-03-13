"""Tests for post_processor_presets.py — US-001."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from post_processor_presets import (
    MODELS_DIR,
    POST_PROCESSOR_PRESETS,
    VOICE_INPUT_DATA_DIR,
)


class TestVoiceInputDataDir:
    """VOICE_INPUT_DATA_DIR and MODELS_DIR consistency."""

    def test_models_dir_derived_from_data_dir(self):
        assert MODELS_DIR == VOICE_INPUT_DATA_DIR / "models"

    def test_data_dir_is_path(self):
        assert isinstance(VOICE_INPUT_DATA_DIR, Path)

    def test_models_dir_is_path(self):
        assert isinstance(MODELS_DIR, Path)


class TestExistingPresetsUnchanged:
    """Existing presets must not be modified."""

    def test_none_preset_exists(self):
        assert "none" in POST_PROCESSOR_PRESETS
        assert POST_PROCESSOR_PRESETS["none"]["framework"] == "regex"

    def test_chinese_text_correction_exists(self):
        assert "chinese-text-correction" in POST_PROCESSOR_PRESETS
        assert POST_PROCESSOR_PRESETS["chinese-text-correction"]["framework"] == "llama-cpp"

    def test_qwen3_exists(self):
        assert "qwen3-0.6b" in POST_PROCESSOR_PRESETS
        assert POST_PROCESSOR_PRESETS["qwen3-0.6b"]["framework"] == "llama-cpp"

    def test_minicpm4_exists(self):
        assert "minicpm4-0.5b" in POST_PROCESSOR_PRESETS
        assert POST_PROCESSOR_PRESETS["minicpm4-0.5b"]["framework"] == "llama-cpp"


class TestHaikuFixPreset:
    """haiku-fix preset structure and config."""

    def test_exists(self):
        assert "haiku-fix" in POST_PROCESSOR_PRESETS

    def test_framework(self):
        assert POST_PROCESSOR_PRESETS["haiku-fix"]["framework"] == "ssh-claude"

    def test_has_name(self):
        assert "name" in POST_PROCESSOR_PRESETS["haiku-fix"]

    def test_has_description(self):
        assert "description" in POST_PROCESSOR_PRESETS["haiku-fix"]

    def test_config_ssh_host(self):
        config = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]
        assert config["ssh_host"] == "oracle-cloud"

    def test_config_claude_path(self):
        config = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]
        assert config["claude_path"] == "/home/ubuntu/.local/bin/claude"

    def test_config_model(self):
        config = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]
        assert config["model"] == "claude-haiku-4-5-20251001"

    def test_config_timeout_15s(self):
        """CRITIC-R1-C1: timeout must be 15s."""
        config = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]
        assert config["timeout"] == 15

    def test_config_vocab_min_count(self):
        config = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]
        assert config["vocab_min_count"] == 3

    def test_system_prompt_is_asr_tool(self):
        """System prompt must declare it's an ASR correction tool, NOT a chatbot."""
        prompt = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]["system_prompt"]
        assert "ASR" in prompt
        assert "NOT a chatbot" in prompt

    def test_system_prompt_fix_english_misrecognized(self):
        prompt = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]["system_prompt"]
        assert "English" in prompt and "misrecognized" in prompt

    def test_system_prompt_fix_homophones(self):
        prompt = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]["system_prompt"]
        assert "homophone" in prompt.lower() or "同音字" in prompt

    def test_system_prompt_remove_repeated(self):
        prompt = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]["system_prompt"]
        assert "repeated" in prompt.lower() or "重复" in prompt

    def test_system_prompt_output_only_corrected(self):
        prompt = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]["system_prompt"]
        assert "ONLY" in prompt and "corrected" in prompt

    def test_system_prompt_anti_answer(self):
        """CRITIC-R3-M1: must include anti-answer instructions."""
        prompt = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]["system_prompt"]
        assert "NEVER answer" in prompt or "never answer" in prompt.lower()
        assert "question" in prompt.lower()

    def test_system_prompt_unchanged_if_no_errors(self):
        prompt = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]["system_prompt"]
        assert "unchanged" in prompt.lower() or "no errors" in prompt.lower()


class TestHaikuExpandPreset:
    """haiku-expand preset (placeholder)."""

    def test_exists(self):
        assert "haiku-expand" in POST_PROCESSOR_PRESETS

    def test_framework(self):
        assert POST_PROCESSOR_PRESETS["haiku-expand"]["framework"] == "ssh-claude"

    def test_name(self):
        assert POST_PROCESSOR_PRESETS["haiku-expand"]["name"] == "Haiku Expand (placeholder)"

    def test_description(self):
        assert POST_PROCESSOR_PRESETS["haiku-expand"]["description"] == "Not yet implemented"

    def test_empty_config(self):
        assert POST_PROCESSOR_PRESETS["haiku-expand"]["config"] == {}
