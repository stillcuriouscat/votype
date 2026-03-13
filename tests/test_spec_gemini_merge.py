"""
Clean-room unit tests for process_with_gemini_merge() and gemini-merge preset.

Derived exclusively from FUNCTION_SPEC.md behavior tables.
Does NOT read implementation source code.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test configuration (from FUNCTION_SPEC.md §Test Configuration Constants)
# ---------------------------------------------------------------------------

MERGE_CONFIG = {
    "ssh_host": "oracle-cloud",
    "proxy_script": "~/vertex_proxy.py",
    "model": "gemini-2.5-flash",
    "vertex_region": "us-central1",
    "timeout": 15,
    "min_text_len": 45,
    "vocab_min_count": 3,
    "system_prompt": "You are a merge editor.",  # inline — avoids file I/O
}


# ---------------------------------------------------------------------------
# Helper: build a text of exact length
# ---------------------------------------------------------------------------

def _text(length: int, char: str = "a") -> str:
    return char * length


# ===========================================================================
# Module 3: process_with_gemini_merge — Behavior Table Tests
# ===========================================================================


class TestGeminiMergeDualSuccess:
    """BT#1: Normal dual merge success."""

    @patch("post_processor_configs.subprocess.run")
    def test_dual_merge_returns_stripped_stdout(self, mock_run):
        from post_processor_configs import process_with_gemini_merge

        mock_run.return_value = MagicMock(
            returncode=0, stdout="  merged output  ", stderr=""
        )
        primary = _text(50)
        secondary = _text(50, "b")

        result = process_with_gemini_merge(primary, secondary, MERGE_CONFIG)

        assert result == "merged output"

    @patch("post_processor_configs.subprocess.run")
    def test_dual_merge_user_input_format(self, mock_run):
        """BT#1 side-effect: user_input = 'Chinese ASR: …\\nEnglish ASR: …'"""
        from post_processor_configs import process_with_gemini_merge

        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok", stderr=""
        )
        primary = _text(50)
        secondary = "whisper text " + _text(37)

        process_with_gemini_merge(primary, secondary, MERGE_CONFIG)

        # Extract the JSON stdin payload
        call_kwargs = mock_run.call_args
        stdin_json = json.loads(call_kwargs.kwargs.get("input", call_kwargs[1].get("input", "")))
        user_input = stdin_json["user_input"]

        assert user_input.startswith("Chinese ASR: ")
        assert "\nEnglish ASR: " in user_input
        assert primary in user_input
        assert secondary in user_input


class TestGeminiMergeSingleFallback:
    """BT#2: Normal single fallback (secondary=None)."""

    @patch("post_processor_configs.subprocess.run")
    def test_single_fallback_returns_output(self, mock_run):
        from post_processor_configs import process_with_gemini_merge

        mock_run.return_value = MagicMock(
            returncode=0, stdout="polished text", stderr=""
        )
        primary = _text(50)

        result = process_with_gemini_merge(primary, None, MERGE_CONFIG)

        assert result == "polished text"

    @patch("post_processor_configs.subprocess.run")
    def test_single_fallback_user_input_has_no_english_asr(self, mock_run):
        """When secondary_text is None, user_input = 'Chinese ASR: …' only."""
        from post_processor_configs import process_with_gemini_merge

        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok", stderr=""
        )
        primary = _text(50)

        process_with_gemini_merge(primary, None, MERGE_CONFIG)

        call_kwargs = mock_run.call_args
        stdin_json = json.loads(call_kwargs.kwargs.get("input", call_kwargs[1].get("input", "")))
        user_input = stdin_json["user_input"]

        assert "Chinese ASR: " in user_input
        assert "English ASR:" not in user_input


class TestGeminiMergeGlossary:
    """BT#3, BT#4: Glossary appended / no glossary."""

    @patch("post_processor_configs.subprocess.run")
    def test_glossary_appended_to_system_prompt(self, mock_run):
        from post_processor_configs import process_with_gemini_merge

        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok", stderr=""
        )
        glossary = "Commonly used terms: Claude"
        primary = _text(50)

        process_with_gemini_merge(primary, None, MERGE_CONFIG, glossary_ctx=glossary)

        call_kwargs = mock_run.call_args
        stdin_json = json.loads(call_kwargs.kwargs.get("input", call_kwargs[1].get("input", "")))
        system_prompt = stdin_json["system_prompt"]

        assert system_prompt.endswith(glossary)

    @patch("post_processor_configs.subprocess.run")
    def test_no_glossary_no_suffix(self, mock_run):
        from post_processor_configs import process_with_gemini_merge

        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok", stderr=""
        )
        primary = _text(50)

        process_with_gemini_merge(primary, None, MERGE_CONFIG, glossary_ctx="")

        call_kwargs = mock_run.call_args
        stdin_json = json.loads(call_kwargs.kwargs.get("input", call_kwargs[1].get("input", "")))
        system_prompt = stdin_json["system_prompt"]

        # No glossary suffix — prompt should equal the inline config value
        assert "Commonly used terms" not in system_prompt


class TestGeminiMergeGuards:
    """BT#5–9, BT#12–15: Guard tests."""

    def test_empty_primary_returns_empty(self):
        """BT#5: empty primary → return '' without calling SSH."""
        from post_processor_configs import process_with_gemini_merge

        result = process_with_gemini_merge("", "anything", MERGE_CONFIG)
        assert result == ""

    def test_primary_below_min_len_returns_primary(self):
        """BT#6: primary below min_text_len → return primary."""
        from post_processor_configs import process_with_gemini_merge

        short = "short"
        assert len(short) < MERGE_CONFIG["min_text_len"]

        result = process_with_gemini_merge(short, "secondary", MERGE_CONFIG)
        assert result == short

    @patch("post_processor_configs.subprocess.run")
    def test_primary_at_exactly_min_len_calls_ssh(self, mock_run):
        """BT#9: primary at exactly min_text_len → SSH IS called."""
        from post_processor_configs import process_with_gemini_merge

        mock_run.return_value = MagicMock(
            returncode=0, stdout="output", stderr=""
        )
        exact = _text(45)
        assert len(exact) == MERGE_CONFIG["min_text_len"]

        result = process_with_gemini_merge(exact, None, MERGE_CONFIG)

        mock_run.assert_called_once()
        assert result == "output"

    @patch("post_processor_configs.subprocess.run")
    def test_hallucination_guard_returns_primary(self, mock_run):
        """BT#12: len(output) > len(primary) * 2 → return primary."""
        from post_processor_configs import process_with_gemini_merge

        primary = _text(50)
        # Output is > 2x primary length
        mock_run.return_value = MagicMock(
            returncode=0, stdout=_text(101), stderr=""
        )

        result = process_with_gemini_merge(primary, "secondary", MERGE_CONFIG)

        assert result == primary

    @patch("post_processor_configs.subprocess.run")
    def test_output_at_exactly_2x_accepted(self, mock_run):
        """BT#13: len(output) == len(primary) * 2 → accepted (not > 2x)."""
        from post_processor_configs import process_with_gemini_merge

        primary = _text(50)
        output_text = _text(100)  # exactly 2x
        mock_run.return_value = MagicMock(
            returncode=0, stdout=output_text, stderr=""
        )

        result = process_with_gemini_merge(primary, "secondary", MERGE_CONFIG)

        assert result == output_text

    @patch("post_processor_configs.subprocess.run")
    def test_question_guard_returns_primary(self, mock_run):
        """BT#14: '？' in primary and '？'/'?' not in output → return primary."""
        from post_processor_configs import process_with_gemini_merge

        primary = "这是一个问题吗？" + _text(37)  # contains ？, len >= min_text_len
        assert "？" in primary
        # Output has no question marks at all
        mock_run.return_value = MagicMock(
            returncode=0, stdout="这是一个问题" + _text(10), stderr=""
        )

        result = process_with_gemini_merge(primary, None, MERGE_CONFIG)

        assert result == primary

    @patch("post_processor_configs.subprocess.run")
    def test_question_guard_ascii_question_accepted(self, mock_run):
        """BT#15: '？' in primary and '?' in output (ASCII) → accepted."""
        from post_processor_configs import process_with_gemini_merge

        primary = "这是一个问题吗？" + _text(37)
        assert "？" in primary
        output = "is this a question?" + _text(10)
        assert "?" in output
        mock_run.return_value = MagicMock(
            returncode=0, stdout=output, stderr=""
        )

        result = process_with_gemini_merge(primary, None, MERGE_CONFIG)

        assert result == output


class TestGeminiMergeErrors:
    """BT#10, BT#11: Timeout and SSH error tests."""

    @patch("post_processor_configs.subprocess.run")
    def test_timeout_returns_primary(self, mock_run):
        """BT#10: SSH timeout → return primary_text."""
        from post_processor_configs import process_with_gemini_merge

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=15)
        primary = _text(50)

        result = process_with_gemini_merge(primary, "secondary", MERGE_CONFIG)

        assert result == primary

    @patch("post_processor_configs.subprocess.run")
    def test_ssh_non_zero_exit_returns_primary(self, mock_run):
        """BT#11: SSH non-zero exit → return primary_text."""
        from post_processor_configs import process_with_gemini_merge

        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Connection refused"
        )
        primary = _text(50)

        result = process_with_gemini_merge(primary, "secondary", MERGE_CONFIG)

        assert result == primary


class TestGeminiMergeSSHCommand:
    """Verify SSH command construction from spec."""

    @patch("post_processor_configs.subprocess.run")
    def test_ssh_command_structure(self, mock_run):
        from post_processor_configs import process_with_gemini_merge

        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok", stderr=""
        )
        primary = _text(50)

        process_with_gemini_merge(primary, None, MERGE_CONFIG)

        call_args = mock_run.call_args
        cmd = call_args[0][0] if call_args[0] else call_args.kwargs.get("args", [])
        assert "ssh" in cmd
        assert "-o" in cmd
        assert "ConnectTimeout=5" in cmd
        assert MERGE_CONFIG["ssh_host"] in cmd
        assert "python3" in cmd
        assert MERGE_CONFIG["proxy_script"] in cmd

    @patch("post_processor_configs.subprocess.run")
    def test_json_stdin_payload_fields(self, mock_run):
        """Verify JSON stdin has system_prompt, user_input, model, region."""
        from post_processor_configs import process_with_gemini_merge

        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok", stderr=""
        )
        primary = _text(50)

        process_with_gemini_merge(primary, "secondary", MERGE_CONFIG)

        call_kwargs = mock_run.call_args
        stdin_str = call_kwargs.kwargs.get("input", call_kwargs[1].get("input", ""))
        payload = json.loads(stdin_str)

        assert "system_prompt" in payload
        assert "user_input" in payload
        assert "model" in payload
        assert "region" in payload
        assert payload["model"] == "gemini-2.5-flash"
        assert payload["region"] == "us-central1"


# ===========================================================================
# Module 3: PostProcessorLoader.load_post_processor — Behavior Table Tests
# ===========================================================================


class TestPostProcessorLoaderGeminiMerge:
    """BT#1, #6, #7 for PostProcessorLoader.load_post_processor."""

    def test_gemini_merge_returns_none(self):
        """BT#1: gemini-merge → None (no local model needed)."""
        from post_processor_configs import PostProcessorLoader

        result = PostProcessorLoader.load_post_processor("gemini-merge")
        assert result is None

    def test_gemini_fix_returns_none(self):
        """BT#2: gemini-fix → None."""
        from post_processor_configs import PostProcessorLoader

        result = PostProcessorLoader.load_post_processor("gemini-fix")
        assert result is None

    def test_none_preset_returns_none(self):
        """BT#3: none → None."""
        from post_processor_configs import PostProcessorLoader

        result = PostProcessorLoader.load_post_processor("none")
        assert result is None

    def test_haiku_fix_returns_none(self):
        """BT#4: haiku-fix → None."""
        from post_processor_configs import PostProcessorLoader

        result = PostProcessorLoader.load_post_processor("haiku-fix")
        assert result is None

    def test_unknown_preset_raises_value_error(self):
        """BT#6: unknown preset → ValueError with exact message."""
        from post_processor_configs import PostProcessorLoader

        with pytest.raises(ValueError, match="Unknown post-processor: nonexistent"):
            PostProcessorLoader.load_post_processor("nonexistent")


# ===========================================================================
# Module 4: POST_PROCESSOR_PRESETS["gemini-merge"] — Data Contract Tests
# ===========================================================================


class TestGeminiMergePreset:
    """Verify gemini-merge preset structure from BT#1-13."""

    @pytest.fixture(autouse=True)
    def _load_presets(self):
        from post_processor_presets import POST_PROCESSOR_PRESETS
        self.presets = POST_PROCESSOR_PRESETS
        self.preset = POST_PROCESSOR_PRESETS["gemini-merge"]
        self.config = self.preset["config"]

    def test_preset_exists(self):
        assert "gemini-merge" in self.presets

    def test_name(self):
        """BT#1: name."""
        assert self.preset["name"] == "Gemini Merge (Dual ASR)"

    def test_description_mentions_fireredasr_and_faster_whisper(self):
        """BT#2: description contains FireRedASR and faster-whisper."""
        desc = self.preset["description"]
        assert "FireRedASR" in desc or "firered" in desc.lower()
        assert "faster-whisper" in desc or "Whisper" in desc

    def test_framework(self):
        """BT#3: framework is vertex-ai-merge."""
        assert self.preset["framework"] == "vertex-ai-merge"

    def test_ssh_host(self):
        """BT#4."""
        assert self.config["ssh_host"] == "oracle-cloud"

    def test_proxy_script(self):
        """BT#5."""
        assert self.config["proxy_script"] == "~/vertex_proxy.py"

    def test_model(self):
        """BT#6."""
        assert self.config["model"] == "gemini-2.5-flash"

    def test_vertex_region(self):
        """BT#7."""
        assert self.config["vertex_region"] == "us-central1"

    def test_timeout(self):
        """BT#8."""
        assert self.config["timeout"] == 15

    def test_min_text_len(self):
        """BT#9."""
        assert self.config["min_text_len"] == 45

    def test_vocab_min_count(self):
        """BT#11."""
        assert self.config["vocab_min_count"] == 3

    def test_system_prompt_file(self):
        """BT#12."""
        assert self.config["system_prompt_file"] == "prompts/gemini-merge-system.txt"

    def test_no_user_prompt_template_file(self):
        """BT#13: user_prompt_template_file key absent."""
        assert "user_prompt_template_file" not in self.config


# ===========================================================================
# Existing presets unchanged (regression)
# ===========================================================================


class TestExistingPresetsUnchanged:
    """Verify existing presets are not modified."""

    @pytest.fixture(autouse=True)
    def _load_presets(self):
        from post_processor_presets import POST_PROCESSOR_PRESETS
        self.presets = POST_PROCESSOR_PRESETS

    def test_none_preset_exists(self):
        assert "none" in self.presets

    def test_gemini_fix_preset_exists(self):
        assert "gemini-fix" in self.presets

    def test_gemini_fix_framework(self):
        assert self.presets["gemini-fix"]["framework"] == "vertex-ai"

    def test_haiku_fix_preset_exists(self):
        assert "haiku-fix" in self.presets
