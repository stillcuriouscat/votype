"""Clean-room integration tests for gemini-merge contracts.

Derived from LOW_LEVEL_DESIGN.md sections 1.3, 1.4, 1.6, 2.1, 2.2, 4.1.
Tests inter-module contracts, data model shapes, error taxonomy, and configuration.
Does NOT read implementation source code.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from post_processor_configs import (
    process_with_gemini_merge,
    PostProcessorLoader,
    PostProcessorInference,
)
from post_processor_presets import POST_PROCESSOR_PRESETS


# ---------------------------------------------------------------------------
# Shared test config (mirrors LLD Section 7.1)
# ---------------------------------------------------------------------------
MERGE_CONFIG = {
    "ssh_host": "oracle-cloud",
    "proxy_script": "~/vertex_proxy.py",
    "model": "gemini-2.5-flash",
    "vertex_region": "us-central1",
    "timeout": 15,
    "min_text_len": 15,
    "vocab_min_count": 3,
    "system_prompt_file": "prompts/gemini-merge-system.txt",
}

# Primary text long enough to pass min_text_len guard (>=15 chars)
PRIMARY_LONG = "This is a sufficiently long primary text for testing the gemini merge function flow"
SECONDARY_TEXT = "This is the secondary whisper transcription with English terms"
MERGED_OUTPUT = "This is a merged output combining both ASR results nicely"


def _make_ssh_success(stdout_text: str):
    """Helper: build a mock CompletedProcess for successful SSH call."""
    mock_result = MagicMock(spec=subprocess.CompletedProcess)
    mock_result.returncode = 0
    mock_result.stdout = stdout_text
    mock_result.stderr = ""
    return mock_result


@pytest.fixture
def mock_prompt_dir(tmp_path):
    """Create a temp directory with a fake prompt file, and patch VOICE_INPUT_DATA_DIR."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_file = prompts_dir / "gemini-merge-system.txt"
    prompt_file.write_text("You are a merge editor.", encoding="utf-8")

    with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", tmp_path):
        yield tmp_path


# ===========================================================================
# 1. Preset Structure Contracts (LLD 1.4, 5.1)
# ===========================================================================


class TestGeminiMergePreset:
    """Verify gemini-merge preset exists and matches documented structure."""

    def test_preset_exists(self):
        assert "gemini-merge" in POST_PROCESSOR_PRESETS

    def test_preset_framework_is_vertex_ai_merge(self):
        preset = POST_PROCESSOR_PRESETS["gemini-merge"]
        assert preset["framework"] == "vertex-ai-merge"

    def test_preset_has_name_and_description(self):
        preset = POST_PROCESSOR_PRESETS["gemini-merge"]
        assert "name" in preset
        assert "description" in preset
        assert isinstance(preset["name"], str)
        assert isinstance(preset["description"], str)

    def test_preset_config_required_keys(self):
        """LLD 5.1: all documented config keys must be present."""
        config = POST_PROCESSOR_PRESETS["gemini-merge"]["config"]
        required_keys = {
            "ssh_host",
            "proxy_script",
            "model",
            "vertex_region",
            "timeout",
            "min_text_len",
            "vocab_min_count",
            "system_prompt_file",
        }
        assert required_keys.issubset(set(config.keys())), (
            f"Missing keys: {required_keys - set(config.keys())}"
        )

    def test_preset_min_text_len_is_15(self):
        """LLD 1.4: min_text_len=15 (same as gemini-fix)."""
        config = POST_PROCESSOR_PRESETS["gemini-merge"]["config"]
        assert config["min_text_len"] == 15

    def test_preset_system_prompt_file_value(self):
        config = POST_PROCESSOR_PRESETS["gemini-merge"]["config"]
        assert config["system_prompt_file"] == "prompts/gemini-merge-system.txt"

    def test_loader_returns_none_for_vertex_ai_merge(self):
        """LLD 1.3: framework='vertex-ai-merge' returns None (no local model)."""
        result = PostProcessorLoader.load_post_processor("gemini-merge")
        assert result is None

    def test_existing_presets_unchanged(self):
        """Regression: existing presets still present."""
        for preset_id in ("none", "haiku-fix", "gemini-fix"):
            assert preset_id in POST_PROCESSOR_PRESETS, f"Missing preset: {preset_id}"


# ===========================================================================
# 2. process_with_gemini_merge Success Contracts (LLD 1.3, 2.1 row 7)
# ===========================================================================


class TestGeminiMergeSuccess:
    """Dual-input and single-input merge success cases."""

    @patch("post_processor_configs.subprocess.run")
    def test_success_returns_merged_text(self, mock_run, mock_prompt_dir):
        """LLD 2.1 row 7: returns str on success."""
        mock_run.return_value = _make_ssh_success(MERGED_OUTPUT)

        result = process_with_gemini_merge(
            PRIMARY_LONG, SECONDARY_TEXT, MERGE_CONFIG, glossary_ctx=""
        )
        assert isinstance(result, str)
        assert result == MERGED_OUTPUT

    @patch("post_processor_configs.subprocess.run")
    def test_user_input_format_dual(self, mock_run, mock_prompt_dir):
        """LLD 2.2: user_input = 'Chinese ASR: ...\\nEnglish ASR: ...' for dual input."""
        mock_run.return_value = _make_ssh_success(MERGED_OUTPUT)

        process_with_gemini_merge(PRIMARY_LONG, SECONDARY_TEXT, MERGE_CONFIG)

        # Inspect the stdin JSON sent to subprocess
        call_args = mock_run.call_args
        stdin_str = call_args.kwargs.get("input") or call_args[1].get("input", "")
        stdin_json = json.loads(stdin_str)
        user_input = stdin_json["user_input"]

        assert "Chinese ASR:" in user_input
        assert "English ASR:" in user_input
        assert PRIMARY_LONG in user_input
        assert SECONDARY_TEXT in user_input

    @patch("post_processor_configs.subprocess.run")
    def test_user_input_format_single_when_secondary_none(self, mock_run, mock_prompt_dir):
        """LLD 1.3: When secondary_text is None, user_input = 'Chinese ASR: ...' only."""
        mock_run.return_value = _make_ssh_success(PRIMARY_LONG)

        process_with_gemini_merge(PRIMARY_LONG, None, MERGE_CONFIG)

        call_args = mock_run.call_args
        stdin_str = call_args.kwargs.get("input") or call_args[1].get("input", "")
        stdin_json = json.loads(stdin_str)
        user_input = stdin_json["user_input"]

        assert "Chinese ASR:" in user_input
        assert "English ASR:" not in user_input

    @patch("post_processor_configs.subprocess.run")
    def test_glossary_context_appended_to_system_prompt(self, mock_run, mock_prompt_dir):
        """LLD 1.3: glossary_ctx appended to system prompt."""
        mock_run.return_value = _make_ssh_success(MERGED_OUTPUT)
        glossary = "Claude -> 克劳德"

        process_with_gemini_merge(PRIMARY_LONG, SECONDARY_TEXT, MERGE_CONFIG, glossary_ctx=glossary)

        call_args = mock_run.call_args
        stdin_str = call_args.kwargs.get("input") or call_args[1].get("input", "")
        stdin_json = json.loads(stdin_str)
        system_prompt = stdin_json["system_prompt"]

        assert glossary in system_prompt

    @patch("post_processor_configs.subprocess.run")
    def test_ssh_payload_contains_model_and_region(self, mock_run, mock_prompt_dir):
        """LLD 3.2 VertexProxyStdinPayload: must include model and region."""
        mock_run.return_value = _make_ssh_success(MERGED_OUTPUT)

        process_with_gemini_merge(PRIMARY_LONG, SECONDARY_TEXT, MERGE_CONFIG)

        call_args = mock_run.call_args
        stdin_str = call_args.kwargs.get("input") or call_args[1].get("input", "")
        stdin_json = json.loads(stdin_str)

        assert stdin_json["model"] == "gemini-2.5-flash"
        assert stdin_json["region"] == "us-central1"


# ===========================================================================
# 3. Guard Contracts (LLD 1.3 Guards section)
# ===========================================================================


class TestGeminiMergeGuards:
    """Guard tests: empty, min/max len, hallucination, question mark."""

    def test_empty_primary_returns_empty(self):
        """LLD: Empty primary_text -> return ''."""
        result = process_with_gemini_merge("", SECONDARY_TEXT, MERGE_CONFIG)
        assert result == ""

    def test_primary_below_min_len_returns_primary(self):
        """LLD: len(primary_text) < min_text_len -> return primary_text."""
        short_text = "short"  # well below 45
        result = process_with_gemini_merge(short_text, SECONDARY_TEXT, MERGE_CONFIG)
        assert result == short_text

    @patch("post_processor_configs.subprocess.run")
    def test_hallucination_guard_returns_primary(self, mock_run, mock_prompt_dir):
        """LLD: len(output) > len(primary_text) * 2 -> return primary_text."""
        hallucinated = PRIMARY_LONG * 3
        mock_run.return_value = _make_ssh_success(hallucinated)

        result = process_with_gemini_merge(PRIMARY_LONG, SECONDARY_TEXT, MERGE_CONFIG)
        assert result == PRIMARY_LONG

    @patch("post_processor_configs.subprocess.run")
    def test_output_within_2x_accepted(self, mock_run, mock_prompt_dir):
        """LLD: output within 2x length is accepted."""
        acceptable = PRIMARY_LONG + " extra"
        mock_run.return_value = _make_ssh_success(acceptable)

        result = process_with_gemini_merge(PRIMARY_LONG, SECONDARY_TEXT, MERGE_CONFIG)
        assert result == acceptable

    @patch("post_processor_configs.subprocess.run")
    def test_question_guard_returns_primary(self, mock_run, mock_prompt_dir):
        """LLD: fullwidth ? in primary and no ? in output -> return primary_text."""
        question_text = "x" * 50 + "\u8fd9\u4e2a\u600e\u4e48\u7528\uff1f"
        no_question_output = "\u8fd9\u4e2a\u600e\u4e48\u7528\u7684\u8bf4\u660e"
        mock_run.return_value = _make_ssh_success(no_question_output)

        result = process_with_gemini_merge(question_text, SECONDARY_TEXT, MERGE_CONFIG)
        assert result == question_text


# ===========================================================================
# 4. Error Contracts (LLD 4.1)
# ===========================================================================


class TestGeminiMergeErrors:
    """Timeout and SSH error tests."""

    @patch("post_processor_configs.subprocess.run")
    def test_timeout_returns_primary(self, mock_run, mock_prompt_dir):
        """LLD 4.1: subprocess.TimeoutExpired -> return primary_text."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=15)

        result = process_with_gemini_merge(PRIMARY_LONG, SECONDARY_TEXT, MERGE_CONFIG)
        assert result == PRIMARY_LONG

    @patch("post_processor_configs.subprocess.run")
    def test_ssh_error_returns_primary(self, mock_run, mock_prompt_dir):
        """LLD 4.1: SSH non-zero exit -> return primary_text."""
        error_result = MagicMock(spec=subprocess.CompletedProcess)
        error_result.returncode = 1
        error_result.stdout = ""
        error_result.stderr = "Connection refused"
        mock_run.return_value = error_result

        result = process_with_gemini_merge(PRIMARY_LONG, SECONDARY_TEXT, MERGE_CONFIG)
        assert result == PRIMARY_LONG

    @patch("post_processor_configs.subprocess.run")
    def test_never_raises_returns_primary_on_any_failure(self, mock_run, mock_prompt_dir):
        """LLD 2.1 row 7: Never raises, returns primary_text on all failures."""
        # Use Exception (broadest) — spec says all exceptions caught
        mock_run.side_effect = Exception("Unexpected error")

        # Per spec: should return primary_text, never raise
        try:
            result = process_with_gemini_merge(PRIMARY_LONG, SECONDARY_TEXT, MERGE_CONFIG)
            assert result == PRIMARY_LONG
        except Exception:
            pytest.fail("process_with_gemini_merge raised an exception; spec says it should never raise")


# ===========================================================================
# 5. Signature Difference Contract (LLD 2.2)
# ===========================================================================


class TestSignatureDifference:
    """Verify gemini-merge has 4-arg signature vs gemini-fix 3-arg."""

    def test_gemini_merge_accepts_four_args(self):
        """LLD 2.2: process_with_gemini_merge takes 4 parameters."""
        import inspect

        sig = inspect.signature(process_with_gemini_merge)
        params = list(sig.parameters.keys())
        assert len(params) == 4
        assert params[0] == "primary_text"
        assert params[1] == "secondary_text"
        assert params[2] == "config"
        assert params[3] == "glossary_ctx"

    def test_gemini_fix_accepts_three_args(self):
        """LLD 2.2: process_with_vertex_ai takes 3 parameters (regression)."""
        import inspect
        from post_processor_configs import process_with_vertex_ai

        sig = inspect.signature(process_with_vertex_ai)
        params = list(sig.parameters.keys())
        assert len(params) == 3
        assert params[0] == "text"


# ===========================================================================
# 6. Prompt File Contract (LLD 1.6)
# ===========================================================================


class TestMergePromptFile:
    """Verify gemini-merge-system.txt content contracts."""

    @pytest.fixture
    def prompt_text(self):
        """Load the prompt file from the project's prompts/ directory."""
        project_root = Path(__file__).parent.parent
        prompt_path = project_root / "prompts" / "gemini-merge-system.txt"
        if not prompt_path.exists():
            pytest.skip("prompts/gemini-merge-system.txt not found in project")
        return prompt_path.read_text(encoding="utf-8")

    def test_prompt_file_exists(self):
        project_root = Path(__file__).parent.parent
        prompt_path = project_root / "prompts" / "gemini-merge-system.txt"
        assert prompt_path.exists(), "prompts/gemini-merge-system.txt must exist"

    def test_prompt_is_dual_purpose(self, prompt_text):
        """LLD 1.6 constraint 1: handles both merge and single-input."""
        assert any(
            phrase in prompt_text
            for phrase in [
                "\u53ea\u63d0\u4f9b\u4e86\u4e00\u4e2a",  # 只提供了一个
                "\u5355\u6587\u672c",  # 单文本
                "one transcription",
                "single",
                "\u6ca1\u6709 English ASR",  # 没有 English ASR
            ]
        ), "Prompt must mention single-input fallback handling"

    def test_prompt_mentions_editor_identity(self, prompt_text):
        """LLD 1.6 constraint 3: 'editor not assistant' identity."""
        assert "\u7f16\u8f91" in prompt_text or "editor" in prompt_text.lower()

    def test_prompt_output_format_clean_text(self, prompt_text):
        """LLD 1.6 constraint 5: output format is clean text only."""
        lower = prompt_text.lower()
        assert any(
            phrase in prompt_text
            for phrase in [
                "\u4e0d\u8981\u89e3\u91ca",  # 不要解释
                "\u53ea\u8f93\u51fa",  # 只输出
                "\u4e0d\u56de\u7b54",  # 不回答
                "no explanation",
                "clean text only",
            ]
        ), "Prompt must instruct clean text output without explanations"
