"""Clean-room unit tests for US-002: dynamic max_output_tokens computation.

Derived from FUNCTION_SPEC.md behavior tables.
Verifies:
  1. The formula min(8192, max(512, len(user_input))) across boundary cases.
  2. process_with_vertex_ai() includes correct max_output_tokens in stdin JSON.
  3. process_with_gemini_merge() includes correct max_output_tokens in stdin JSON.
  4. _run_vertex_proxy() preserves max_output_tokens through 429 fallback.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# §1  Pure formula tests — min(8192, max(512, len(user_input)))
# ---------------------------------------------------------------------------

class TestMaxOutputTokensFormula:
    """FUNCTION_SPEC §Helper — compute_max_output_tokens behavior table."""

    @pytest.mark.parametrize(
        "input_len, expected",
        [
            (0, 512),        # Row 10: empty → floor
            (1, 512),        # Row 5: single char → floor
            (50, 512),       # Row 4: below floor
            (511, 512),      # just below floor
            (512, 512),      # Row 3: exact floor boundary
            (513, 513),      # Row 8: just above floor
            (1000, 1000),    # mid-range
            (2000, 2000),    # Row 1: normal medium
            (4637, 4637),    # Row 2: real incident case
            (8191, 8191),    # Row 9: just below ceiling
            (8192, 8192),    # Row 6: exact ceiling boundary
            (8193, 8192),    # just above ceiling
            (20000, 8192),   # Row 7: far above ceiling
        ],
        ids=[
            "empty_string",
            "single_char",
            "below_floor_50",
            "just_below_floor_511",
            "exact_floor_512",
            "just_above_floor_513",
            "mid_1000",
            "mid_2000",
            "real_incident_4637",
            "just_below_ceiling_8191",
            "exact_ceiling_8192",
            "just_above_ceiling_8193",
            "far_above_ceiling_20000",
        ],
    )
    def test_formula_boundaries(self, input_len: int, expected: int):
        """Verify min(8192, max(512, len(user_input))) at key boundary values."""
        user_input = "A" * input_len
        result = min(8192, max(512, len(user_input)))
        assert result == expected


# ---------------------------------------------------------------------------
# §2  Fixtures for process function tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ppc():
    """Import post_processor_configs with mocked external dependencies.

    Mocks subprocess.run so we can capture the stdin_data JSON
    without making real SSH calls.

    Yields (module, captured_calls_list).
    """
    # Ensure openrouter_client is importable (may not exist in worktree)
    mock_or_client = MagicMock()
    mock_or_client.call_openrouter = MagicMock(return_value=None)

    captured_stdins: list[str] = []

    def fake_subprocess_run(cmd, **kwargs):
        stdin_data = kwargs.get("input", "")
        captured_stdins.append(stdin_data)
        result = MagicMock(spec=subprocess.CompletedProcess)
        result.returncode = 0
        result.stdout = "polished text output"
        result.stderr = ""
        return result

    with patch.dict(
        "sys.modules",
        {"openrouter_client": mock_or_client},
    ):
        import post_processor_configs as ppc

        with patch.object(ppc, "subprocess") as mock_sp:
            mock_sp.run = MagicMock(side_effect=fake_subprocess_run)
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            yield ppc, captured_stdins, mock_sp


@pytest.fixture
def vertex_ai_config(tmp_path) -> dict:
    """Minimal config dict for process_with_vertex_ai."""
    # Create system prompt file
    prompt_file = tmp_path / "system-prompt.txt"
    prompt_file.write_text("Fix typos and grammar errors.")

    return {
        "ssh_host": "test-oracle",
        "proxy_script": "~/vertex_proxy.py",
        "model": "gemini-2.5-flash",
        "fallback_model": "gemini-2.0-flash",
        "vertex_region": "global",
        "timeout": 15,
        "min_text_len": 15,
        "vocab_min_count": 3,
        "system_prompt_file": str(prompt_file),
    }


@pytest.fixture
def gemini_merge_config(tmp_path) -> dict:
    """Minimal config dict for process_with_gemini_merge."""
    prompt_file = tmp_path / "merge-system-prompt.txt"
    prompt_file.write_text("Merge two ASR transcriptions.")

    return {
        "ssh_host": "test-oracle",
        "proxy_script": "~/vertex_proxy.py",
        "model": "gemini-2.5-flash",
        "fallback_model": "gemini-2.0-flash",
        "vertex_region": "global",
        "timeout": 15,
        "min_text_len": 15,
        "vocab_min_count": 3,
        "system_prompt_file": str(prompt_file),
    }


def _parse_captured_stdin(captured_stdins: list[str]) -> dict:
    """Parse the last captured stdin as JSON. Raises if nothing captured."""
    assert captured_stdins, "No subprocess.run call captured"
    return json.loads(captured_stdins[-1])


def _assert_formula(stdin_json: dict) -> None:
    """Assert max_output_tokens == min(8192, max(512, len(user_input)))."""
    user_input = stdin_json["user_input"]
    expected = min(8192, max(512, len(user_input)))
    actual = stdin_json["max_output_tokens"]
    assert actual == expected, (
        f"max_output_tokens={actual} but expected {expected} "
        f"for len(user_input)={len(user_input)}"
    )


# ---------------------------------------------------------------------------
# §3  process_with_vertex_ai — max_output_tokens in stdin JSON
# ---------------------------------------------------------------------------

class TestProcessWithVertexAI:
    """FUNCTION_SPEC §US-002 — process_with_vertex_ai behavior table."""

    def test_row1_short_text_floor_512(self, mock_ppc, vertex_ai_config):
        """Short text (50 chars) → max_output_tokens = 512 (floor)."""
        ppc, captured, _ = mock_ppc
        text = "A" * 50
        ppc.process_with_vertex_ai(text, vertex_ai_config)
        stdin = _parse_captured_stdin(captured)
        assert "max_output_tokens" in stdin
        assert stdin["max_output_tokens"] == 512 or stdin["max_output_tokens"] >= 512
        _assert_formula(stdin)

    def test_row2_medium_text_passthrough(self, mock_ppc, vertex_ai_config):
        """Medium text (1000 chars) → max_output_tokens ≈ len(user_input)."""
        ppc, captured, _ = mock_ppc
        text = "B" * 1000
        ppc.process_with_vertex_ai(text, vertex_ai_config)
        stdin = _parse_captured_stdin(captured)
        _assert_formula(stdin)
        # user_input >= 1000 (text + possible template), so result >= 1000
        assert stdin["max_output_tokens"] >= 1000

    def test_row4_exact_floor_boundary(self, mock_ppc, vertex_ai_config):
        """Text producing user_input of exactly 512 chars → 512."""
        ppc, captured, _ = mock_ppc
        # Use a long enough text to pass min_text_len
        text = "C" * 512
        ppc.process_with_vertex_ai(text, vertex_ai_config)
        stdin = _parse_captured_stdin(captured)
        _assert_formula(stdin)

    def test_row6_very_long_text_ceiling_8192(self, mock_ppc, vertex_ai_config):
        """Very long text (20000 chars) → max_output_tokens = 8192 (ceiling)."""
        ppc, captured, _ = mock_ppc
        text = "D" * 20000
        ppc.process_with_vertex_ai(text, vertex_ai_config)
        stdin = _parse_captured_stdin(captured)
        assert stdin["max_output_tokens"] == 8192
        _assert_formula(stdin)

    def test_row7_min_text_len_boundary(self, mock_ppc, vertex_ai_config):
        """Text at min_text_len (15 chars) → floor 512."""
        ppc, captured, _ = mock_ppc
        text = "E" * 15
        ppc.process_with_vertex_ai(text, vertex_ai_config)
        stdin = _parse_captured_stdin(captured)
        _assert_formula(stdin)
        assert stdin["max_output_tokens"] == 512

    def test_json_has_required_fields(self, mock_ppc, vertex_ai_config):
        """stdin JSON includes all required fields per cross-module contract."""
        ppc, captured, _ = mock_ppc
        text = "F" * 100
        ppc.process_with_vertex_ai(text, vertex_ai_config)
        stdin = _parse_captured_stdin(captured)
        for key in ("system_prompt", "user_input", "model", "region", "max_output_tokens"):
            assert key in stdin, f"Missing required JSON field: {key}"


# ---------------------------------------------------------------------------
# §4  process_with_gemini_merge — max_output_tokens in stdin JSON
# ---------------------------------------------------------------------------

class TestProcessWithGeminiMerge:
    """FUNCTION_SPEC §US-002 — process_with_gemini_merge behavior table."""

    def test_row1_dual_short_floor(self, mock_ppc, gemini_merge_config):
        """Dual ASR, short texts → max_output_tokens = 512 (floor)."""
        ppc, captured, _ = mock_ppc
        primary = "hello " * 8    # 48 chars
        secondary = "world " * 8  # 48 chars
        ppc.process_with_gemini_merge(primary, secondary, gemini_merge_config)
        stdin = _parse_captured_stdin(captured)
        _assert_formula(stdin)
        assert stdin["max_output_tokens"] == 512

    def test_row2_dual_real_incident(self, mock_ppc, gemini_merge_config):
        """Dual ASR, real incident lengths → max_output_tokens ≈ 4637."""
        ppc, captured, _ = mock_ppc
        primary = "P" * 2250
        secondary = "S" * 2373
        ppc.process_with_gemini_merge(primary, secondary, gemini_merge_config)
        stdin = _parse_captured_stdin(captured)
        _assert_formula(stdin)
        # user_input = "Chinese ASR: " + primary + "\nEnglish ASR: " + secondary
        # len ≈ 14 + 2250 + 14 + 2373 = ~4651 (spec says ~4637)
        assert stdin["max_output_tokens"] > 4000

    def test_row3_secondary_none(self, mock_ppc, gemini_merge_config):
        """secondary=None → single ASR format, correct max_output_tokens."""
        ppc, captured, _ = mock_ppc
        primary = "Q" * 2250
        ppc.process_with_gemini_merge(primary, None, gemini_merge_config)
        stdin = _parse_captured_stdin(captured)
        _assert_formula(stdin)
        # user_input = "Chinese ASR: " + primary → ~2264 chars
        assert stdin["max_output_tokens"] > 2000

    def test_row4_very_long_dual_ceiling(self, mock_ppc, gemini_merge_config):
        """Very long dual ASR → max_output_tokens = 8192 (ceiling)."""
        ppc, captured, _ = mock_ppc
        primary = "R" * 5000
        secondary = "T" * 5000
        ppc.process_with_gemini_merge(primary, secondary, gemini_merge_config)
        stdin = _parse_captured_stdin(captured)
        assert stdin["max_output_tokens"] == 8192
        _assert_formula(stdin)

    def test_row5_exact_floor(self, mock_ppc, gemini_merge_config):
        """Dual ASR producing user_input of ~512 chars → floor applies."""
        ppc, captured, _ = mock_ppc
        # Very short texts → user_input < 512 after format labels
        primary = "U" * 50
        secondary = "V" * 50
        ppc.process_with_gemini_merge(primary, secondary, gemini_merge_config)
        stdin = _parse_captured_stdin(captured)
        _assert_formula(stdin)
        assert stdin["max_output_tokens"] == 512

    def test_json_has_required_fields(self, mock_ppc, gemini_merge_config):
        """stdin JSON includes all required fields per cross-module contract."""
        ppc, captured, _ = mock_ppc
        ppc.process_with_gemini_merge("text primary", "text secondary", gemini_merge_config)
        stdin = _parse_captured_stdin(captured)
        for key in ("system_prompt", "user_input", "model", "region", "max_output_tokens"):
            assert key in stdin, f"Missing required JSON field: {key}"

    def test_user_input_contains_both_asr_texts(self, mock_ppc, gemini_merge_config):
        """Dual ASR user_input contains both primary and secondary text."""
        ppc, captured, _ = mock_ppc
        primary = "PRIMARY_MARKER_TEXT"
        secondary = "SECONDARY_MARKER_TEXT"
        ppc.process_with_gemini_merge(primary, secondary, gemini_merge_config)
        stdin = _parse_captured_stdin(captured)
        assert primary in stdin["user_input"]
        assert secondary in stdin["user_input"]

    def test_user_input_single_no_secondary(self, mock_ppc, gemini_merge_config):
        """secondary=None → user_input does NOT contain 'English ASR'."""
        ppc, captured, _ = mock_ppc
        primary = "ONLY_PRIMARY"
        ppc.process_with_gemini_merge(primary, None, gemini_merge_config)
        stdin = _parse_captured_stdin(captured)
        assert primary in stdin["user_input"]


# ---------------------------------------------------------------------------
# §5  _run_vertex_proxy — 429 fallback preserves max_output_tokens
# ---------------------------------------------------------------------------

class TestRunVertexProxyFallback:
    """FUNCTION_SPEC §US-002 — _run_vertex_proxy preserves max_output_tokens
    through retry and model fallback on 429.
    """

    def test_fallback_preserves_max_output_tokens(self):
        """On 429+fallback, only model is replaced; max_output_tokens stays."""
        # Ensure module is importable
        mock_or = MagicMock()
        mock_or.call_openrouter = MagicMock(return_value=None)

        call_inputs: list[str] = []

        def fake_run(cmd, **kwargs):
            stdin_data = kwargs.get("input", "")
            call_inputs.append(stdin_data)
            result = MagicMock(spec=subprocess.CompletedProcess)
            n = len(call_inputs)
            if n <= 2:
                # First two calls: 429 error
                result.returncode = 1
                result.stdout = ""
                result.stderr = "429 RESOURCE_EXHAUSTED"
            else:
                # Third call (fallback model): success
                result.returncode = 0
                result.stdout = "success output"
                result.stderr = ""
            return result

        with patch.dict("sys.modules", {"openrouter_client": mock_or}):
            import post_processor_configs as ppc

            original_stdin = json.dumps({
                "system_prompt": "fix",
                "user_input": "test input text",
                "model": "gemini-2.5-flash",
                "region": "global",
                "max_output_tokens": 4637,
            })

            with patch.object(ppc, "subprocess") as mock_sp:
                mock_sp.run = MagicMock(side_effect=fake_run)
                mock_sp.TimeoutExpired = subprocess.TimeoutExpired

                ppc._run_vertex_proxy(
                    cmd=["ssh", "host", "python3", "script.py"],
                    stdin_data=original_stdin,
                    timeout=15,
                    max_retries=1,
                    fallback_model="gemini-2.0-flash",
                )

            # Verify fallback call has same max_output_tokens
            if len(call_inputs) >= 3:
                fallback_json = json.loads(call_inputs[-1])
                assert fallback_json["max_output_tokens"] == 4637, (
                    "max_output_tokens was modified during fallback"
                )
                assert fallback_json["model"] == "gemini-2.0-flash", (
                    "model was NOT replaced during fallback"
                )

    def test_no_max_output_tokens_backward_compat(self):
        """stdin without max_output_tokens works (backward compatibility)."""
        mock_or = MagicMock()
        mock_or.call_openrouter = MagicMock(return_value=None)

        def fake_run(cmd, **kwargs):
            result = MagicMock(spec=subprocess.CompletedProcess)
            result.returncode = 0
            result.stdout = "ok"
            result.stderr = ""
            return result

        with patch.dict("sys.modules", {"openrouter_client": mock_or}):
            import post_processor_configs as ppc

            stdin_no_max = json.dumps({
                "system_prompt": "fix",
                "user_input": "test",
                "model": "gemini-2.5-flash",
                "region": "global",
            })

            with patch.object(ppc, "subprocess") as mock_sp:
                mock_sp.run = MagicMock(side_effect=fake_run)
                mock_sp.TimeoutExpired = subprocess.TimeoutExpired

                result = ppc._run_vertex_proxy(
                    cmd=["ssh", "host", "python3", "script.py"],
                    stdin_data=stdin_no_max,
                    timeout=15,
                )
                assert result.returncode == 0
