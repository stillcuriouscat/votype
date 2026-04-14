"""
Clean-room integration tests: max_output_tokens cross-module contract.

Verifies the contract between post_processor_configs.py (sender) and
vertex_proxy.py (receiver) for the max_output_tokens JSON field.

Derived from LOW_LEVEL_DESIGN.md Section 2 (Inter-Module Contracts),
Section 3 (Data Models), and Section 6 (Test Contracts US-001/US-002).

Clean-room: these tests were written solely from the spec documents,
without reading the implementation source code.
"""

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import post_processor_configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_expected_max_output_tokens(user_input_len: int) -> int:
    """Reference formula from LOW_LEVEL_DESIGN.md Section 3."""
    return min(8192, max(512, user_input_len))


def _extract_stdin_payload(mock_run) -> dict | None:
    """Extract the first JSON payload containing 'user_input' from mock calls."""
    for c in mock_run.call_args_list:
        _, kwargs = c
        input_data = kwargs.get("input", "")
        if isinstance(input_data, str) and "user_input" in input_data:
            try:
                return json.loads(input_data)
            except json.JSONDecodeError:
                continue
    return None


def _extract_all_stdin_payloads(mock_run) -> list[dict]:
    """Extract all JSON payloads containing 'user_input' from mock calls."""
    payloads = []
    for c in mock_run.call_args_list:
        _, kwargs = c
        input_data = kwargs.get("input", "")
        if isinstance(input_data, str) and "user_input" in input_data:
            try:
                payloads.append(json.loads(input_data))
            except json.JSONDecodeError:
                continue
    return payloads


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vertex_ai_data_dir(tmp_path):
    """Temp VOICE_INPUT_DATA_DIR with required prompt files."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    (prompts_dir / "gemini-fix-system.txt").write_text(
        "You are a text editor. Fix the text."
    )
    (prompts_dir / "haiku-fix-user.txt").write_text(
        "输入：{text}\n输出："
    )
    (prompts_dir / "gemini-merge-system.txt").write_text(
        "You are a merge editor. Merge the texts."
    )

    return tmp_path


@pytest.fixture
def vertex_ai_config():
    """Config dict for process_with_vertex_ai (vertex-ai framework)."""
    return {
        "ssh_host": "test-host",
        "proxy_script": "~/vertex_proxy.py",
        "model": "gemini-2.5-flash",
        "fallback_model": "gemini-2.0-flash-lite",
        "vertex_region": "global",
        "timeout": 15,
        "min_text_len": 15,
        "vocab_min_count": 3,
        "system_prompt_file": "prompts/gemini-fix-system.txt",
        "user_prompt_template_file": "prompts/haiku-fix-user.txt",
    }


@pytest.fixture
def gemini_merge_config():
    """Config dict for process_with_gemini_merge (vertex-ai-merge framework)."""
    return {
        "ssh_host": "test-host",
        "proxy_script": "~/vertex_proxy.py",
        "model": "gemini-2.5-flash",
        "fallback_model": "gemini-2.0-flash-lite",
        "vertex_region": "global",
        "timeout": 15,
        "min_text_len": 15,
        "vocab_min_count": 3,
        "system_prompt_file": "prompts/gemini-merge-system.txt",
    }


def _make_successful_run(stdout_text: str = "polished text"):
    """Return a subprocess.CompletedProcess simulating vertex_proxy success."""
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout=stdout_text, stderr=""
    )


# ===========================================================================
# US-002 Contract: process_with_vertex_ai includes max_output_tokens
# ===========================================================================

class TestProcessWithVertexAIMaxOutputTokens:
    """process_with_vertex_ai must include max_output_tokens in stdin JSON."""

    def test_short_text_uses_floor(self, vertex_ai_data_dir, vertex_ai_config):
        """Short text → max_output_tokens = 512 (floor).

        Spec: Section 6, US-002, row 'Short text (50 chars)'.
        """
        text = "这是一段短文本" * 4  # ~28 chars, above min_text_len=15

        with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", vertex_ai_data_dir), \
             patch("subprocess.run", return_value=_make_successful_run(text)) as mock_run:
            post_processor_configs.process_with_vertex_ai(text, vertex_ai_config)

            payload = _extract_stdin_payload(mock_run)
            assert payload is not None, "subprocess.run not called with JSON input"
            assert "max_output_tokens" in payload
            # With template "输入：{text}\n输出：", user_input is ~35 chars → floor 512
            assert payload["max_output_tokens"] == 512

    def test_medium_text_passthrough(self, vertex_ai_data_dir, vertex_ai_config):
        """Medium text (1000 chars) → max_output_tokens = len(user_input).

        Spec: Section 6, US-002, row 'Medium text (1000 chars)'.
        """
        text = "中" * 1000

        with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", vertex_ai_data_dir), \
             patch("subprocess.run", return_value=_make_successful_run(text)) as mock_run:
            post_processor_configs.process_with_vertex_ai(text, vertex_ai_config)

            payload = _extract_stdin_payload(mock_run)
            assert payload is not None
            user_input = payload["user_input"]
            expected = compute_expected_max_output_tokens(len(user_input))
            assert payload["max_output_tokens"] == expected
            assert expected > 512, "Should be in pass-through range for 1000-char text"

    def test_long_text_uses_ceiling(self, vertex_ai_data_dir, vertex_ai_config):
        """Very long text (20000 chars) → max_output_tokens = 8192 (ceiling).

        Spec: Section 6, US-002, row 'Very long text (20000 chars)'.
        """
        text = "长" * 20000

        with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", vertex_ai_data_dir), \
             patch("subprocess.run", return_value=_make_successful_run(text)) as mock_run:
            post_processor_configs.process_with_vertex_ai(text, vertex_ai_config)

            payload = _extract_stdin_payload(mock_run)
            assert payload is not None
            assert payload["max_output_tokens"] == 8192

    def test_formula_uses_user_input_not_raw_text(
        self, vertex_ai_data_dir, vertex_ai_config
    ):
        """max_output_tokens is computed from len(user_input), not len(text).

        Spec: Section 6, US-002 note — 'user_input may differ from text if
        user_prompt_template_file wraps it.'
        Template: "输入：{text}\\n输出：" adds ~7 chars overhead.
        """
        text = "中" * 600  # 600 raw chars

        with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", vertex_ai_data_dir), \
             patch("subprocess.run", return_value=_make_successful_run(text)) as mock_run:
            post_processor_configs.process_with_vertex_ai(text, vertex_ai_config)

            payload = _extract_stdin_payload(mock_run)
            assert payload is not None
            user_input = payload["user_input"]
            assert len(user_input) >= len(text), (
                "user_input should be >= raw text due to template"
            )
            assert payload["max_output_tokens"] == compute_expected_max_output_tokens(
                len(user_input)
            )

    def test_json_schema_has_all_fields(self, vertex_ai_data_dir, vertex_ai_config):
        """JSON payload matches VertexProxyInput schema from spec Section 3.

        Required fields: system_prompt, user_input, model, region, max_output_tokens.
        """
        text = "测试文本" * 10

        with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", vertex_ai_data_dir), \
             patch("subprocess.run", return_value=_make_successful_run(text)) as mock_run:
            post_processor_configs.process_with_vertex_ai(text, vertex_ai_config)

            payload = _extract_stdin_payload(mock_run)
            assert payload is not None
            for field in ("system_prompt", "user_input", "model", "region", "max_output_tokens"):
                assert field in payload, f"Missing field '{field}' in JSON payload"
            assert isinstance(payload["system_prompt"], str)
            assert isinstance(payload["user_input"], str)
            assert isinstance(payload["model"], str)
            assert isinstance(payload["region"], str)
            assert isinstance(payload["max_output_tokens"], int)


# ===========================================================================
# US-002 Contract: process_with_gemini_merge includes max_output_tokens
# ===========================================================================

class TestProcessWithGeminiMergeMaxOutputTokens:
    """process_with_gemini_merge must include max_output_tokens in stdin JSON."""

    def test_dual_asr_short_text_floor(
        self, vertex_ai_data_dir, gemini_merge_config
    ):
        """Short dual ASR → max_output_tokens = 512 (floor).

        Spec: FUNCTION_SPEC.md, process_with_gemini_merge, row 1.
        """
        primary = "你好世界" * 5  # 20 chars
        secondary = "hello world " * 5  # 60 chars

        with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", vertex_ai_data_dir), \
             patch("subprocess.run", return_value=_make_successful_run(primary)) as mock_run:
            post_processor_configs.process_with_gemini_merge(
                primary, secondary, gemini_merge_config
            )

            payload = _extract_stdin_payload(mock_run)
            assert payload is not None, "subprocess.run not called with JSON input"
            assert "max_output_tokens" in payload
            assert payload["max_output_tokens"] == 512

    def test_dual_asr_real_incident_case(
        self, vertex_ai_data_dir, gemini_merge_config
    ):
        """Real incident: ~2250 primary + ~2373 secondary → passthrough.

        Spec: FUNCTION_SPEC.md, process_with_gemini_merge, row 2.
        user_input = "Chinese ASR: {p}\\nEnglish ASR: {s}" → ~4637 chars.
        """
        primary = "中" * 2250
        secondary = "x" * 2373

        with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", vertex_ai_data_dir), \
             patch("subprocess.run", return_value=_make_successful_run(primary)) as mock_run:
            post_processor_configs.process_with_gemini_merge(
                primary, secondary, gemini_merge_config
            )

            payload = _extract_stdin_payload(mock_run)
            assert payload is not None
            user_input = payload["user_input"]
            expected_ui = f"Chinese ASR: {primary}\nEnglish ASR: {secondary}"
            assert user_input == expected_ui, "Dual ASR user_input format mismatch"
            expected_tokens = compute_expected_max_output_tokens(len(expected_ui))
            assert payload["max_output_tokens"] == expected_tokens
            # ~4637 chars should be passthrough (512 < 4637 < 8192)
            assert 512 < expected_tokens < 8192

    def test_dual_asr_very_long_ceiling(
        self, vertex_ai_data_dir, gemini_merge_config
    ):
        """Very long dual ASR → max_output_tokens = 8192 (ceiling).

        Spec: FUNCTION_SPEC.md, process_with_gemini_merge, row 4.
        """
        primary = "中" * 5000
        secondary = "x" * 5000

        with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", vertex_ai_data_dir), \
             patch("subprocess.run", return_value=_make_successful_run(primary)) as mock_run:
            post_processor_configs.process_with_gemini_merge(
                primary, secondary, gemini_merge_config
            )

            payload = _extract_stdin_payload(mock_run)
            assert payload is not None
            assert payload["max_output_tokens"] == 8192

    def test_single_asr_secondary_none(
        self, vertex_ai_data_dir, gemini_merge_config
    ):
        """secondary=None → single ASR format, correct max_output_tokens.

        Spec: FUNCTION_SPEC.md, process_with_gemini_merge, row 3.
        user_input = "Chinese ASR: {primary}" only.
        """
        primary = "中" * 2250

        with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", vertex_ai_data_dir), \
             patch("subprocess.run", return_value=_make_successful_run(primary)) as mock_run:
            post_processor_configs.process_with_gemini_merge(
                primary, None, gemini_merge_config
            )

            payload = _extract_stdin_payload(mock_run)
            assert payload is not None
            user_input = payload["user_input"]
            assert "English ASR:" not in user_input
            assert "Chinese ASR:" in user_input
            expected_tokens = compute_expected_max_output_tokens(len(user_input))
            assert payload["max_output_tokens"] == expected_tokens

    def test_json_schema_has_all_fields(
        self, vertex_ai_data_dir, gemini_merge_config
    ):
        """JSON payload matches VertexProxyInput schema from spec Section 3."""
        primary = "测试文本" * 10
        secondary = "test text " * 10

        with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", vertex_ai_data_dir), \
             patch("subprocess.run", return_value=_make_successful_run(primary)) as mock_run:
            post_processor_configs.process_with_gemini_merge(
                primary, secondary, gemini_merge_config
            )

            payload = _extract_stdin_payload(mock_run)
            assert payload is not None
            for field in ("system_prompt", "user_input", "model", "region", "max_output_tokens"):
                assert field in payload, f"Missing field '{field}'"
            assert isinstance(payload["max_output_tokens"], int)


# ===========================================================================
# US-002 Contract: _run_vertex_proxy fallback preserves max_output_tokens
# ===========================================================================

class TestFallbackPreservesMaxOutputTokens:
    """_run_vertex_proxy 429 fallback must NOT modify max_output_tokens.

    Spec: LOW_LEVEL_DESIGN.md Section 2 — 'payload["model"] is replaced but
    payload["max_output_tokens"] is NOT'.
    FUNCTION_SPEC.md, _run_vertex_proxy, rows 2–3.
    """

    def test_429_retry_and_fallback_preserve_value(
        self, vertex_ai_data_dir, gemini_merge_config
    ):
        """All retry/fallback calls keep the same max_output_tokens."""
        primary = "中" * 2000
        secondary = "x" * 2000

        captured_payloads: list[dict] = []

        def mock_subprocess_run(*args, **kwargs):
            input_data = kwargs.get("input", "")
            if isinstance(input_data, str) and "user_input" in input_data:
                try:
                    captured_payloads.append(json.loads(input_data))
                except json.JSONDecodeError:
                    pass

            # First calls: 429 → triggers retry + fallback
            if len(captured_payloads) <= 2:
                return subprocess.CompletedProcess(
                    args=[], returncode=1,
                    stdout="", stderr="429 RESOURCE_EXHAUSTED"
                )
            # Final call: success
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout=primary, stderr=""
            )

        with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", vertex_ai_data_dir), \
             patch("subprocess.run", side_effect=mock_subprocess_run), \
             patch("time.sleep"):
            post_processor_configs.process_with_gemini_merge(
                primary, secondary, gemini_merge_config
            )

        assert len(captured_payloads) >= 2, (
            f"Expected >=2 subprocess calls, got {len(captured_payloads)}"
        )
        original_tokens = captured_payloads[0]["max_output_tokens"]
        for i, payload in enumerate(captured_payloads[1:], start=1):
            assert payload["max_output_tokens"] == original_tokens, (
                f"Call {i} changed max_output_tokens: "
                f"{payload['max_output_tokens']} != {original_tokens}"
            )

    def test_fallback_changes_model_not_tokens(
        self, vertex_ai_data_dir, gemini_merge_config
    ):
        """Fallback replaces model but keeps max_output_tokens unchanged.

        Spec: FUNCTION_SPEC.md, _run_vertex_proxy, row 3.
        """
        primary = "中" * 2000
        secondary = "x" * 2000

        captured_payloads: list[dict] = []

        def mock_subprocess_run(*args, **kwargs):
            input_data = kwargs.get("input", "")
            if isinstance(input_data, str) and "user_input" in input_data:
                try:
                    captured_payloads.append(json.loads(input_data))
                except json.JSONDecodeError:
                    pass

            if len(captured_payloads) <= 2:
                return subprocess.CompletedProcess(
                    args=[], returncode=1,
                    stdout="", stderr="429 RESOURCE_EXHAUSTED"
                )
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout=primary, stderr=""
            )

        with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", vertex_ai_data_dir), \
             patch("subprocess.run", side_effect=mock_subprocess_run), \
             patch("time.sleep"):
            post_processor_configs.process_with_gemini_merge(
                primary, secondary, gemini_merge_config
            )

        if len(captured_payloads) >= 3:
            original_model = captured_payloads[0]["model"]
            fallback_payload = captured_payloads[-1]
            # Model should have changed to fallback
            assert fallback_payload["model"] != original_model, (
                "Fallback should use a different model"
            )
            # But max_output_tokens must be preserved
            assert fallback_payload["max_output_tokens"] == captured_payloads[0]["max_output_tokens"]


# ===========================================================================
# US-001 Contract: backward compatibility (max_output_tokens absent)
# ===========================================================================

class TestBackwardCompatibility:
    """vertex_proxy.py defaults max_output_tokens to 512 when absent.

    Spec: Section 4 (Error Taxonomy), row 1 — 'max_output_tokens absent → 512'.
    Spec: Section 5 (Configuration Contract) — 'default 512'.

    Note: vertex_proxy.py runs on Oracle Cloud and is tested indirectly.
    This test verifies the contract from the sender side: the caller ALWAYS
    includes max_output_tokens (never absent), but the receiver must still
    handle absence gracefully.
    """

    def test_sender_always_includes_max_output_tokens(
        self, vertex_ai_data_dir, vertex_ai_config
    ):
        """Caller (post_processor_configs) always sends max_output_tokens.

        After US-002, the field is never absent in practice. This test
        documents that the sender side of the contract is always fulfilled.
        """
        text = "测试" * 20

        with patch("post_processor_configs.VOICE_INPUT_DATA_DIR", vertex_ai_data_dir), \
             patch("subprocess.run", return_value=_make_successful_run(text)) as mock_run:
            post_processor_configs.process_with_vertex_ai(text, vertex_ai_config)

            payload = _extract_stdin_payload(mock_run)
            assert payload is not None
            assert "max_output_tokens" in payload, (
                "Sender must always include max_output_tokens per US-002 contract"
            )
            assert isinstance(payload["max_output_tokens"], int)
            assert payload["max_output_tokens"] >= 512, (
                "max_output_tokens must be >= 512 (floor)"
            )
            assert payload["max_output_tokens"] <= 8192, (
                "max_output_tokens must be <= 8192 (ceiling)"
            )


# ===========================================================================
# Formula boundary value tests (pure logic, no module interaction)
# ===========================================================================

class TestMaxOutputTokensFormula:
    """Boundary value verification for min(8192, max(512, len(user_input))).

    Spec: LOW_LEVEL_DESIGN.md Section 3, 'max_output_tokens formula'.
    FUNCTION_SPEC.md, 'Helper: max_output_tokens formula'.
    """

    @pytest.mark.parametrize("input_len,expected", [
        (0, 512),       # empty (floor)
        (1, 512),       # single char (floor)
        (50, 512),      # well below floor
        (511, 512),     # just below floor
        (512, 512),     # exact floor
        (513, 513),     # just above floor
        (1000, 1000),   # medium passthrough
        (2000, 2000),   # medium passthrough
        (4637, 4637),   # real incident case
        (8191, 8191),   # just below ceiling
        (8192, 8192),   # exact ceiling
        (8193, 8192),   # just above ceiling
        (20000, 8192),  # well above ceiling
    ])
    def test_formula_boundary(self, input_len: int, expected: int):
        """Boundary values match spec examples."""
        result = compute_expected_max_output_tokens(input_len)
        assert result == expected, (
            f"Formula({input_len}) = {result}, expected {expected}"
        )
