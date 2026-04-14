"""Clean-room unit tests for US-001: vertex_proxy.py max_output_tokens.

Derived from FUNCTION_SPEC.md behavior tables.
Verifies that vertex_proxy.main() reads max_output_tokens from stdin JSON
and passes it to GenerateContentConfig, defaulting to 512 when absent.
Also verifies print_help() documents the new field.
"""

import io
import json
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vertex_env():
    """Import vertex_proxy with mocked google.genai SDK.

    Yields (vertex_proxy_module, mock_genai) so tests can inspect
    how GenerateContentConfig was called.
    """
    mock_genai = MagicMock()
    # Happy-path Gemini response
    mock_response = MagicMock()
    mock_response.text = "corrected output"
    mock_genai.Client.return_value.models.generate_content.return_value = (
        mock_response
    )

    mock_google = MagicMock()
    mock_google.genai = mock_genai

    saved_module = sys.modules.pop("vertex_proxy", None)

    with patch.dict(
        "sys.modules",
        {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_genai.types,
        },
    ):
        import vertex_proxy as vp

        yield vp, mock_genai

    # Cleanup — avoid polluting other test modules
    sys.modules.pop("vertex_proxy", None)
    if saved_module is not None:
        sys.modules["vertex_proxy"] = saved_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(**overrides) -> str:
    """Build valid stdin JSON for vertex_proxy.py main()."""
    data = {
        "system_prompt": "Fix typos and grammar",
        "user_input": "hello world",
        "model": "gemini-2.5-flash",
        "region": "global",
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def _run_main(vp_module, stdin_json: str) -> None:
    """Execute vertex_proxy.main() with given stdin, suppressing stdout/stderr."""
    with (
        patch("sys.stdin", io.StringIO(stdin_json)),
        patch("sys.stdout", new_callable=io.StringIO),
        patch("sys.stderr", new_callable=io.StringIO),
        patch("sys.argv", ["vertex_proxy.py"]),
    ):
        try:
            vp_module.main()
        except SystemExit:
            pass


def _get_config_kwargs(vertex_env, stdin_json: str) -> dict:
    """Run main() and return the kwargs passed to GenerateContentConfig."""
    vp, mock_genai = vertex_env
    _run_main(vp, stdin_json)
    config_cls = mock_genai.types.GenerateContentConfig
    assert config_cls.called, "GenerateContentConfig was never called"
    return config_cls.call_args.kwargs


# ===========================================================================
# Behavior Table: vertex_proxy.main() reads max_output_tokens
# ===========================================================================

class TestMainMaxOutputTokens:
    """FUNCTION_SPEC §US-001 — main() behavior table rows 1-5."""

    def test_row1_absent_defaults_to_512(self, vertex_env):
        """max_output_tokens absent → GenerateContentConfig gets 512."""
        kwargs = _get_config_kwargs(vertex_env, _make_input())
        assert kwargs["max_output_tokens"] == 512

    def test_row2_explicit_2048(self, vertex_env):
        """max_output_tokens=2048 → GenerateContentConfig gets 2048."""
        kwargs = _get_config_kwargs(
            vertex_env, _make_input(max_output_tokens=2048)
        )
        assert kwargs["max_output_tokens"] == 2048

    def test_row3_explicit_512_identical_to_absent(self, vertex_env):
        """max_output_tokens=512 (explicit) → same as default."""
        kwargs = _get_config_kwargs(
            vertex_env, _make_input(max_output_tokens=512)
        )
        assert kwargs["max_output_tokens"] == 512

    def test_row4_explicit_8192(self, vertex_env):
        """max_output_tokens=8192 → GenerateContentConfig gets 8192."""
        kwargs = _get_config_kwargs(
            vertex_env, _make_input(max_output_tokens=8192)
        )
        assert kwargs["max_output_tokens"] == 8192

    def test_row5_edge_min_value_1(self, vertex_env):
        """max_output_tokens=1 → passed through without clamping."""
        kwargs = _get_config_kwargs(
            vertex_env, _make_input(max_output_tokens=1)
        )
        assert kwargs["max_output_tokens"] == 1


class TestMainMaxOutputTokensErrorCase:
    """FUNCTION_SPEC §US-001 — main() behavior table row 6 (error)."""

    def test_row6_string_value_causes_exit_1(self, vertex_env):
        """max_output_tokens="abc" → Gemini SDK error → exit 1.

        When a non-integer max_output_tokens reaches GenerateContentConfig,
        the SDK raises TypeError, caught by existing try/except → exit 1.
        """
        vp, mock_genai = vertex_env
        # Make GenerateContentConfig raise TypeError for non-int
        mock_genai.types.GenerateContentConfig.side_effect = TypeError(
            "max_output_tokens must be int"
        )

        stderr_buf = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO(_make_input(max_output_tokens="abc"))),
            patch("sys.stdout", new_callable=io.StringIO),
            patch("sys.stderr", stderr_buf),
            patch("sys.argv", ["vertex_proxy.py"]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                vp.main()
            assert exc_info.value.code == 1


# ===========================================================================
# Behavior Table: vertex_proxy.print_help()
# ===========================================================================

class TestPrintHelp:
    """FUNCTION_SPEC §US-001 — print_help() behavior table rows 1-3."""

    def _capture_help(self, vertex_env) -> str:
        """Run print_help() and return captured stdout."""
        vp, _ = vertex_env
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            try:
                vp.print_help()
            except SystemExit:
                pass
        return buf.getvalue()

    def test_row1_mentions_max_output_tokens(self, vertex_env):
        """--help output contains 'max_output_tokens'."""
        help_text = self._capture_help(vertex_env)
        assert "max_output_tokens" in help_text

    def test_row2_documents_default_512(self, vertex_env):
        """Help text mentions the default value 512."""
        help_text = self._capture_help(vertex_env)
        assert "max_output_tokens" in help_text
        assert "512" in help_text

    def test_row3_existing_fields_still_present(self, vertex_env):
        """Help still documents all original stdin JSON fields."""
        help_text = self._capture_help(vertex_env)
        for field in ("system_prompt", "user_input", "model", "region"):
            assert field in help_text, f"Help text missing existing field: {field}"
