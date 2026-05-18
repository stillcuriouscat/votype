"""Clean-room unit tests for US-001: anthropic_proxy.py.

Derived from FUNCTION_SPEC.md Module A behavior tables.
Verifies _trace, _import_anthropic, _read_api_key, print_help, run_test, main.
"""

import ast
import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
ANTHROPIC_PROXY_PATH = PROJECT_ROOT / "anthropic_proxy.py"
ALLOWED_IMPORTS = {
    "json", "sys", "time", "warnings", "argparse", "pathlib", "anthropic",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def anth_env(tmp_path, monkeypatch):
    """Import anthropic_proxy with mocked anthropic SDK + temp key file."""
    mock_anthropic = MagicMock()
    mock_text_block = MagicMock()
    mock_text_block.text = "corrected output"
    mock_response = MagicMock()
    mock_response.content = [mock_text_block]
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        mock_response
    )

    key_path = tmp_path / "claude.secret"
    key_path.write_text("sk-ant-xxxx\n", encoding="utf-8")

    saved_module = sys.modules.pop("anthropic_proxy", None)

    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        # Make `from anthropic import Anthropic` work
        mock_anthropic.Anthropic = mock_anthropic.Anthropic
        import anthropic_proxy as ap
        monkeypatch.setattr(ap, "ANTHROPIC_KEY_PATH", key_path)
        yield ap, mock_anthropic, key_path

    if saved_module is not None:
        sys.modules["anthropic_proxy"] = saved_module
    else:
        sys.modules.pop("anthropic_proxy", None)


def _run_main(ap, stdin_text, argv=None):
    """Helper to invoke main() with stdin + argv, capturing exit code/streams."""
    if argv is None:
        argv = ["anthropic_proxy.py"]
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with patch.object(sys, "argv", argv), \
         patch.object(sys, "stdin", io.StringIO(stdin_text)), \
         patch.object(sys, "stdout", captured_stdout), \
         patch.object(sys, "stderr", captured_stderr):
        with pytest.raises(SystemExit) as exc_info:
            ap.main()
    return exc_info.value.code, captured_stdout.getvalue(), captured_stderr.getvalue()


# ---------------------------------------------------------------------------
# _trace
# ---------------------------------------------------------------------------

class TestTrace:
    def test_writes_to_stderr_with_prefix(self, anth_env, capsys):
        ap, _, _ = anth_env
        ap._trace("sdk_init: 0.12s")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == "[TRACE] sdk_init: 0.12s\n"

    def test_empty_string(self, anth_env, capsys):
        ap, _, _ = anth_env
        ap._trace("")
        captured = capsys.readouterr()
        assert captured.err == "[TRACE] \n"


# ---------------------------------------------------------------------------
# _import_anthropic
# ---------------------------------------------------------------------------

class TestImportAnthropic:
    def test_returns_module_and_class(self, anth_env):
        ap, mock_anth, _ = anth_env
        mod, cls = ap._import_anthropic()
        assert mod is mock_anth
        assert cls is mock_anth.Anthropic


# ---------------------------------------------------------------------------
# _read_api_key
# ---------------------------------------------------------------------------

class TestReadApiKey:
    def test_normal(self, anth_env):
        ap, _, _ = anth_env
        assert ap._read_api_key() == "sk-ant-xxxx"

    def test_strips_whitespace(self, anth_env):
        ap, _, key_path = anth_env
        key_path.write_text("  sk-ant-yyy  \n\n", encoding="utf-8")
        assert ap._read_api_key() == "sk-ant-yyy"

    def test_missing_file_raises(self, anth_env):
        ap, _, key_path = anth_env
        key_path.unlink()
        with pytest.raises(FileNotFoundError):
            ap._read_api_key()

    def test_empty_file_raises_value_error(self, anth_env):
        ap, _, key_path = anth_env
        key_path.write_text("   \n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            ap._read_api_key()


# ---------------------------------------------------------------------------
# print_help
# ---------------------------------------------------------------------------

class TestPrintHelp:
    def test_contains_required_substrings(self, anth_env, capsys):
        ap, _, _ = anth_env
        ap.print_help()
        out = capsys.readouterr().out
        for needle in ("anthropic_proxy.py", "--help", "--test",
                       "system_prompt", "user_input", "model",
                       "max_tokens", "Stdout", "Exit 0"):
            assert needle in out, f"missing: {needle}"


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------

class TestRunTest:
    def test_success(self, anth_env, capsys):
        ap, _, _ = anth_env
        with pytest.raises(SystemExit) as exc:
            ap.run_test()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "OK: SDK import + API key file readable." in out

    def test_key_missing(self, anth_env, capsys):
        ap, _, key_path = anth_env
        key_path.unlink()
        with pytest.raises(SystemExit) as exc:
            ap.run_test()
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "FAIL:" in err

    def test_key_empty(self, anth_env, capsys):
        ap, _, key_path = anth_env
        key_path.write_text("", encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            ap.run_test()
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "FAIL:" in err
        assert "empty" in err


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def test_normal_success(self, anth_env):
        ap, mock_anth, _ = anth_env
        rc, out, err = _run_main(ap, '{"user_input":"hello"}')
        assert rc == 0
        assert out == "corrected output\n"
        assert "[TRACE] sdk_init:" in err
        assert "[TRACE] anthropic_api:" in err

    def test_all_fields(self, anth_env):
        ap, mock_anth, _ = anth_env
        payload = json.dumps({
            "system_prompt": "sys",
            "user_input": "u",
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 2048,
        })
        rc, out, err = _run_main(ap, payload)
        assert rc == 0
        call = mock_anth.Anthropic.return_value.messages.create.call_args
        assert call.kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call.kwargs["max_tokens"] == 2048
        assert call.kwargs["system"] == "sys"
        assert call.kwargs["messages"] == [{"role": "user", "content": "u"}]

    def test_defaults_applied(self, anth_env):
        ap, mock_anth, _ = anth_env
        rc, out, err = _run_main(ap, '{"user_input":"x"}')
        assert rc == 0
        call = mock_anth.Anthropic.return_value.messages.create.call_args
        assert call.kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call.kwargs["max_tokens"] == 1024
        assert call.kwargs["system"] == ""

    def test_output_stripped(self, anth_env):
        ap, mock_anth, _ = anth_env
        mock_anth.Anthropic.return_value.messages.create.return_value.content[0].text = "  hi  \n"
        rc, out, err = _run_main(ap, '{"user_input":"x"}')
        assert rc == 0
        assert out == "hi\n"

    def test_help_flag(self, anth_env, capsys):
        ap, _, _ = anth_env
        with patch.object(sys, "argv", ["anthropic_proxy.py", "--help"]):
            with pytest.raises(SystemExit) as exc:
                ap.main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "anthropic_proxy.py" in out

    def test_h_short_flag(self, anth_env, capsys):
        ap, _, _ = anth_env
        with patch.object(sys, "argv", ["anthropic_proxy.py", "-h"]):
            with pytest.raises(SystemExit) as exc:
                ap.main()
        assert exc.value.code == 0

    def test_unknown_argv(self, anth_env, capsys):
        ap, _, _ = anth_env
        with patch.object(sys, "argv", ["anthropic_proxy.py", "--foo"]):
            with pytest.raises(SystemExit) as exc:
                ap.main()
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Unknown argument: --foo" in err

    def test_invalid_json(self, anth_env):
        ap, _, _ = anth_env
        rc, out, err = _run_main(ap, "not json")
        assert rc == 1
        assert "Invalid JSON input: " in err

    def test_missing_user_input(self, anth_env):
        ap, _, _ = anth_env
        rc, out, err = _run_main(ap, "{}")
        assert rc == 1
        assert "Missing 'user_input' in JSON" in err

    def test_empty_user_input(self, anth_env):
        ap, _, _ = anth_env
        rc, out, err = _run_main(ap, '{"user_input":""}')
        assert rc == 1
        assert "Missing 'user_input' in JSON" in err

    def test_api_key_file_missing(self, anth_env):
        ap, _, key_path = anth_env
        key_path.unlink()
        rc, out, err = _run_main(ap, '{"user_input":"x"}')
        assert rc == 1
        assert "API key error: " in err

    def test_api_key_file_empty(self, anth_env):
        ap, _, key_path = anth_env
        key_path.write_text("", encoding="utf-8")
        rc, out, err = _run_main(ap, '{"user_input":"x"}')
        assert rc == 1
        assert "API key error: " in err
        assert "empty" in err

    def test_api_error(self, anth_env):
        ap, mock_anth, _ = anth_env
        mock_anth.Anthropic.return_value.messages.create.side_effect = (
            RuntimeError("503 Service Unavailable")
        )
        rc, out, err = _run_main(ap, '{"user_input":"x"}')
        assert rc == 1
        assert "Anthropic API error: " in err
        assert "503" in err

    def test_empty_content_list(self, anth_env):
        ap, mock_anth, _ = anth_env
        mock_anth.Anthropic.return_value.messages.create.return_value.content = []
        rc, out, err = _run_main(ap, '{"user_input":"x"}')
        assert rc == 1
        assert "Anthropic returned empty response" in err

    def test_content_text_empty(self, anth_env):
        ap, mock_anth, _ = anth_env
        mock_anth.Anthropic.return_value.messages.create.return_value.content[0].text = ""
        rc, out, err = _run_main(ap, '{"user_input":"x"}')
        assert rc == 1
        assert "Anthropic returned empty response" in err

    def test_content_text_none(self, anth_env):
        ap, mock_anth, _ = anth_env
        mock_anth.Anthropic.return_value.messages.create.return_value.content[0].text = None
        rc, out, err = _run_main(ap, '{"user_input":"x"}')
        assert rc == 1
        assert "Anthropic returned empty response" in err


# ---------------------------------------------------------------------------
# AST invariant: no project imports
# ---------------------------------------------------------------------------

class TestImportInvariant:
    def test_only_stdlib_and_anthropic(self):
        tree = ast.parse(ANTHROPIC_PROXY_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in ALLOWED_IMPORTS, f"bad import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                top = node.module.split(".")[0]
                assert top in ALLOWED_IMPORTS, f"bad import: {node.module}"

    def test_no_project_imports(self):
        tree = ast.parse(ANTHROPIC_PROXY_PATH.read_text(encoding="utf-8"))
        forbidden = {"voice_input", "post_processor_configs",
                     "post_processor_presets", "state_db", "openrouter_client"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in forbidden, f"forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                top = node.module.split(".")[0]
                assert top not in forbidden, f"forbidden import: {node.module}"
