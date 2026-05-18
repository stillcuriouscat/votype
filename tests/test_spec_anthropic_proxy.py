"""Clean-room unit tests for anthropic_proxy.py.

Derived from FUNCTION_SPEC.md Module A. Tests are spec-driven; no
implementation source has been read.
"""

import ast
import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_anthropic_module(text="hi", raise_exc=None):
    """Build a fake `anthropic` module that mimics the SDK surface used."""
    fake_module = MagicMock(name="anthropic_module")
    fake_client = MagicMock(name="Anthropic_client")
    if raise_exc is not None:
        fake_client.messages.create.side_effect = raise_exc
    else:
        response = MagicMock(name="Message")
        content_block = MagicMock(name="TextBlock")
        content_block.text = text
        response.content = [content_block]
        fake_client.messages.create.return_value = response
    fake_module.Anthropic.return_value = fake_client
    return fake_module, fake_client


def _install_fake_anthropic(monkeypatch, **kwargs):
    fake, client = _fake_anthropic_module(**kwargs)
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    return fake, client


def _install_key(monkeypatch, tmp_path, content="sk-ant-test\n"):
    import anthropic_proxy

    key_file = tmp_path / "claude.secret"
    key_file.write_text(content, encoding="utf-8")
    monkeypatch.setattr(anthropic_proxy, "ANTHROPIC_KEY_PATH", key_file)
    return key_file


# ---------------------------------------------------------------------------
# Module-level
# ---------------------------------------------------------------------------


def test_module_imports_successfully():
    import anthropic_proxy  # noqa: F401


def test_module_exposes_required_callables():
    import anthropic_proxy

    for name in ("_trace", "_import_anthropic", "_read_api_key",
                "print_help", "run_test", "main"):
        assert callable(getattr(anthropic_proxy, name)), name


def test_module_constants():
    import anthropic_proxy

    assert anthropic_proxy.DEFAULT_MODEL == "claude-haiku-4-5-20251001"
    assert anthropic_proxy.DEFAULT_MAX_TOKENS == 1024
    assert isinstance(anthropic_proxy.ANTHROPIC_KEY_PATH, Path)
    assert anthropic_proxy.ANTHROPIC_KEY_PATH == Path(
        "~/.config/claude.secret").expanduser()


# ---------------------------------------------------------------------------
# Invariant: no project imports (AST scan)
# ---------------------------------------------------------------------------


def test_no_voice_input_project_imports():
    """Module must not import any voice_input package modules."""
    import anthropic_proxy

    src = Path(anthropic_proxy.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    allowed = {
        "json", "sys", "time", "warnings", "argparse", "pathlib",
        "anthropic", "os", "io", "typing",
    }
    forbidden_substrings = ("voice_input", "post_processor_",
                            "state_db", "openrouter_client")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert not any(f in alias.name for f in forbidden_substrings), \
                    f"Forbidden import: {alias.name}"
                assert root in allowed, f"Unexpected import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            assert not any(f in mod for f in forbidden_substrings), \
                f"Forbidden from-import: {mod}"
            assert root in allowed, f"Unexpected from-import: {mod}"


# ---------------------------------------------------------------------------
# _trace
# ---------------------------------------------------------------------------


class TestTrace:
    def test_trace_normal(self, capsys):
        import anthropic_proxy
        result = anthropic_proxy._trace("sdk_init: 0.12s")
        captured = capsys.readouterr()
        assert result is None
        assert captured.err == "[TRACE] sdk_init: 0.12s\n"
        assert captured.out == ""

    def test_trace_empty_string(self, capsys):
        import anthropic_proxy
        anthropic_proxy._trace("")
        captured = capsys.readouterr()
        assert captured.err == "[TRACE] \n"

    def test_trace_with_newline(self, capsys):
        import anthropic_proxy
        anthropic_proxy._trace("a\nb")
        captured = capsys.readouterr()
        assert captured.err == "[TRACE] a\nb\n"


# ---------------------------------------------------------------------------
# _import_anthropic
# ---------------------------------------------------------------------------


class TestImportAnthropic:
    def test_returns_module_and_class(self, monkeypatch):
        import anthropic_proxy
        fake, _ = _install_fake_anthropic(monkeypatch)
        mod, klass = anthropic_proxy._import_anthropic()
        assert mod is fake
        assert klass is fake.Anthropic

    def test_repeated_call_same_class(self, monkeypatch):
        import anthropic_proxy
        _install_fake_anthropic(monkeypatch)
        m1, c1 = anthropic_proxy._import_anthropic()
        m2, c2 = anthropic_proxy._import_anthropic()
        assert c1 is c2

    def test_import_error_when_missing(self, monkeypatch):
        import anthropic_proxy
        monkeypatch.setitem(sys.modules, "anthropic", None)
        with pytest.raises(ImportError):
            anthropic_proxy._import_anthropic()


# ---------------------------------------------------------------------------
# _read_api_key
# ---------------------------------------------------------------------------


class TestReadApiKey:
    def test_normal(self, monkeypatch, tmp_path):
        import anthropic_proxy
        _install_key(monkeypatch, tmp_path, "sk-ant-xxxx\n")
        assert anthropic_proxy._read_api_key() == "sk-ant-xxxx"

    def test_whitespace_stripped(self, monkeypatch, tmp_path):
        import anthropic_proxy
        _install_key(monkeypatch, tmp_path, "  sk-ant-yyy  \n\n")
        assert anthropic_proxy._read_api_key() == "sk-ant-yyy"

    def test_missing_file_raises(self, monkeypatch, tmp_path):
        import anthropic_proxy
        missing = tmp_path / "no-such.secret"
        monkeypatch.setattr(anthropic_proxy, "ANTHROPIC_KEY_PATH", missing)
        with pytest.raises(FileNotFoundError):
            anthropic_proxy._read_api_key()

    def test_empty_file_raises_valueerror(self, monkeypatch, tmp_path):
        import anthropic_proxy
        _install_key(monkeypatch, tmp_path, "")
        with pytest.raises(ValueError, match="API key file is empty"):
            anthropic_proxy._read_api_key()

    def test_whitespace_only_raises_valueerror(self, monkeypatch, tmp_path):
        import anthropic_proxy
        _install_key(monkeypatch, tmp_path, "   \n")
        with pytest.raises(ValueError, match="API key file is empty"):
            anthropic_proxy._read_api_key()


# ---------------------------------------------------------------------------
# print_help
# ---------------------------------------------------------------------------


def test_print_help_contains_required_substrings(capsys):
    import anthropic_proxy
    result = anthropic_proxy.print_help()
    captured = capsys.readouterr()
    assert result is None
    required = [
        "anthropic_proxy.py", "--help", "--test",
        "system_prompt", "user_input", "model",
        "max_tokens", "Stdout", "Exit 0",
    ]
    for substr in required:
        assert substr in captured.out, f"help text missing: {substr!r}"


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------


class TestRunTest:
    def test_success_exits_zero(self, capsys, monkeypatch, tmp_path):
        import anthropic_proxy
        _install_fake_anthropic(monkeypatch)
        _install_key(monkeypatch, tmp_path, "sk-ant-xxx")
        with pytest.raises(SystemExit) as exc:
            anthropic_proxy.run_test()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "OK: SDK import + API key file readable." in captured.out

    def test_sdk_missing_exits_one(self, capsys, monkeypatch, tmp_path):
        import anthropic_proxy
        monkeypatch.setitem(sys.modules, "anthropic", None)
        _install_key(monkeypatch, tmp_path, "sk-ant-xxx")
        with pytest.raises(SystemExit) as exc:
            anthropic_proxy.run_test()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "FAIL: " in captured.err
        assert "anthropic" in captured.err

    def test_missing_key_exits_one(self, capsys, monkeypatch, tmp_path):
        import anthropic_proxy
        _install_fake_anthropic(monkeypatch)
        missing = tmp_path / "absent.secret"
        monkeypatch.setattr(anthropic_proxy, "ANTHROPIC_KEY_PATH", missing)
        with pytest.raises(SystemExit) as exc:
            anthropic_proxy.run_test()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "FAIL: " in captured.err

    def test_empty_key_exits_one(self, capsys, monkeypatch, tmp_path):
        import anthropic_proxy
        _install_fake_anthropic(monkeypatch)
        _install_key(monkeypatch, tmp_path, "")
        with pytest.raises(SystemExit) as exc:
            anthropic_proxy.run_test()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "FAIL: " in captured.err
        assert "empty" in captured.err.lower()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _run_main(monkeypatch, argv=None, stdin_text=""):
    """Invoke anthropic_proxy.main() with patched argv/stdin."""
    import anthropic_proxy
    monkeypatch.setattr(sys, "argv", argv or ["anthropic_proxy.py"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    with pytest.raises(SystemExit) as exc:
        anthropic_proxy.main()
    return exc.value.code


class TestMainArgvDispatch:
    def test_help_flag_prints_help_exit_0(self, monkeypatch, capsys):
        code = _run_main(monkeypatch, argv=["anthropic_proxy.py", "--help"])
        out = capsys.readouterr().out
        assert code == 0
        assert "anthropic_proxy.py" in out
        assert "--help" in out

    def test_short_help_flag(self, monkeypatch, capsys):
        code = _run_main(monkeypatch, argv=["anthropic_proxy.py", "-h"])
        out = capsys.readouterr().out
        assert code == 0
        assert "anthropic_proxy.py" in out

    def test_test_flag(self, monkeypatch, capsys, tmp_path):
        _install_fake_anthropic(monkeypatch)
        _install_key(monkeypatch, tmp_path, "sk-ant-xxx")
        code = _run_main(monkeypatch, argv=["anthropic_proxy.py", "--test"])
        captured = capsys.readouterr()
        assert code == 0
        assert "OK: SDK import + API key file readable." in captured.out

    def test_unknown_argv(self, monkeypatch, capsys):
        code = _run_main(monkeypatch, argv=["anthropic_proxy.py", "--foo"])
        captured = capsys.readouterr()
        assert code == 1
        assert "Unknown argument: --foo" in captured.err


class TestMainJsonProcessing:
    def test_normal_success(self, monkeypatch, capsys, tmp_path):
        _install_fake_anthropic(monkeypatch, text="你好，世界 abc")
        _install_key(monkeypatch, tmp_path, "sk-ant-test")
        stdin = json.dumps({"user_input": "你好世界abc"})
        code = _run_main(monkeypatch, stdin_text=stdin)
        captured = capsys.readouterr()
        assert code == 0
        assert captured.out == "你好，世界 abc\n"
        assert "[TRACE] sdk_init:" in captured.err
        assert "[TRACE] anthropic_api:" in captured.err

    def test_all_fields_passed(self, monkeypatch, tmp_path):
        fake, client = _install_fake_anthropic(monkeypatch, text="ok")
        _install_key(monkeypatch, tmp_path)
        stdin = json.dumps({
            "system_prompt": "sys",
            "user_input": "u",
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 2048,
        })
        code = _run_main(monkeypatch, stdin_text=stdin)
        assert code == 0
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5-20251001"
        assert kwargs["max_tokens"] == 2048
        assert kwargs["system"] == "sys"
        assert kwargs["messages"] == [{"role": "user", "content": "u"}]

    def test_extra_fields_ignored(self, monkeypatch, capsys, tmp_path):
        _install_fake_anthropic(monkeypatch, text="ok")
        _install_key(monkeypatch, tmp_path)
        stdin = json.dumps({"user_input": "x", "unknown": "y"})
        code = _run_main(monkeypatch, stdin_text=stdin)
        assert code == 0
        assert capsys.readouterr().out == "ok\n"

    def test_output_stripped(self, monkeypatch, capsys, tmp_path):
        _install_fake_anthropic(monkeypatch, text="  hi  \n")
        _install_key(monkeypatch, tmp_path)
        stdin = json.dumps({"user_input": "abc"})
        code = _run_main(monkeypatch, stdin_text=stdin)
        assert code == 0
        assert capsys.readouterr().out == "hi\n"

    def test_defaults_applied(self, monkeypatch, tmp_path):
        fake, client = _install_fake_anthropic(monkeypatch, text="ok")
        _install_key(monkeypatch, tmp_path)
        stdin = json.dumps({"user_input": "u"})
        _run_main(monkeypatch, stdin_text=stdin)
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5-20251001"
        assert kwargs["max_tokens"] == 1024
        assert kwargs["system"] == ""


class TestMainErrors:
    def test_invalid_json(self, monkeypatch, capsys, tmp_path):
        _install_fake_anthropic(monkeypatch)
        _install_key(monkeypatch, tmp_path)
        code = _run_main(monkeypatch, stdin_text="not json")
        captured = capsys.readouterr()
        assert code == 1
        assert captured.err.startswith("Invalid JSON input: ")

    def test_missing_user_input(self, monkeypatch, capsys, tmp_path):
        _install_fake_anthropic(monkeypatch)
        _install_key(monkeypatch, tmp_path)
        code = _run_main(monkeypatch, stdin_text="{}")
        captured = capsys.readouterr()
        assert code == 1
        assert "Missing 'user_input' in JSON" in captured.err

    def test_empty_user_input(self, monkeypatch, capsys, tmp_path):
        _install_fake_anthropic(monkeypatch)
        _install_key(monkeypatch, tmp_path)
        stdin = json.dumps({"user_input": ""})
        code = _run_main(monkeypatch, stdin_text=stdin)
        captured = capsys.readouterr()
        assert code == 1
        assert "Missing 'user_input' in JSON" in captured.err

    def test_sdk_import_failure(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setitem(sys.modules, "anthropic", None)
        _install_key(monkeypatch, tmp_path)
        stdin = json.dumps({"user_input": "x"})
        code = _run_main(monkeypatch, stdin_text=stdin)
        captured = capsys.readouterr()
        assert code == 1
        assert captured.err.startswith("anthropic SDK not installed: ") or \
            "anthropic SDK not installed: " in captured.err

    def test_api_key_missing(self, monkeypatch, capsys, tmp_path):
        import anthropic_proxy
        _install_fake_anthropic(monkeypatch)
        monkeypatch.setattr(anthropic_proxy, "ANTHROPIC_KEY_PATH",
                            tmp_path / "absent.secret")
        stdin = json.dumps({"user_input": "x"})
        code = _run_main(monkeypatch, stdin_text=stdin)
        captured = capsys.readouterr()
        assert code == 1
        assert "API key error: " in captured.err

    def test_api_key_empty(self, monkeypatch, capsys, tmp_path):
        _install_fake_anthropic(monkeypatch)
        _install_key(monkeypatch, tmp_path, "")
        stdin = json.dumps({"user_input": "x"})
        code = _run_main(monkeypatch, stdin_text=stdin)
        captured = capsys.readouterr()
        assert code == 1
        assert "API key error: " in captured.err
        assert "empty" in captured.err.lower()

    def test_anthropic_5xx_error(self, monkeypatch, capsys, tmp_path):
        _install_fake_anthropic(monkeypatch,
                                raise_exc=RuntimeError("503 Service Unavailable"))
        _install_key(monkeypatch, tmp_path)
        stdin = json.dumps({"user_input": "abc"})
        code = _run_main(monkeypatch, stdin_text=stdin)
        captured = capsys.readouterr()
        assert code == 1
        assert "Anthropic API error: " in captured.err
        assert "503" in captured.err

    def test_anthropic_429_error(self, monkeypatch, capsys, tmp_path):
        _install_fake_anthropic(
            monkeypatch,
            raise_exc=RuntimeError("rate_limit_error: too many requests"))
        _install_key(monkeypatch, tmp_path)
        stdin = json.dumps({"user_input": "abc"})
        code = _run_main(monkeypatch, stdin_text=stdin)
        captured = capsys.readouterr()
        assert code == 1
        assert "Anthropic API error: " in captured.err
        assert "rate_limit_error" in captured.err

    def test_empty_content_list(self, monkeypatch, capsys, tmp_path):
        fake_module = MagicMock()
        fake_client = MagicMock()
        response = MagicMock()
        response.content = []
        fake_client.messages.create.return_value = response
        fake_module.Anthropic.return_value = fake_client
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)
        _install_key(monkeypatch, tmp_path)
        stdin = json.dumps({"user_input": "abc"})
        code = _run_main(monkeypatch, stdin_text=stdin)
        captured = capsys.readouterr()
        assert code == 1
        assert "Anthropic returned empty response" in captured.err

    def test_content_text_empty(self, monkeypatch, capsys, tmp_path):
        _install_fake_anthropic(monkeypatch, text="")
        _install_key(monkeypatch, tmp_path)
        stdin = json.dumps({"user_input": "abc"})
        code = _run_main(monkeypatch, stdin_text=stdin)
        captured = capsys.readouterr()
        assert code == 1
        assert "Anthropic returned empty response" in captured.err

    def test_content_text_none(self, monkeypatch, capsys, tmp_path):
        _install_fake_anthropic(monkeypatch, text=None)
        _install_key(monkeypatch, tmp_path)
        stdin = json.dumps({"user_input": "abc"})
        code = _run_main(monkeypatch, stdin_text=stdin)
        captured = capsys.readouterr()
        assert code == 1
        assert "Anthropic returned empty response" in captured.err
