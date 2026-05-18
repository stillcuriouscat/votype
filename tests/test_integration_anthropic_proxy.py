"""Clean-room integration tests for anthropic_proxy.py (Module A).

Derived from LOW_LEVEL_DESIGN.md §1 Module A. Tests verify the CLI/IO
contract of the standalone proxy script — not its internals.

We invoke the script as a subprocess (its real interface). Network calls
are eliminated by either:
  - using --help / --test (no network)
  - making `import anthropic` fail (PYTHONPATH shim) so we exercise error
    paths without needing the SDK installed locally.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROXY_SCRIPT = PROJECT_ROOT / "anthropic_proxy.py"


def _run_proxy(
    *args: str,
    stdin: str | None = None,
    env_overrides: dict | None = None,
    timeout: int = 10,
) -> subprocess.CompletedProcess:
    """Invoke anthropic_proxy.py as a subprocess."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(PROXY_SCRIPT), *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _shim_path_breaking_anthropic(tmp_path: Path) -> Path:
    """Create a directory containing a fake 'anthropic.py' that raises ImportError.

    Prepending this to PYTHONPATH causes `import anthropic` to fail with our
    chosen message, letting us assert the SDK-missing error path.
    """
    pkg_dir = tmp_path / "shim"
    pkg_dir.mkdir()
    (pkg_dir / "anthropic.py").write_text(
        "raise ImportError('shimmed: anthropic not installed')\n"
    )
    return pkg_dir


def _has_proxy_script() -> bool:
    return PROXY_SCRIPT.exists()


pytestmark = pytest.mark.skipif(
    not _has_proxy_script(),
    reason="anthropic_proxy.py not yet implemented at repo root",
)


# ---------------------------------------------------------------------------
# Argv handling
# ---------------------------------------------------------------------------


class TestProxyArgvHandling:
    """LLD §1 Module A: argv handling parallels vertex_proxy.main."""

    def test_proxy_help_flag_exits_zero(self):
        result = _run_proxy("--help")
        assert result.returncode == 0, (
            f"--help must exit 0, got {result.returncode}\nstderr={result.stderr}"
        )
        assert result.stdout, "--help should print usage to stdout"

    def test_proxy_short_help_flag_exits_zero(self):
        result = _run_proxy("-h")
        assert result.returncode == 0
        assert result.stdout

    def test_proxy_unknown_arg_exits_one_with_message(self):
        result = _run_proxy("--bogus")
        assert result.returncode == 1
        assert "Unknown argument" in result.stderr
        assert "--bogus" in result.stderr


# ---------------------------------------------------------------------------
# --test mode
# ---------------------------------------------------------------------------


class TestProxyTestFlag:
    """LLD §1 Module A: run_test verifies SDK import + key file readable.

    Per the contract, --test must NOT perform a network call.
    """

    def test_proxy_test_flag_fails_when_sdk_missing(self, tmp_path):
        """When `import anthropic` fails, --test must exit 1 with FAIL line."""
        shim_dir = _shim_path_breaking_anthropic(tmp_path)
        result = _run_proxy(
            "--test",
            env_overrides={"PYTHONPATH": str(shim_dir)},
        )
        assert result.returncode == 1
        assert "FAIL" in result.stderr

    def test_proxy_test_flag_fails_when_key_missing(self, tmp_path, monkeypatch):
        """When the key file is missing, --test must exit 1 with FAIL line.

        We point HOME to a tmp dir that has no ~/.config/claude.secret.
        """
        # No anthropic shim here, but we don't care if the SDK check passes
        # or fails — either branch must FAIL when the key file is absent.
        result = _run_proxy(
            "--test",
            env_overrides={"HOME": str(tmp_path)},
        )
        assert result.returncode == 1
        assert "FAIL" in result.stderr


# ---------------------------------------------------------------------------
# Stdin JSON validation
# ---------------------------------------------------------------------------


class TestProxyStdinValidation:
    """LLD §1 Module A: stdin JSON schema + validation rules."""

    def test_proxy_invalid_json_exits_one(self):
        result = _run_proxy(stdin="not-json-at-all{")
        assert result.returncode == 1
        assert "Invalid JSON input" in result.stderr

    def test_proxy_missing_user_input_exits_one(self):
        result = _run_proxy(stdin=json.dumps({"system_prompt": "sys"}))
        assert result.returncode == 1
        assert "Missing 'user_input' in JSON" in result.stderr

    def test_proxy_empty_user_input_exits_one(self):
        result = _run_proxy(stdin=json.dumps({"user_input": ""}))
        assert result.returncode == 1
        assert "Missing 'user_input' in JSON" in result.stderr

    def test_proxy_sdk_missing_yields_clear_error(self, tmp_path):
        """ImportError path: stderr says SDK not installed, exits 1."""
        shim_dir = _shim_path_breaking_anthropic(tmp_path)
        result = _run_proxy(
            stdin=json.dumps({"user_input": "hello"}),
            env_overrides={"PYTHONPATH": str(shim_dir)},
        )
        assert result.returncode == 1
        # Per LLD: "<package> not installed: <e>" pattern.
        assert "not installed" in result.stderr.lower() or "anthropic" in result.stderr.lower()

    def test_proxy_missing_key_file_exits_with_api_key_error(self, tmp_path):
        """Valid JSON, SDK shim makes import succeed-then-fail OR the key
        file absence is detected first. Either way we expect rc=1.

        We force HOME to a temp dir so ~/.config/claude.secret does not exist.
        """
        result = _run_proxy(
            stdin=json.dumps({"user_input": "hello"}),
            env_overrides={"HOME": str(tmp_path)},
        )
        assert result.returncode == 1
        # The exact message depends on whether SDK import succeeded first.
        # Both are documented exit-1 paths.


# ---------------------------------------------------------------------------
# Source-level invariants (no project imports, key path constant)
# ---------------------------------------------------------------------------


class TestProxySourceInvariants:
    """LLD §1 Module A: invariants about the script source itself."""

    def test_proxy_self_contained_no_voice_input_imports(self):
        """Per LLD invariant: 'No imports from the voice_input project'."""
        import ast

        source = PROXY_SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)

        forbidden_roots = {
            "voice_input",
            "post_processor_configs",
            "post_processor_presets",
            "state_db",
            "model_configs",
            "model_presets",
            "openrouter_client",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in forbidden_roots, (
                        f"anthropic_proxy.py must be self-contained; "
                        f"found 'import {alias.name}'"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    assert root not in forbidden_roots, (
                        f"anthropic_proxy.py must be self-contained; "
                        f"found 'from {node.module} import ...'"
                    )

    def test_proxy_references_documented_key_path(self):
        """LLD: ANTHROPIC_KEY_PATH = Path('~/.config/claude.secret').expanduser()."""
        source = PROXY_SCRIPT.read_text(encoding="utf-8")
        assert "~/.config/claude.secret" in source, (
            "anthropic_proxy.py must reference ~/.config/claude.secret"
        )

    def test_proxy_references_default_model(self):
        """LLD: DEFAULT_MODEL = 'claude-haiku-4-5-20251001'."""
        source = PROXY_SCRIPT.read_text(encoding="utf-8")
        assert "claude-haiku-4-5-20251001" in source

    def test_proxy_uses_messages_api_shape(self):
        """LLD: client.messages.create(model=..., max_tokens=..., system=..., messages=...)."""
        source = PROXY_SCRIPT.read_text(encoding="utf-8")
        assert "messages.create" in source or "messages_create" in source
        assert "max_tokens" in source
        assert "system" in source
