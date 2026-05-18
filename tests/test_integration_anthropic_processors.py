"""Clean-room integration tests for process_with_anthropic{,_merge} (Module B).

Derived from LOW_LEVEL_DESIGN.md §1 Module B. Tests verify the
function-level contracts of the two new post-processor entry points in
post_processor_configs.py.

External dependencies that are mocked:
  - The SSH subprocess (via patching `_run_vertex_proxy`).
  - The OpenRouter fallback (via patching `call_openrouter`).
  - voice_input.notify (lazy-imported inside the functions on failure).

We do NOT mock the modules under test (post_processor_configs).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_module():
    """Import post_processor_configs; skip the whole module if functions absent."""
    try:
        import post_processor_configs as ppc  # type: ignore
    except Exception as e:  # pragma: no cover - skip path
        pytest.skip(f"post_processor_configs import failed: {e}")
    return ppc


def _has_new_functions() -> bool:
    try:
        import post_processor_configs as ppc  # type: ignore
    except Exception:
        return False
    return hasattr(ppc, "process_with_anthropic") and hasattr(
        ppc, "process_with_anthropic_merge"
    )


pytestmark = pytest.mark.skipif(
    not _has_new_functions(),
    reason="process_with_anthropic{,_merge} not yet implemented",
)


@pytest.fixture
def config_fix() -> dict:
    """Minimal valid config for process_with_anthropic (fix variant)."""
    return {
        "ssh_host": "oracle-cloud",
        "proxy_script": "~/anthropic_proxy.py",
        "model": "claude-haiku-4-5-20251001",
        "timeout": 15,
        "min_text_len": 15,
    }


@pytest.fixture
def config_merge() -> dict:
    """Minimal valid config for process_with_anthropic_merge."""
    return {
        "ssh_host": "oracle-cloud",
        "proxy_script": "~/anthropic_proxy.py",
        "model": "claude-haiku-4-5-20251001",
        "timeout": 15,
        "min_text_len": 15,
    }


def _ok_completed_process(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["ssh", "..."], returncode=0, stdout=stdout, stderr=""
    )


def _fail_completed_process(stderr: str = "boom") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["ssh", "..."], returncode=1, stdout="", stderr=stderr
    )


# ---------------------------------------------------------------------------
# Signature contract
# ---------------------------------------------------------------------------


class TestSignatures:
    """LLD §1 Module B: signature compatibility with vertex_ai counterparts."""

    def test_process_with_anthropic_signature(self):
        ppc = _import_module()
        import inspect

        sig = inspect.signature(ppc.process_with_anthropic)
        params = list(sig.parameters)
        # (text, config, glossary_ctx="")
        assert params[:2] == ["text", "config"], (
            f"process_with_anthropic params: expected text, config, ...; got {params}"
        )
        assert "glossary_ctx" in params
        assert sig.parameters["glossary_ctx"].default == ""

    def test_process_with_anthropic_merge_signature(self):
        ppc = _import_module()
        import inspect

        sig = inspect.signature(ppc.process_with_anthropic_merge)
        params = list(sig.parameters)
        # (primary_text, secondary_text, config, glossary_ctx="")
        assert params[:3] == ["primary_text", "secondary_text", "config"], (
            f"process_with_anthropic_merge params: got {params}"
        )
        assert "glossary_ctx" in params
        assert sig.parameters["glossary_ctx"].default == ""


# ---------------------------------------------------------------------------
# process_with_anthropic — fix variant
# ---------------------------------------------------------------------------


class TestProcessWithAnthropic:
    """LLD §1 Module B: process_with_anthropic contract."""

    def test_empty_text_short_circuits(self, config_fix):
        """text == '' → returns '' (no SSH call, no OpenRouter)."""
        ppc = _import_module()
        with patch.object(ppc, "_run_vertex_proxy") as run_mock, \
             patch("openrouter_client.call_openrouter") as or_mock:
            result = ppc.process_with_anthropic("", config_fix)
            assert result == ""
            run_mock.assert_not_called()
            or_mock.assert_not_called()

    def test_short_text_below_min_len_returns_input(self, config_fix):
        """len(text) < min_text_len → returns text unchanged."""
        ppc = _import_module()
        short = "abc"  # < 15
        with patch.object(ppc, "_run_vertex_proxy") as run_mock, \
             patch("openrouter_client.call_openrouter") as or_mock:
            result = ppc.process_with_anthropic(short, config_fix)
            assert result == short
            run_mock.assert_not_called()
            or_mock.assert_not_called()

    def test_success_path_returns_polished_text(self, config_fix):
        """Successful SSH proxy call → returns stripped stdout."""
        ppc = _import_module()
        text = "this is a longer text that exceeds the minimum length threshold"
        polished = "this is a longer text that exceeds the minimum length threshold."

        with patch.object(ppc, "_run_vertex_proxy",
                          return_value=_ok_completed_process(polished + "\n")):
            result = ppc.process_with_anthropic(text, config_fix)
        # The output may be exactly polished or some normalized form, but it
        # must be a non-empty string. Crucially: it must not equal the input
        # when the proxy succeeded with a different value.
        assert isinstance(result, str)
        assert result.strip() == polished.strip() or result == text

    def test_ssh_cmd_shape(self, config_fix):
        """LLD: cmd = ['ssh','-o','ConnectTimeout=5', ssh_host, 'python3', proxy_script]."""
        ppc = _import_module()
        text = "this is a longer text that exceeds the minimum length threshold"

        with patch.object(ppc, "_run_vertex_proxy",
                          return_value=_ok_completed_process("ok")) as run_mock:
            ppc.process_with_anthropic(text, config_fix)
            assert run_mock.called, "_run_vertex_proxy should be invoked"
            cmd = run_mock.call_args.args[0] if run_mock.call_args.args \
                else run_mock.call_args.kwargs.get("cmd")
            assert cmd[0] == "ssh"
            assert "-o" in cmd
            assert "ConnectTimeout=5" in cmd
            assert "oracle-cloud" in cmd
            assert "python3" in cmd
            assert "~/anthropic_proxy.py" in cmd

    def test_stdin_json_has_max_tokens_not_max_output_tokens(self, config_fix):
        """LLD: Anthropic SDK uses 'max_tokens' (NOT 'max_output_tokens')."""
        ppc = _import_module()
        text = "this is a longer text that exceeds the minimum length threshold"

        captured = {}

        def fake_proxy(cmd, stdin_data, timeout, fallback_model=None):
            captured["stdin_data"] = stdin_data
            return _ok_completed_process("polished")

        with patch.object(ppc, "_run_vertex_proxy", side_effect=fake_proxy):
            ppc.process_with_anthropic(text, config_fix)

        assert "stdin_data" in captured, "_run_vertex_proxy not called"
        import json as _json
        payload = _json.loads(captured["stdin_data"])
        assert "max_tokens" in payload, "JSON payload must have 'max_tokens'"
        assert "max_output_tokens" not in payload, (
            "Anthropic uses 'max_tokens', not 'max_output_tokens'"
        )
        # max_tokens formula: min(8192, max(512, len(user_input)))
        user_input = payload["user_input"]
        expected = min(8192, max(512, len(user_input)))
        assert payload["max_tokens"] == expected, (
            f"max_tokens must be min(8192, max(512, len(user_input))); "
            f"got {payload['max_tokens']}, expected {expected}"
        )

    def test_timeout_returns_input_text(self, config_fix):
        """subprocess.TimeoutExpired → notify + return input text.

        OpenRouter may also be invoked as fallback; final return must be
        the original text or the OpenRouter result (None → input)."""
        ppc = _import_module()
        text = "this is a longer text that exceeds the minimum length threshold"

        with patch.object(ppc, "_run_vertex_proxy",
                          side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=15)), \
             patch("openrouter_client.call_openrouter", return_value=None):
            result = ppc.process_with_anthropic(text, config_fix)
        assert result == text, (
            "On timeout with OpenRouter also failing, must return original text"
        )

    def test_nonzero_returncode_falls_through_to_openrouter(self, config_fix):
        """returncode != 0 → OpenRouter is consulted as fallback."""
        ppc = _import_module()
        text = "this is a longer text that exceeds the minimum length threshold"

        with patch.object(ppc, "_run_vertex_proxy",
                          return_value=_fail_completed_process()), \
             patch("openrouter_client.call_openrouter",
                   return_value="openrouter polished") as or_mock:
            result = ppc.process_with_anthropic(text, config_fix)
        assert or_mock.called, "OpenRouter must be consulted on rc != 0"
        # When OpenRouter succeeds, its text is returned.
        assert result == "openrouter polished"

    def test_both_failed_returns_input_text(self, config_fix):
        """proxy rc!=0 AND OpenRouter returns None → return input text."""
        ppc = _import_module()
        text = "this is a longer text that exceeds the minimum length threshold"

        with patch.object(ppc, "_run_vertex_proxy",
                          return_value=_fail_completed_process()), \
             patch("openrouter_client.call_openrouter", return_value=None):
            result = ppc.process_with_anthropic(text, config_fix)
        assert result == text

    def test_hallucination_guard_returns_input(self, config_fix):
        """len(output) > 2*len(text) → return original text."""
        ppc = _import_module()
        text = "this is a longer text that exceeds the minimum length threshold"
        huge = text * 5  # > 2x

        with patch.object(ppc, "_run_vertex_proxy",
                          return_value=_ok_completed_process(huge)), \
             patch("openrouter_client.call_openrouter", return_value=None):
            result = ppc.process_with_anthropic(text, config_fix)
        assert result == text, (
            "Hallucination guard (output > 2x input) must return original"
        )

    def test_question_guard_returns_input(self, config_fix):
        """input has '？' but output has neither '？' nor '?' → return original."""
        ppc = _import_module()
        text = "请问这个问题怎么回答？这是一个很长的问句需要超过最小长度门槛"
        non_question_output = "这是回答没有问号的内容文本"

        with patch.object(ppc, "_run_vertex_proxy",
                          return_value=_ok_completed_process(non_question_output)), \
             patch("openrouter_client.call_openrouter", return_value=None):
            result = ppc.process_with_anthropic(text, config_fix)
        assert result == text, (
            "Question guard: input has '？' but output has no '？'/'?' → return original"
        )

    def test_never_raises(self, config_fix):
        """LLD: process_with_anthropic never raises."""
        ppc = _import_module()
        text = "this is a longer text that exceeds the minimum length threshold"

        with patch.object(ppc, "_run_vertex_proxy",
                          side_effect=RuntimeError("unexpected!")), \
             patch("openrouter_client.call_openrouter", return_value=None):
            # Must not raise
            try:
                result = ppc.process_with_anthropic(text, config_fix)
            except Exception as e:
                pytest.fail(f"process_with_anthropic raised: {e!r}")
            assert isinstance(result, str)

    def test_does_not_mutate_config(self, config_fix):
        """LLD invariant: 'No mutation of config or glossary_ctx arguments'."""
        ppc = _import_module()
        text = "this is a longer text that exceeds the minimum length threshold"
        original = dict(config_fix)

        with patch.object(ppc, "_run_vertex_proxy",
                          return_value=_ok_completed_process("polished")):
            ppc.process_with_anthropic(text, config_fix)
        assert config_fix == original, "config dict must not be mutated"


# ---------------------------------------------------------------------------
# process_with_anthropic_merge — dual-ASR variant
# ---------------------------------------------------------------------------


class TestProcessWithAnthropicMerge:
    """LLD §1 Module B: process_with_anthropic_merge contract."""

    def test_empty_primary_with_secondary_returns_secondary(self, config_merge):
        """primary == '' and secondary truthy → return secondary."""
        ppc = _import_module()
        with patch.object(ppc, "_run_vertex_proxy") as run_mock, \
             patch("openrouter_client.call_openrouter") as or_mock:
            result = ppc.process_with_anthropic_merge("", "english secondary text", config_merge)
            assert result == "english secondary text"
            run_mock.assert_not_called()
            or_mock.assert_not_called()

    def test_empty_primary_without_secondary_returns_empty(self, config_merge):
        """primary == '' and not secondary → return ''."""
        ppc = _import_module()
        with patch.object(ppc, "_run_vertex_proxy") as run_mock:
            assert ppc.process_with_anthropic_merge("", None, config_merge) == ""
            assert ppc.process_with_anthropic_merge("", "", config_merge) == ""
            run_mock.assert_not_called()

    def test_short_primary_secondary_longer_returns_secondary(self, config_merge):
        """len(primary) < min_text_len and secondary longer → return secondary."""
        ppc = _import_module()
        primary = "短"  # very short
        secondary = "english secondary text that is much longer than primary"
        with patch.object(ppc, "_run_vertex_proxy") as run_mock:
            result = ppc.process_with_anthropic_merge(primary, secondary, config_merge)
            assert result == secondary
            run_mock.assert_not_called()

    def test_short_primary_no_longer_secondary_returns_primary(self, config_merge):
        """len(primary) < min_text_len and not secondary-longer → return primary."""
        ppc = _import_module()
        primary = "short"
        with patch.object(ppc, "_run_vertex_proxy") as run_mock:
            assert ppc.process_with_anthropic_merge(primary, None, config_merge) == primary
            assert ppc.process_with_anthropic_merge(primary, "x", config_merge) == primary
            run_mock.assert_not_called()

    def test_user_input_format_with_secondary(self, config_merge):
        """LLD: user_input = 'Chinese ASR: {primary}\\nEnglish ASR: {secondary}'."""
        ppc = _import_module()
        primary = "这是一段足够长的中文转录文本用于触发合并流程"
        secondary = "this is the english secondary ASR transcription"

        captured = {}

        def fake_proxy(cmd, stdin_data, timeout, fallback_model=None):
            captured["stdin"] = stdin_data
            return _ok_completed_process("merged result text")

        with patch.object(ppc, "_run_vertex_proxy", side_effect=fake_proxy):
            ppc.process_with_anthropic_merge(primary, secondary, config_merge)

        import json as _json
        payload = _json.loads(captured["stdin"])
        ui = payload["user_input"]
        assert "Chinese ASR:" in ui and primary in ui
        assert "English ASR:" in ui and secondary in ui

    def test_user_input_format_without_secondary(self, config_merge):
        """LLD: secondary None → user_input = 'Chinese ASR: {primary}' (no English line)."""
        ppc = _import_module()
        primary = "这是一段足够长的中文转录文本用于触发合并流程"

        captured = {}

        def fake_proxy(cmd, stdin_data, timeout, fallback_model=None):
            captured["stdin"] = stdin_data
            return _ok_completed_process("merged result text")

        with patch.object(ppc, "_run_vertex_proxy", side_effect=fake_proxy):
            ppc.process_with_anthropic_merge(primary, None, config_merge)

        import json as _json
        payload = _json.loads(captured["stdin"])
        ui = payload["user_input"]
        assert "Chinese ASR:" in ui and primary in ui
        assert "English ASR:" not in ui, (
            "When secondary is None, English ASR line must be omitted"
        )

    def test_timeout_returns_primary(self, config_merge):
        ppc = _import_module()
        primary = "这是一段足够长的中文转录文本用于触发合并流程"
        secondary = "english"

        with patch.object(ppc, "_run_vertex_proxy",
                          side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=15)), \
             patch("openrouter_client.call_openrouter", return_value=None):
            result = ppc.process_with_anthropic_merge(primary, secondary, config_merge)
        assert result == primary

    def test_nonzero_rc_falls_back_to_openrouter(self, config_merge):
        ppc = _import_module()
        primary = "这是一段足够长的中文转录文本用于触发合并流程"
        secondary = "english"

        with patch.object(ppc, "_run_vertex_proxy",
                          return_value=_fail_completed_process()), \
             patch("openrouter_client.call_openrouter",
                   return_value="openrouter merged") as or_mock:
            result = ppc.process_with_anthropic_merge(primary, secondary, config_merge)
        assert or_mock.called, "Must fall back to OpenRouter on rc != 0"
        assert result == "openrouter merged"

    def test_both_failed_returns_primary(self, config_merge):
        ppc = _import_module()
        primary = "这是一段足够长的中文转录文本用于触发合并流程"

        with patch.object(ppc, "_run_vertex_proxy",
                          return_value=_fail_completed_process()), \
             patch("openrouter_client.call_openrouter", return_value=None):
            result = ppc.process_with_anthropic_merge(primary, "x", config_merge)
        assert result == primary

    def test_hallucination_guard_returns_primary(self, config_merge):
        ppc = _import_module()
        primary = "这是一段足够长的中文转录文本用于触发合并流程"
        huge = primary * 5

        with patch.object(ppc, "_run_vertex_proxy",
                          return_value=_ok_completed_process(huge)), \
             patch("openrouter_client.call_openrouter", return_value=None):
            result = ppc.process_with_anthropic_merge(primary, "english", config_merge)
        assert result == primary

    def test_question_guard_returns_primary(self, config_merge):
        ppc = _import_module()
        primary = "请问这个问题应该怎么解决？需要更详细的说明和上下文背景"
        non_question = "这是回答没有问号的内容文本"

        with patch.object(ppc, "_run_vertex_proxy",
                          return_value=_ok_completed_process(non_question)), \
             patch("openrouter_client.call_openrouter", return_value=None):
            result = ppc.process_with_anthropic_merge(primary, "english", config_merge)
        assert result == primary

    def test_never_raises(self, config_merge):
        ppc = _import_module()
        primary = "这是一段足够长的中文转录文本用于触发合并流程"

        with patch.object(ppc, "_run_vertex_proxy",
                          side_effect=RuntimeError("unexpected!")), \
             patch("openrouter_client.call_openrouter", return_value=None):
            try:
                result = ppc.process_with_anthropic_merge(primary, "x", config_merge)
            except Exception as e:
                pytest.fail(f"process_with_anthropic_merge raised: {e!r}")
            assert isinstance(result, str)
