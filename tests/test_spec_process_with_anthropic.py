"""Clean-room unit tests for post_processor_configs.process_with_anthropic{,_merge}.

Derived from FUNCTION_SPEC.md Module B. Mirrors the contract of
process_with_vertex_ai / process_with_gemini_merge but routes via
anthropic_proxy with the JSON key `max_tokens` (not `max_output_tokens`).
"""

import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _base_config(**overrides):
    cfg = {
        "ssh_host": "oracle-cloud",
        "proxy_script": "~/anthropic_proxy.py",
        "model": "claude-haiku-4-5-20251001",
        "timeout": 15,
        "min_text_len": 15,
        "system_prompt": "you are an ASR editor",
        "user_prompt_template": "Input: {text}",
    }
    cfg.update(overrides)
    return cfg


def _proxy_result(returncode=0, stdout="", stderr=""):
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


def _patch_proxy(monkeypatch, *, returncode=0, stdout="", stderr="",
                 timeout_exc=False):
    """Patch _run_vertex_proxy. Returns the MagicMock."""
    import post_processor_configs as ppc

    if timeout_exc:
        mock = MagicMock(side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=15))
    else:
        mock = MagicMock(return_value=_proxy_result(returncode, stdout, stderr))
    monkeypatch.setattr(ppc, "_run_vertex_proxy", mock)
    return mock


def _patch_openrouter(monkeypatch, return_value):
    """Patch openrouter_client.call_openrouter on every bind point we can see."""
    import post_processor_configs as ppc
    import openrouter_client

    fake = MagicMock(return_value=return_value)
    monkeypatch.setattr(openrouter_client, "call_openrouter", fake, raising=False)
    if hasattr(ppc, "call_openrouter"):
        monkeypatch.setattr(ppc, "call_openrouter", fake, raising=False)
    return fake


def _patch_notify(monkeypatch):
    """Patch voice_input.notify."""
    import voice_input
    import post_processor_configs as ppc

    fake = MagicMock()
    monkeypatch.setattr(voice_input, "notify", fake, raising=False)
    if hasattr(ppc, "notify"):
        monkeypatch.setattr(ppc, "notify", fake, raising=False)
    return fake


def _stdin_payload(call):
    """Extract the JSON payload from a _run_vertex_proxy positional call."""
    args, _kwargs = call.call_args
    assert len(args) >= 2, f"_run_vertex_proxy expects >=2 positional args: {args!r}"
    stdin_data = args[1]
    return json.loads(stdin_data)


def _cmd_from(call):
    args, _kwargs = call.call_args
    return args[0]


# ---------------------------------------------------------------------------
# process_with_anthropic — single-text
# ---------------------------------------------------------------------------


class TestProcessWithAnthropicAttribute:
    def test_function_exists(self):
        import post_processor_configs as ppc
        assert callable(getattr(ppc, "process_with_anthropic", None)), \
            "process_with_anthropic must be defined in post_processor_configs"


class TestProcessWithAnthropicEarlyReturns:
    def test_empty_text_returns_empty(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch)
        out = ppc.process_with_anthropic("", _base_config(), "")
        assert out == ""
        mock.assert_not_called()

    def test_short_text_returns_input(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch)
        out = ppc.process_with_anthropic("abc", _base_config(), "")
        assert out == "abc"
        mock.assert_not_called()


class TestProcessWithAnthropicSshCmd:
    def test_cmd_shape(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        ppc.process_with_anthropic("this is a long enough text", _base_config(), "")
        cmd = _cmd_from(mock)
        assert cmd[:3] == ["ssh", "-o", "ConnectTimeout=5"]
        assert "oracle-cloud" in cmd
        assert "python3" in cmd
        assert "~/anthropic_proxy.py" in cmd


class TestProcessWithAnthropicNormal:
    def test_success_returns_stripped_stdout(self, monkeypatch):
        import post_processor_configs as ppc
        _patch_proxy(monkeypatch, stdout="你好这是测试文本\n")
        out = ppc.process_with_anthropic("你好呃这是测试文本", _base_config(), "")
        assert out == "你好这是测试文本"

    def test_stdin_payload_uses_max_tokens_key(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        ppc.process_with_anthropic("this is a long enough text", _base_config(), "")
        payload = _stdin_payload(mock)
        assert "max_tokens" in payload
        assert "max_output_tokens" not in payload, \
            "Anthropic SDK uses max_tokens, not max_output_tokens"

    def test_stdin_payload_has_required_keys(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        ppc.process_with_anthropic("this is a long enough text", _base_config(), "")
        payload = _stdin_payload(mock)
        for k in ("system_prompt", "user_input", "model", "max_tokens"):
            assert k in payload, f"missing payload key: {k}"

    def test_default_model_used_when_missing(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        cfg = _base_config()
        cfg.pop("model")
        ppc.process_with_anthropic("this is a long enough text", cfg, "")
        payload = _stdin_payload(mock)
        assert payload["model"] == "claude-haiku-4-5-20251001"

    def test_explicit_model_passed_through(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        cfg = _base_config(model="claude-opus-4-1-20250805")
        ppc.process_with_anthropic("this is a long enough text", cfg, "")
        payload = _stdin_payload(mock)
        assert payload["model"] == "claude-opus-4-1-20250805"


class TestProcessWithAnthropicMaxTokensFormula:
    @pytest.mark.parametrize("user_input_len,expected", [
        (50_000, 8192),    # cap
        (1500, 1500),      # linear
        (200, 512),        # floor (>= min_text_len but < 512)
    ])
    def test_max_tokens_formula(self, monkeypatch, user_input_len, expected):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        cfg = _base_config()
        # user_prompt_template adds "Input: " prefix (7 chars). Use raw template
        # so we can control len(user_input).
        cfg["user_prompt_template"] = "{text}"
        text = "x" * user_input_len
        ppc.process_with_anthropic(text, cfg, "")
        payload = _stdin_payload(mock)
        assert payload["max_tokens"] == expected


class TestProcessWithAnthropicGlossary:
    def test_glossary_ctx_appended(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        ppc.process_with_anthropic("this is a long enough text",
                                   _base_config(), "Commonly used: A, B")
        payload = _stdin_payload(mock)
        assert payload["system_prompt"].endswith("\n\nCommonly used: A, B")

    def test_empty_glossary_no_change(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        ppc.process_with_anthropic("this is a long enough text", _base_config(), "")
        payload = _stdin_payload(mock)
        # system_prompt should not have a trailing blank-line append
        assert not payload["system_prompt"].endswith("\n\n")


class TestProcessWithAnthropicFallback:
    def test_rc_nonzero_openrouter_success(self, monkeypatch):
        import post_processor_configs as ppc
        _patch_proxy(monkeypatch, returncode=1, stderr="proxy failed")
        _patch_openrouter(monkeypatch, "recovered text")
        out = ppc.process_with_anthropic("this is a long enough text",
                                         _base_config(), "")
        assert out == "recovered text"

    def test_rc_nonzero_openrouter_none(self, monkeypatch):
        import post_processor_configs as ppc
        _patch_proxy(monkeypatch, returncode=1, stderr="proxy failed")
        _patch_openrouter(monkeypatch, None)
        notify = _patch_notify(monkeypatch)
        text = "this is a long enough text"
        out = ppc.process_with_anthropic(text, _base_config(), "")
        assert out == text
        notify.assert_called_once()
        # 3rd-positional or kwargs — assert message includes the required text
        call = notify.call_args
        all_args = list(call.args) + list(call.kwargs.values())
        joined = " ".join(str(a) for a in all_args)
        assert "Anthropic + OpenRouter both failed" in joined

    def test_timeout_openrouter_success(self, monkeypatch):
        import post_processor_configs as ppc
        _patch_proxy(monkeypatch, timeout_exc=True)
        _patch_openrouter(monkeypatch, "fast")
        out = ppc.process_with_anthropic("this is a long enough text",
                                         _base_config(), "")
        assert out == "fast"

    def test_timeout_openrouter_none(self, monkeypatch):
        import post_processor_configs as ppc
        _patch_proxy(monkeypatch, timeout_exc=True)
        _patch_openrouter(monkeypatch, None)
        notify = _patch_notify(monkeypatch)
        text = "this is a long enough text"
        out = ppc.process_with_anthropic(text, _base_config(), "")
        assert out == text
        notify.assert_called_once()


class TestProcessWithAnthropicGuards:
    def test_hallucination_guard(self, monkeypatch):
        import post_processor_configs as ppc
        text = "this is a long input"  # len(text) = 20
        bloated = "x" * 200             # > 2 * len(text)
        _patch_proxy(monkeypatch, stdout=bloated)
        out = ppc.process_with_anthropic(text, _base_config(), "")
        assert out == text

    def test_question_dropped(self, monkeypatch):
        import post_processor_configs as ppc
        text = "你好这是测试文本吗？真的"
        _patch_proxy(monkeypatch, stdout="你好这是测试文本，真的。")
        out = ppc.process_with_anthropic(text, _base_config(), "")
        assert out == text

    def test_question_preserved_as_ascii(self, monkeypatch):
        import post_processor_configs as ppc
        text = "你好这是测试文本？"
        out_text = "你好这是测试文本?"
        _patch_proxy(monkeypatch, stdout=out_text)
        out = ppc.process_with_anthropic(text, _base_config(), "")
        assert out == out_text


class TestProcessWithAnthropicRobustness:
    def test_never_raises_on_missing_prompt_file(self, monkeypatch, tmp_path):
        import post_processor_configs as ppc
        _patch_proxy(monkeypatch, stdout="ok")
        cfg = _base_config()
        cfg["system_prompt_file"] = "prompts/this-file-does-not-exist.txt"
        cfg.pop("system_prompt", None)
        text = "this is a long enough text"
        # Must NOT raise per spec; either returns text or proxy output.
        out = ppc.process_with_anthropic(text, cfg, "")
        assert isinstance(out, str)

    def test_never_raises_on_subprocess_exception(self, monkeypatch):
        import post_processor_configs as ppc

        def boom(*args, **kwargs):
            raise OSError("ssh failed")

        monkeypatch.setattr(ppc, "_run_vertex_proxy", boom)
        _patch_openrouter(monkeypatch, None)
        _patch_notify(monkeypatch)
        text = "this is a long enough text"
        out = ppc.process_with_anthropic(text, _base_config(), "")
        assert out == text


# ---------------------------------------------------------------------------
# process_with_anthropic_merge — dual-ASR
# ---------------------------------------------------------------------------


class TestProcessWithAnthropicMergeAttribute:
    def test_function_exists(self):
        import post_processor_configs as ppc
        assert callable(getattr(ppc, "process_with_anthropic_merge", None)), \
            "process_with_anthropic_merge must be defined in post_processor_configs"


def _merge_config(**overrides):
    cfg = {
        "ssh_host": "oracle-cloud",
        "proxy_script": "~/anthropic_proxy.py",
        "model": "claude-haiku-4-5-20251001",
        "timeout": 15,
        "min_text_len": 15,
        "system_prompt": "you are a merge editor",
    }
    cfg.update(overrides)
    return cfg


class TestProcessWithAnthropicMergeEarlyReturns:
    def test_empty_primary_with_secondary(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        out = ppc.process_with_anthropic_merge("", "fallback text",
                                               _merge_config(), "")
        assert out == "fallback text"
        mock.assert_not_called()

    def test_empty_primary_no_secondary(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        out = ppc.process_with_anthropic_merge("", None, _merge_config(), "")
        assert out == ""
        mock.assert_not_called()

    def test_empty_primary_empty_secondary(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        out = ppc.process_with_anthropic_merge("", "", _merge_config(), "")
        assert out == ""
        mock.assert_not_called()

    def test_short_primary_longer_secondary(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        out = ppc.process_with_anthropic_merge("hi", "this is longer",
                                               _merge_config(), "")
        assert out == "this is longer"
        mock.assert_not_called()

    def test_short_primary_no_secondary(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        out = ppc.process_with_anthropic_merge("hi", None, _merge_config(), "")
        assert out == "hi"
        mock.assert_not_called()

    def test_short_primary_secondary_shorter(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        out = ppc.process_with_anthropic_merge("hi", "x", _merge_config(), "")
        assert out == "hi"
        mock.assert_not_called()


class TestProcessWithAnthropicMergeUserInput:
    def test_user_input_dual(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="merged ok")
        ppc.process_with_anthropic_merge(
            "今天天气很好这是测试文本", "It's sunny today",
            _merge_config(), "")
        payload = _stdin_payload(mock)
        assert payload["user_input"] == \
            "Chinese ASR: 今天天气很好这是测试文本\nEnglish ASR: It's sunny today"

    def test_user_input_single_when_secondary_none(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="merged ok")
        ppc.process_with_anthropic_merge(
            "hello world这是测试文本", None, _merge_config(), "")
        payload = _stdin_payload(mock)
        assert payload["user_input"] == "Chinese ASR: hello world这是测试文本"
        assert "English ASR:" not in payload["user_input"]


class TestProcessWithAnthropicMergePayload:
    def test_payload_uses_max_tokens_key(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        ppc.process_with_anthropic_merge("this is a long enough text",
                                         "sec text", _merge_config(), "")
        payload = _stdin_payload(mock)
        assert "max_tokens" in payload
        assert "max_output_tokens" not in payload

    def test_default_model_used_when_missing(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        cfg = _merge_config()
        cfg.pop("model")
        ppc.process_with_anthropic_merge("this is a long enough text",
                                         "sec", cfg, "")
        payload = _stdin_payload(mock)
        assert payload["model"] == "claude-haiku-4-5-20251001"

    def test_glossary_appended_to_system(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        ppc.process_with_anthropic_merge("this is a long enough text",
                                         "sec", _merge_config(),
                                         "terms: A,B")
        payload = _stdin_payload(mock)
        assert payload["system_prompt"].endswith("\n\nterms: A,B")

    def test_max_tokens_cap(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        primary = "x" * 20_000
        ppc.process_with_anthropic_merge(primary, None, _merge_config(), "")
        payload = _stdin_payload(mock)
        assert payload["max_tokens"] == 8192

    def test_max_tokens_floor(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        primary = "x" * 30  # >= min_text_len but user_input < 512
        ppc.process_with_anthropic_merge(primary, None, _merge_config(), "")
        payload = _stdin_payload(mock)
        assert payload["max_tokens"] == 512


class TestProcessWithAnthropicMergeSshCmd:
    def test_cmd_shape(self, monkeypatch):
        import post_processor_configs as ppc
        mock = _patch_proxy(monkeypatch, stdout="ok")
        ppc.process_with_anthropic_merge("this is a long enough text",
                                         "sec", _merge_config(), "")
        cmd = _cmd_from(mock)
        assert cmd[:3] == ["ssh", "-o", "ConnectTimeout=5"]
        assert "oracle-cloud" in cmd
        assert "python3" in cmd
        assert "~/anthropic_proxy.py" in cmd


class TestProcessWithAnthropicMergeFallback:
    def test_rc_nonzero_openrouter_success(self, monkeypatch):
        import post_processor_configs as ppc
        _patch_proxy(monkeypatch, returncode=1, stderr="error")
        _patch_openrouter(monkeypatch, "recovered")
        out = ppc.process_with_anthropic_merge(
            "this is a long enough text", "sec", _merge_config(), "")
        assert out == "recovered"

    def test_rc_nonzero_openrouter_none_returns_primary(self, monkeypatch):
        import post_processor_configs as ppc
        _patch_proxy(monkeypatch, returncode=1, stderr="error")
        _patch_openrouter(monkeypatch, None)
        notify = _patch_notify(monkeypatch)
        primary = "this is a long enough text"
        out = ppc.process_with_anthropic_merge(primary, "sec",
                                               _merge_config(), "")
        assert out == primary
        notify.assert_called_once()
        joined = " ".join(str(a) for a in
                          list(notify.call_args.args) +
                          list(notify.call_args.kwargs.values()))
        assert "Anthropic merge + OpenRouter both failed" in joined

    def test_timeout_openrouter_success(self, monkeypatch):
        import post_processor_configs as ppc
        _patch_proxy(monkeypatch, timeout_exc=True)
        _patch_openrouter(monkeypatch, "fast")
        out = ppc.process_with_anthropic_merge(
            "this is a long enough text", "sec", _merge_config(), "")
        assert out == "fast"

    def test_timeout_openrouter_none_returns_primary(self, monkeypatch):
        import post_processor_configs as ppc
        _patch_proxy(monkeypatch, timeout_exc=True)
        _patch_openrouter(monkeypatch, None)
        notify = _patch_notify(monkeypatch)
        primary = "this is a long enough text"
        out = ppc.process_with_anthropic_merge(primary, "sec",
                                               _merge_config(), "")
        assert out == primary
        notify.assert_called_once()


class TestProcessWithAnthropicMergeGuards:
    def test_hallucination_guard(self, monkeypatch):
        import post_processor_configs as ppc
        primary = "this is a long input"  # 20 chars
        bloated = "y" * 200
        _patch_proxy(monkeypatch, stdout=bloated)
        out = ppc.process_with_anthropic_merge(primary, None,
                                               _merge_config(), "")
        assert out == primary

    def test_question_dropped_returns_primary(self, monkeypatch):
        import post_processor_configs as ppc
        primary = "你好这是测试文本吗？真的"
        _patch_proxy(monkeypatch, stdout="你好这是测试文本，真的。")
        out = ppc.process_with_anthropic_merge(primary, None,
                                               _merge_config(), "")
        assert out == primary

    def test_question_preserved_as_ascii(self, monkeypatch):
        import post_processor_configs as ppc
        primary = "你好这是测试文本？"
        new_text = "你好这是测试文本?"
        _patch_proxy(monkeypatch, stdout=new_text)
        out = ppc.process_with_anthropic_merge(primary, None,
                                               _merge_config(), "")
        assert out == new_text


class TestProcessWithAnthropicMergeRobustness:
    def test_never_raises_on_missing_prompt_file(self, monkeypatch):
        import post_processor_configs as ppc
        _patch_proxy(monkeypatch, stdout="ok")
        cfg = _merge_config()
        cfg["system_prompt_file"] = "prompts/missing-merge.txt"
        cfg.pop("system_prompt", None)
        out = ppc.process_with_anthropic_merge(
            "this is a long enough text", None, cfg, "")
        assert isinstance(out, str)

    def test_never_raises_on_subprocess_exception(self, monkeypatch):
        import post_processor_configs as ppc

        def boom(*args, **kwargs):
            raise OSError("ssh failed")

        monkeypatch.setattr(ppc, "_run_vertex_proxy", boom)
        _patch_openrouter(monkeypatch, None)
        _patch_notify(monkeypatch)
        primary = "this is a long enough text"
        out = ppc.process_with_anthropic_merge(primary, None,
                                               _merge_config(), "")
        assert out == primary
