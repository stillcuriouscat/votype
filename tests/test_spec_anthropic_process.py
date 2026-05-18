"""Clean-room unit tests for US-002: process_with_anthropic + process_with_anthropic_merge.

Derived from FUNCTION_SPEC.md Module B behavior tables.
Tests fallback chain (Anthropic → OpenRouter → notify → fallback to input),
guards (hallucination, question mark), and max_tokens formula.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from post_processor_configs import (
    process_with_anthropic,
    process_with_anthropic_merge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIX_CONFIG = {
    "ssh_host": "oracle-cloud",
    "proxy_script": "~/anthropic_proxy.py",
    "model": "claude-haiku-4-5-20251001",
    "timeout": 15,
    "min_text_len": 15,
    "vocab_min_count": 3,
}

MERGE_CONFIG = {
    "ssh_host": "oracle-cloud",
    "proxy_script": "~/anthropic_proxy.py",
    "model": "claude-haiku-4-5-20251001",
    "timeout": 15,
    "min_text_len": 15,
    "vocab_min_count": 3,
}

LONG_TEXT = "这是一段足够长的测试文本用来通过最小长度检查"
LONG_PRIMARY = "这是一段中文ASR的转录结果足够长可以触发处理流程"
LONG_SECONDARY = "this is english ASR text"


def _make_proxy_result(rc: int = 0, stdout: str = "", stderr: str = ""):
    m = MagicMock(spec=subprocess.CompletedProcess)
    m.returncode = rc
    m.stdout = stdout
    m.stderr = stderr
    return m


# ===========================================================================
# process_with_anthropic
# ===========================================================================

class TestProcessWithAnthropic:
    def test_empty_text_returns_empty(self):
        assert process_with_anthropic("", FIX_CONFIG) == ""

    def test_short_text_returns_input(self, caplog):
        with caplog.at_level("INFO"):
            assert process_with_anthropic("abc", FIX_CONFIG) == "abc"
        assert any("below min_text_len" in r.message for r in caplog.records)

    def test_success_primary(self, caplog):
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "polished text", "")) as mock_run:
            with caplog.at_level("INFO"):
                result = process_with_anthropic(LONG_TEXT, FIX_CONFIG)
        assert result == "polished text"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:3] == ["ssh", "-o", "ConnectTimeout=5"]
        assert cmd[3] == "oracle-cloud"
        assert cmd[4:] == ["python3", "~/anthropic_proxy.py"]
        stdin_data = mock_run.call_args[0][1]
        payload = json.loads(stdin_data)
        assert "max_tokens" in payload
        assert "max_output_tokens" not in payload
        assert payload["model"] == "claude-haiku-4-5-20251001"
        assert any("anthropic-fix success" in r.message for r in caplog.records)

    def test_max_tokens_floor(self):
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "x", "")) as mock_run:
            process_with_anthropic(LONG_TEXT, FIX_CONFIG)
        payload = json.loads(mock_run.call_args[0][1])
        assert payload["max_tokens"] == 512

    def test_max_tokens_cap(self):
        long = "x" * 20000
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "y", "")) as mock_run:
            process_with_anthropic(long, FIX_CONFIG)
        payload = json.loads(mock_run.call_args[0][1])
        assert payload["max_tokens"] == 8192

    def test_max_tokens_linear(self):
        text = "x" * 1500
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "z", "")) as mock_run:
            process_with_anthropic(text, FIX_CONFIG)
        payload = json.loads(mock_run.call_args[0][1])
        assert payload["max_tokens"] == 1500

    def test_default_model_when_missing(self):
        cfg = {k: v for k, v in FIX_CONFIG.items() if k != "model"}
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "y", "")) as mock_run:
            process_with_anthropic(LONG_TEXT, cfg)
        payload = json.loads(mock_run.call_args[0][1])
        assert payload["model"] == "claude-haiku-4-5-20251001"

    def test_glossary_ctx_appended(self):
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "y", "")) as mock_run:
            process_with_anthropic(LONG_TEXT, FIX_CONFIG,
                                   glossary_ctx="Commonly used terms: A, B")
        payload = json.loads(mock_run.call_args[0][1])
        assert payload["system_prompt"].endswith("\n\nCommonly used terms: A, B")

    def test_openrouter_fallback_on_rc_nonzero(self, caplog):
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(1, "", "boom")), \
             patch("openrouter_client.call_openrouter",
                   return_value="recovered text"):
            with caplog.at_level("INFO"):
                result = process_with_anthropic(LONG_TEXT, FIX_CONFIG)
        assert result == "recovered text"
        assert any("[OPENROUTER] fallback success for anthropic-fix" in r.message
                   for r in caplog.records)

    def test_openrouter_none_calls_notify(self):
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(1, "", "boom")), \
             patch("openrouter_client.call_openrouter", return_value=None), \
             patch("voice_input.notify") as mock_notify:
            result = process_with_anthropic(LONG_TEXT, FIX_CONFIG)
        assert result == LONG_TEXT
        mock_notify.assert_called_once()
        title, msg = mock_notify.call_args[0][0], mock_notify.call_args[0][1]
        assert "Anthropic + OpenRouter both failed" in msg

    def test_timeout_fallback_to_openrouter(self, caplog):
        with patch("post_processor_configs._run_vertex_proxy",
                   side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=15)), \
             patch("openrouter_client.call_openrouter", return_value="fast"):
            with caplog.at_level("WARNING"):
                result = process_with_anthropic(LONG_TEXT, FIX_CONFIG)
        assert result == "fast"
        assert any("timed out" in r.message.lower() for r in caplog.records)

    def test_timeout_and_openrouter_none(self):
        with patch("post_processor_configs._run_vertex_proxy",
                   side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=15)), \
             patch("openrouter_client.call_openrouter", return_value=None), \
             patch("voice_input.notify"):
            result = process_with_anthropic(LONG_TEXT, FIX_CONFIG)
        assert result == LONG_TEXT

    def test_hallucination_guard(self, caplog):
        # output > 2x input → return input
        hallucinated = "y" * (len(LONG_TEXT) * 3)
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, hallucinated, "")):
            with caplog.at_level("WARNING"):
                result = process_with_anthropic(LONG_TEXT, FIX_CONFIG)
        assert result == LONG_TEXT
        assert any("output too long" in r.message for r in caplog.records)

    def test_question_guard_strip(self, caplog):
        # 30-char input; output 30 chars to stay within hallucination guard (2x)
        text_with_q = "你好世界这是一个非常重要的问题？请回答这个问题吧"
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "answer no qmark output text", "")):
            with caplog.at_level("WARNING"):
                result = process_with_anthropic(text_with_q, FIX_CONFIG)
        assert result == text_with_q
        assert any("dropped question marks" in r.message for r in caplog.records)

    def test_question_guard_ascii_qmark_passes(self):
        text_with_q = "你好世界这是一个非常重要的问题？请回答这个问题吧"
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "polished with ? mark", "")):
            result = process_with_anthropic(text_with_q, FIX_CONFIG)
        assert result == "polished with ? mark"

    def test_never_raises_on_subprocess_unexpected_error(self):
        with patch("post_processor_configs._run_vertex_proxy",
                   side_effect=RuntimeError("boom")), \
             patch("openrouter_client.call_openrouter", return_value=None), \
             patch("voice_input.notify"):
            result = process_with_anthropic(LONG_TEXT, FIX_CONFIG)
        assert result == LONG_TEXT

    def test_never_raises_on_prompt_file_missing(self):
        cfg = {**FIX_CONFIG, "system_prompt_file": "nonexistent/path.txt"}
        # Should not raise, returns text
        result = process_with_anthropic(LONG_TEXT, cfg)
        assert result == LONG_TEXT


# ===========================================================================
# process_with_anthropic_merge
# ===========================================================================

class TestProcessWithAnthropicMerge:
    def test_empty_primary_falls_back_to_secondary(self):
        assert process_with_anthropic_merge("", "fallback text", MERGE_CONFIG) == "fallback text"

    def test_empty_primary_no_secondary_returns_empty(self):
        assert process_with_anthropic_merge("", None, MERGE_CONFIG) == ""

    def test_empty_primary_empty_secondary_returns_empty(self):
        assert process_with_anthropic_merge("", "", MERGE_CONFIG) == ""

    def test_short_primary_longer_secondary_returns_secondary(self):
        assert process_with_anthropic_merge("hi", "this is longer", MERGE_CONFIG) == "this is longer"

    def test_short_primary_no_secondary_returns_primary(self):
        assert process_with_anthropic_merge("hi", None, MERGE_CONFIG) == "hi"

    def test_short_primary_short_secondary_returns_primary(self):
        assert process_with_anthropic_merge("hi", "x", MERGE_CONFIG) == "hi"

    def test_normal_merge_dual(self):
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "merged ok output here", "")) as mock_run:
            result = process_with_anthropic_merge(LONG_PRIMARY, LONG_SECONDARY, MERGE_CONFIG)
        assert result == "merged ok output here"
        payload = json.loads(mock_run.call_args[0][1])
        assert payload["user_input"] == f"Chinese ASR: {LONG_PRIMARY}\nEnglish ASR: {LONG_SECONDARY}"
        assert "max_tokens" in payload
        assert "max_output_tokens" not in payload

    def test_normal_merge_secondary_none(self):
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "single polished text out", "")) as mock_run:
            result = process_with_anthropic_merge(LONG_PRIMARY, None, MERGE_CONFIG)
        assert result == "single polished text out"
        payload = json.loads(mock_run.call_args[0][1])
        assert payload["user_input"] == f"Chinese ASR: {LONG_PRIMARY}"

    def test_glossary_ctx_appended(self):
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "y", "")) as mock_run:
            process_with_anthropic_merge(LONG_PRIMARY, None, MERGE_CONFIG,
                                          glossary_ctx="terms: A,B")
        payload = json.loads(mock_run.call_args[0][1])
        assert payload["system_prompt"].endswith("\n\nterms: A,B")

    def test_max_tokens_floor(self):
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "y", "")) as mock_run:
            process_with_anthropic_merge(LONG_PRIMARY, None, MERGE_CONFIG)
        payload = json.loads(mock_run.call_args[0][1])
        assert payload["max_tokens"] == 512

    def test_max_tokens_cap(self):
        long = "x" * 20000
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "y", "")) as mock_run:
            process_with_anthropic_merge(long, None, MERGE_CONFIG)
        payload = json.loads(mock_run.call_args[0][1])
        assert payload["max_tokens"] == 8192

    def test_openrouter_fallback_on_rc_nonzero(self, caplog):
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(1, "", "boom")), \
             patch("openrouter_client.call_openrouter", return_value="recovered"):
            with caplog.at_level("INFO"):
                result = process_with_anthropic_merge(LONG_PRIMARY, LONG_SECONDARY, MERGE_CONFIG)
        assert result == "recovered"
        assert any("[OPENROUTER] fallback success for anthropic-merge" in r.message
                   for r in caplog.records)

    def test_openrouter_none_calls_notify(self):
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(1, "", "boom")), \
             patch("openrouter_client.call_openrouter", return_value=None), \
             patch("voice_input.notify") as mock_notify:
            result = process_with_anthropic_merge(LONG_PRIMARY, LONG_SECONDARY, MERGE_CONFIG)
        assert result == LONG_PRIMARY
        msg = mock_notify.call_args[0][1]
        assert "Anthropic merge + OpenRouter both failed" in msg

    def test_timeout_fallback(self):
        with patch("post_processor_configs._run_vertex_proxy",
                   side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=15)), \
             patch("openrouter_client.call_openrouter", return_value="ok"):
            result = process_with_anthropic_merge(LONG_PRIMARY, None, MERGE_CONFIG)
        assert result == "ok"

    def test_timeout_and_openrouter_none(self):
        with patch("post_processor_configs._run_vertex_proxy",
                   side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=15)), \
             patch("openrouter_client.call_openrouter", return_value=None), \
             patch("voice_input.notify"):
            result = process_with_anthropic_merge(LONG_PRIMARY, None, MERGE_CONFIG)
        assert result == LONG_PRIMARY

    def test_hallucination_guard(self):
        hallucinated = "y" * (len(LONG_PRIMARY) * 3)
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, hallucinated, "")):
            result = process_with_anthropic_merge(LONG_PRIMARY, None, MERGE_CONFIG)
        assert result == LONG_PRIMARY

    def test_question_guard(self):
        primary_with_q = "你好世界这是一个非常重要的问题？请回答这个问题吧"
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "answer no qmark text out", "")):
            result = process_with_anthropic_merge(primary_with_q, None, MERGE_CONFIG)
        assert result == primary_with_q

    def test_question_guard_ascii_passes(self):
        primary_with_q = "你好世界这是一个非常重要的问题？请回答这个问题吧"
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "polished with ? mark", "")):
            result = process_with_anthropic_merge(primary_with_q, None, MERGE_CONFIG)
        assert result == "polished with ? mark"

    def test_default_model_when_missing(self):
        cfg = {k: v for k, v in MERGE_CONFIG.items() if k != "model"}
        with patch("post_processor_configs._run_vertex_proxy",
                   return_value=_make_proxy_result(0, "y", "")) as mock_run:
            process_with_anthropic_merge(LONG_PRIMARY, None, cfg)
        payload = json.loads(mock_run.call_args[0][1])
        assert payload["model"] == "claude-haiku-4-5-20251001"

    def test_never_raises_on_prompt_file_missing(self):
        cfg = {**MERGE_CONFIG, "system_prompt_file": "nonexistent/path.txt"}
        result = process_with_anthropic_merge(LONG_PRIMARY, None, cfg)
        assert result == LONG_PRIMARY
