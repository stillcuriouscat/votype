"""Tests for 429 fallback to alternative Gemini model — US-002."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from post_processor_configs import _run_vertex_proxy, process_with_vertex_ai, process_with_gemini_merge
from post_processor_presets import POST_PROCESSOR_PRESETS


# Minimal configs for testing
VERTEX_AI_CONFIG = {
    "ssh_host": "oracle-cloud",
    "proxy_script": "~/vertex_proxy.py",
    "model": "gemini-2.5-flash",
    "fallback_model": "gemini-2.5-flash-lite",
    "vertex_region": "us-central1",
    "timeout": 15,
    "min_text_len": 15,
    "system_prompt": "You are an ASR correction tool.",
}

MERGE_CONFIG = {
    "ssh_host": "oracle-cloud",
    "proxy_script": "~/vertex_proxy.py",
    "model": "gemini-2.5-flash",
    "fallback_model": "gemini-2.5-flash-lite",
    "vertex_region": "us-central1",
    "timeout": 15,
    "min_text_len": 15,
    "system_prompt": "You are a merge editor.",
}


def _make_429_result(stderr_msg="429 RESOURCE_EXHAUSTED"):
    """Create a mock subprocess result simulating a 429 error."""
    return MagicMock(returncode=1, stdout="", stderr=stderr_msg)


def _make_success_result(stdout="polished text"):
    """Create a mock subprocess result simulating success."""
    return MagicMock(returncode=0, stdout=stdout, stderr="")


def _make_non_429_error_result():
    """Create a mock subprocess result simulating a non-429 error."""
    return MagicMock(returncode=1, stdout="", stderr="Connection refused")


class TestRunVertexProxyFallback:
    """Unit tests for _run_vertex_proxy fallback_model behavior."""

    @patch("post_processor_configs.time.sleep")
    @patch("post_processor_configs.subprocess.run")
    def test_fallback_tried_after_429_retry_failure(self, mock_run, mock_sleep):
        """After initial 429 and retry 429, fallback model is tried."""
        mock_run.side_effect = [
            _make_429_result(),   # Initial: 429
            _make_429_result(),   # Retry: 429
            _make_success_result("fallback output"),  # Fallback: success
        ]
        cmd = ["ssh", "host", "python3", "proxy.py"]
        stdin_data = json.dumps({"model": "gemini-2.5-flash", "system_prompt": "test", "user_input": "test", "region": "us-central1"})

        result = _run_vertex_proxy(cmd, stdin_data, timeout=15, fallback_model="gemini-2.5-flash-lite")

        assert mock_run.call_count == 3
        assert result.stdout == "fallback output"
        assert result.returncode == 0

        # Verify fallback call used the fallback model in stdin
        third_call_stdin = mock_run.call_args_list[2].kwargs.get("input") or mock_run.call_args_list[2][1].get("input", "")
        if not third_call_stdin:
            third_call_stdin = mock_run.call_args_list[2].kwargs["input"]
        fallback_payload = json.loads(third_call_stdin)
        assert fallback_payload["model"] == "gemini-2.5-flash-lite"

    @patch("post_processor_configs.time.sleep")
    @patch("post_processor_configs.subprocess.run")
    def test_no_fallback_when_fallback_model_is_none(self, mock_run, mock_sleep):
        """When fallback_model is None, no fallback attempted after 429 retry failure."""
        mock_run.side_effect = [
            _make_429_result(),   # Initial: 429
            _make_429_result(),   # Retry: 429
        ]
        cmd = ["ssh", "host", "python3", "proxy.py"]
        stdin_data = json.dumps({"model": "gemini-2.5-flash", "system_prompt": "test", "user_input": "test", "region": "us-central1"})

        result = _run_vertex_proxy(cmd, stdin_data, timeout=15, fallback_model=None)

        assert mock_run.call_count == 2  # Only initial + retry, no fallback
        assert result.returncode == 1

    @patch("post_processor_configs.time.sleep")
    @patch("post_processor_configs.subprocess.run")
    def test_no_fallback_on_non_429_errors(self, mock_run, mock_sleep):
        """Non-429 errors should not trigger fallback."""
        mock_run.return_value = _make_non_429_error_result()
        cmd = ["ssh", "host", "python3", "proxy.py"]
        stdin_data = json.dumps({"model": "gemini-2.5-flash", "system_prompt": "test", "user_input": "test", "region": "us-central1"})

        result = _run_vertex_proxy(cmd, stdin_data, timeout=15, fallback_model="gemini-2.5-flash-lite")

        # Non-429 error: no retry, no fallback
        assert mock_run.call_count == 1
        assert result.returncode == 1

    @patch("post_processor_configs.time.sleep")
    @patch("post_processor_configs.subprocess.run")
    def test_fallback_success_returns_fallback_result(self, mock_run, mock_sleep):
        """Successful fallback returns the fallback response."""
        mock_run.side_effect = [
            _make_429_result(),
            _make_429_result(),
            _make_success_result("fallback text here"),
        ]
        cmd = ["ssh", "host", "python3", "proxy.py"]
        stdin_data = json.dumps({"model": "gemini-2.5-flash", "system_prompt": "test", "user_input": "test", "region": "us-central1"})

        result = _run_vertex_proxy(cmd, stdin_data, timeout=15, fallback_model="gemini-2.5-flash-lite")

        assert result.returncode == 0
        assert result.stdout == "fallback text here"

    @patch("post_processor_configs.time.sleep")
    @patch("post_processor_configs.subprocess.run")
    def test_fallback_failure_returns_failed_result(self, mock_run, mock_sleep):
        """When fallback also fails, returns the fallback's failed result."""
        mock_run.side_effect = [
            _make_429_result(),
            _make_429_result(),
            MagicMock(returncode=1, stdout="", stderr="fallback also failed"),
        ]
        cmd = ["ssh", "host", "python3", "proxy.py"]
        stdin_data = json.dumps({"model": "gemini-2.5-flash", "system_prompt": "test", "user_input": "test", "region": "us-central1"})

        result = _run_vertex_proxy(cmd, stdin_data, timeout=15, fallback_model="gemini-2.5-flash-lite")

        assert result.returncode == 1
        assert "fallback also failed" in result.stderr

    @patch("post_processor_configs.time.sleep")
    @patch("post_processor_configs.subprocess.run")
    def test_no_fallback_when_retry_succeeds(self, mock_run, mock_sleep):
        """When retry succeeds, no fallback needed."""
        mock_run.side_effect = [
            _make_429_result(),             # Initial: 429
            _make_success_result("retry ok"),  # Retry: success
        ]
        cmd = ["ssh", "host", "python3", "proxy.py"]
        stdin_data = json.dumps({"model": "gemini-2.5-flash", "system_prompt": "test", "user_input": "test", "region": "us-central1"})

        result = _run_vertex_proxy(cmd, stdin_data, timeout=15, fallback_model="gemini-2.5-flash-lite")

        assert mock_run.call_count == 2  # Initial + retry, no fallback
        assert result.returncode == 0
        assert result.stdout == "retry ok"

    @patch("post_processor_configs.time.sleep")
    @patch("post_processor_configs.subprocess.run")
    def test_fallback_uses_same_timeout(self, mock_run, mock_sleep):
        """Fallback attempt should use the same timeout as original."""
        mock_run.side_effect = [
            _make_429_result(),
            _make_429_result(),
            _make_success_result("ok"),
        ]
        cmd = ["ssh", "host", "python3", "proxy.py"]
        stdin_data = json.dumps({"model": "gemini-2.5-flash", "system_prompt": "test", "user_input": "test", "region": "us-central1"})

        _run_vertex_proxy(cmd, stdin_data, timeout=42, fallback_model="gemini-2.5-flash-lite")

        # All three calls should use the same timeout
        for c in mock_run.call_args_list:
            assert c.kwargs["timeout"] == 42

    @patch("post_processor_configs.time.sleep")
    @patch("post_processor_configs.subprocess.run")
    def test_fallback_log_contains_model_name(self, mock_run, mock_sleep, caplog):
        """Fallback attempt should log with the fallback model name."""
        import logging
        mock_run.side_effect = [
            _make_429_result(),
            _make_429_result(),
            _make_success_result("ok"),
        ]
        cmd = ["ssh", "host", "python3", "proxy.py"]
        stdin_data = json.dumps({"model": "gemini-2.5-flash", "system_prompt": "test", "user_input": "test", "region": "us-central1"})

        with caplog.at_level(logging.INFO):
            _run_vertex_proxy(cmd, stdin_data, timeout=15, fallback_model="gemini-2.5-flash-lite")

        # Check log contains fallback trace
        fallback_logs = [r for r in caplog.records if "fallback to gemini-2.5-flash-lite" in r.message]
        assert len(fallback_logs) == 1


class TestPresetsFallbackModel:
    """Verify gemini-fix and gemini-merge presets have fallback_model field."""

    def test_gemini_fix_has_fallback_model(self):
        config = POST_PROCESSOR_PRESETS["gemini-fix"]["config"]
        assert "fallback_model" in config
        assert config["fallback_model"] == "gemini-2.5-flash-lite"

    def test_gemini_merge_has_fallback_model(self):
        config = POST_PROCESSOR_PRESETS["gemini-merge"]["config"]
        assert "fallback_model" in config
        assert config["fallback_model"] == "gemini-2.5-flash-lite"

    def test_haiku_fix_no_fallback_model(self):
        """haiku-fix uses SSH claude, should not have fallback_model."""
        config = POST_PROCESSOR_PRESETS["haiku-fix"]["config"]
        assert "fallback_model" not in config

    def test_none_preset_no_fallback(self):
        """'none' preset has no config, should not have fallback_model."""
        preset = POST_PROCESSOR_PRESETS["none"]
        assert "fallback_model" not in preset.get("config", {})


class TestIntegration429FallbackFlow:
    """Integration test: mock subprocess to simulate 429 → retry 429 → fallback success."""

    @patch("post_processor_configs.time.sleep")
    @patch("post_processor_configs.subprocess.run")
    def test_vertex_ai_429_retry_fallback_success(self, mock_run, mock_sleep):
        """process_with_vertex_ai: 429 → retry 429 → fallback success returns fallback text."""
        text = "a" * 50
        mock_run.side_effect = [
            _make_429_result(),
            _make_429_result(),
            _make_success_result("fallback corrected text"),
        ]
        result = process_with_vertex_ai(text, VERTEX_AI_CONFIG)

        assert result == "fallback corrected text"
        assert mock_run.call_count == 3

    @patch("post_processor_configs.time.sleep")
    @patch("post_processor_configs.subprocess.run")
    def test_gemini_merge_429_retry_fallback_success(self, mock_run, mock_sleep):
        """process_with_gemini_merge: 429 → retry 429 → fallback success returns fallback text."""
        primary = "a" * 50
        secondary = "b" * 50
        mock_run.side_effect = [
            _make_429_result(),
            _make_429_result(),
            _make_success_result("fallback merged text"),
        ]
        result = process_with_gemini_merge(primary, secondary, MERGE_CONFIG)

        assert result == "fallback merged text"
        assert mock_run.call_count == 3

    @patch("voice_input.notify")
    @patch("post_processor_configs.time.sleep")
    @patch("post_processor_configs.subprocess.run")
    def test_vertex_ai_429_all_fail_returns_original(self, mock_run, mock_sleep, mock_notify):
        """process_with_vertex_ai: 429 → retry 429 → fallback 429 → returns original text."""
        text = "a" * 50
        mock_run.side_effect = [
            _make_429_result(),
            _make_429_result(),
            _make_429_result(),  # Fallback also fails
        ]
        result = process_with_vertex_ai(text, VERTEX_AI_CONFIG)

        assert result == text  # Falls back to original
        assert mock_run.call_count == 3

    @patch("post_processor_configs.time.sleep")
    @patch("post_processor_configs.subprocess.run")
    def test_vertex_ai_no_fallback_without_config(self, mock_run, mock_sleep):
        """process_with_vertex_ai without fallback_model in config: no fallback attempt."""
        config = {**VERTEX_AI_CONFIG}
        del config["fallback_model"]
        text = "a" * 50
        mock_run.side_effect = [
            _make_429_result(),
            _make_429_result(),
        ]

        with patch("voice_input.notify"):
            result = process_with_vertex_ai(text, config)

        assert result == text
        assert mock_run.call_count == 2  # No fallback

    @patch("post_processor_configs.time.sleep")
    @patch("post_processor_configs.subprocess.run")
    def test_normal_path_unaffected_by_fallback_config(self, mock_run, mock_sleep):
        """Normal success path should work the same with fallback_model configured."""
        text = "a" * 50
        mock_run.return_value = _make_success_result("clean text")

        result = process_with_vertex_ai(text, VERTEX_AI_CONFIG)

        assert result == "clean text"
        assert mock_run.call_count == 1  # No retry, no fallback
