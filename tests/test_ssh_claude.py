"""Tests for process_with_ssh_claude() in post_processor_configs.py — US-003."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from post_processor_configs import process_with_ssh_claude


# Minimal config matching haiku-fix preset structure
BASE_CONFIG = {
    "ssh_host": "oracle-cloud",
    "claude_path": "/home/ubuntu/.local/bin/claude",
    "model": "claude-haiku-4-5-20251001",
    "timeout": 15,
    "max_text_len": 200,
    "vocab_min_count": 3,
    "system_prompt": "You are an ASR correction tool.",
}


class TestProcessWithSshClaudeSuccess:
    """Tests for successful SSH Claude invocations."""

    @patch("post_processor_configs.subprocess.run")
    def test_success_returns_stripped_stdout(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="  polished text  \n", stderr=""
        )
        result = process_with_ssh_claude("raw text", BASE_CONFIG)
        assert result == "polished text"

    @patch("post_processor_configs.subprocess.run")
    def test_command_construction(self, mock_run):
        """Verify the SSH command includes all required parts."""
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")
        process_with_ssh_claude("hello", BASE_CONFIG)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]

        # Check SSH with ConnectTimeout
        assert cmd[0] == "ssh"
        assert "-o" in cmd
        assert "ConnectTimeout=5" in cmd

        # Check host
        assert "oracle-cloud" in cmd

        # Check claude path and model
        assert "/home/ubuntu/.local/bin/claude" in cmd
        assert "--model" in cmd
        assert "claude-haiku-4-5-20251001" in cmd

        # Check --system-prompt and -p flags
        assert "--system-prompt" in cmd
        assert "-p" in cmd

        # Check stdin input
        assert call_args.kwargs["input"] == "hello"
        assert call_args.kwargs["text"] is True
        assert call_args.kwargs["timeout"] == 15

    @patch("post_processor_configs.subprocess.run")
    def test_glossary_context_appended_to_system_prompt(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")
        process_with_ssh_claude("hello", BASE_CONFIG, glossary_ctx="Commonly used terms: Ralph")

        cmd = mock_run.call_args[0][0]
        sp_idx = cmd.index("--system-prompt")
        system_prompt_arg = cmd[sp_idx + 1]
        # shlex.quote wraps in single quotes
        assert "Commonly used terms: Ralph" in system_prompt_arg

    @patch("post_processor_configs.subprocess.run")
    def test_no_glossary_context_keeps_original_prompt(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")
        process_with_ssh_claude("hello", BASE_CONFIG, glossary_ctx="")

        cmd = mock_run.call_args[0][0]
        sp_idx = cmd.index("--system-prompt")
        system_prompt_arg = cmd[sp_idx + 1]
        assert "Commonly used terms" not in system_prompt_arg


class TestProcessWithSshClaudeEmptyAndLong:
    """Tests for empty text and text exceeding max length."""

    @patch("post_processor_configs.subprocess.run")
    def test_empty_text_returns_empty(self, mock_run):
        result = process_with_ssh_claude("", BASE_CONFIG)
        assert result == ""
        mock_run.assert_not_called()

    @patch("post_processor_configs.subprocess.run")
    def test_text_exceeding_max_len_returns_original(self, mock_run):
        long_text = "a" * 201
        result = process_with_ssh_claude(long_text, BASE_CONFIG)
        assert result == long_text
        mock_run.assert_not_called()

    @patch("post_processor_configs.subprocess.run")
    def test_text_at_max_len_still_processed(self, mock_run):
        """Text exactly at max_text_len should be processed."""
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")
        text = "a" * 200
        process_with_ssh_claude(text, BASE_CONFIG)
        mock_run.assert_called_once()


class TestProcessWithSshClaudeErrors:
    """Tests for timeout and SSH errors."""

    @patch("voice_input.notify")
    @patch("post_processor_configs.subprocess.run")
    def test_timeout_returns_original_text(self, mock_run, mock_notify):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh ...", timeout=15)
        result = process_with_ssh_claude("raw text", BASE_CONFIG)
        assert result == "raw text"

    @patch("voice_input.notify")
    @patch("post_processor_configs.subprocess.run")
    def test_timeout_sends_notification(self, mock_run, mock_notify):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh ...", timeout=15)
        process_with_ssh_claude("raw text", BASE_CONFIG)
        mock_notify.assert_called_once()
        assert "timed out" in mock_notify.call_args[0][1]

    @patch("voice_input.notify")
    @patch("post_processor_configs.subprocess.run")
    def test_ssh_error_returns_original_text(self, mock_run, mock_notify):
        mock_run.return_value = MagicMock(
            returncode=255, stdout="", stderr="Connection refused"
        )
        result = process_with_ssh_claude("raw text", BASE_CONFIG)
        assert result == "raw text"

    @patch("voice_input.notify")
    @patch("post_processor_configs.subprocess.run")
    def test_ssh_error_sends_notification(self, mock_run, mock_notify):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Host not found"
        )
        process_with_ssh_claude("raw text", BASE_CONFIG)
        mock_notify.assert_called_once()
        assert "error" in mock_notify.call_args[0][1].lower()


class TestProcessWithSshClaudeHallucinationGuard:
    """Tests for hallucination guard (output > 5x input)."""

    @patch("post_processor_configs.subprocess.run")
    def test_hallucination_guard_returns_original(self, mock_run):
        """Output > 5x input length triggers guard."""
        input_text = "hi"
        # 5x of "hi" (len 2) = 10 chars, so 11+ triggers guard
        mock_run.return_value = MagicMock(
            returncode=0, stdout="a" * 11, stderr=""
        )
        result = process_with_ssh_claude(input_text, BASE_CONFIG)
        assert result == input_text

    @patch("post_processor_configs.subprocess.run")
    def test_output_within_5x_accepted(self, mock_run):
        """Output at exactly 5x input length is accepted."""
        input_text = "hi"
        mock_run.return_value = MagicMock(
            returncode=0, stdout="a" * 10, stderr=""
        )
        result = process_with_ssh_claude(input_text, BASE_CONFIG)
        assert result == "a" * 10

    @patch("post_processor_configs.subprocess.run")
    def test_normal_expansion_accepted(self, mock_run):
        """Chinese 2 chars -> English 7 chars (3.5x) is normal."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="session", stderr=""
        )
        result = process_with_ssh_claude("赛什", BASE_CONFIG)
        assert result == "session"
