"""Tests for process_with_vertex_ai() and gemini-fix integration — US-010."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from post_processor_configs import process_with_vertex_ai, PostProcessorLoader
from post_processor_presets import POST_PROCESSOR_PRESETS


# Minimal config matching gemini-fix preset structure
BASE_CONFIG = {
    "ssh_host": "oracle-cloud",
    "proxy_script": "~/vertex_proxy.py",
    "model": "gemini-2.5-flash",
    "vertex_region": "us-central1",
    "timeout": 15,
    "min_text_len": 45,
    "max_text_len": 200,
    "vocab_min_count": 3,
    "system_prompt": "You are an ASR correction tool.",
}


class TestProcessWithVertexAiSuccess:
    """Tests for successful Vertex AI invocations."""

    @patch("post_processor_configs.subprocess.run")
    def test_success_returns_stripped_stdout(self, mock_run):
        text = "a" * 50  # Over min_text_len
        mock_run.return_value = MagicMock(
            returncode=0, stdout="  polished text  \n", stderr=""
        )
        result = process_with_vertex_ai(text, BASE_CONFIG)
        assert result == "polished text"

    @patch("post_processor_configs.subprocess.run")
    def test_command_construction(self, mock_run):
        """Verify the SSH command includes correct parts."""
        text = "a" * 50
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")
        process_with_vertex_ai(text, BASE_CONFIG)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]

        # Check SSH with ConnectTimeout
        assert cmd[0] == "ssh"
        assert "-o" in cmd
        assert "ConnectTimeout=5" in cmd

        # Check host
        assert "oracle-cloud" in cmd

        # Check proxy script
        assert "python3" in cmd
        assert "~/vertex_proxy.py" in cmd

    @patch("post_processor_configs.subprocess.run")
    def test_json_stdin_format(self, mock_run):
        """Verify JSON stdin has correct fields."""
        text = "a" * 50
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")
        process_with_vertex_ai(text, BASE_CONFIG, glossary_ctx="Terms: Claude")

        call_args = mock_run.call_args
        stdin_data = json.loads(call_args.kwargs["input"])

        assert "system_prompt" in stdin_data
        assert "user_input" in stdin_data
        assert stdin_data["model"] == "gemini-2.5-flash"
        assert stdin_data["region"] == "us-central1"

    @patch("post_processor_configs.subprocess.run")
    def test_glossary_context_appended_to_system_prompt(self, mock_run):
        text = "a" * 50
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")
        process_with_vertex_ai(text, BASE_CONFIG, glossary_ctx="Commonly used terms: Ralph")

        call_args = mock_run.call_args
        stdin_data = json.loads(call_args.kwargs["input"])
        assert "Commonly used terms: Ralph" in stdin_data["system_prompt"]

    @patch("post_processor_configs.subprocess.run")
    def test_no_glossary_context_keeps_original_prompt(self, mock_run):
        text = "a" * 50
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")
        process_with_vertex_ai(text, BASE_CONFIG, glossary_ctx="")

        call_args = mock_run.call_args
        stdin_data = json.loads(call_args.kwargs["input"])
        assert "Commonly used terms" not in stdin_data["system_prompt"]


class TestProcessWithVertexAiGuards:
    """Tests for empty text, min/max length, and hallucination guards."""

    @patch("post_processor_configs.subprocess.run")
    def test_empty_text_returns_empty(self, mock_run):
        result = process_with_vertex_ai("", BASE_CONFIG)
        assert result == ""
        mock_run.assert_not_called()

    @patch("post_processor_configs.subprocess.run")
    def test_text_below_min_len_returns_original(self, mock_run):
        short_text = "hi"  # Below min_text_len=45
        result = process_with_vertex_ai(short_text, BASE_CONFIG)
        assert result == short_text
        mock_run.assert_not_called()

    @patch("post_processor_configs.subprocess.run")
    def test_text_exceeding_max_len_returns_original(self, mock_run):
        long_text = "a" * 201
        result = process_with_vertex_ai(long_text, BASE_CONFIG)
        assert result == long_text
        mock_run.assert_not_called()

    @patch("post_processor_configs.subprocess.run")
    def test_text_at_max_len_still_processed(self, mock_run):
        """Text exactly at max_text_len should be processed."""
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")
        text = "a" * 200
        process_with_vertex_ai(text, BASE_CONFIG)
        mock_run.assert_called_once()


class TestProcessWithVertexAiErrors:
    """Tests for timeout and SSH errors."""

    @patch("voice_input.notify")
    @patch("post_processor_configs.subprocess.run")
    def test_timeout_returns_original_text(self, mock_run, mock_notify):
        text = "a" * 50
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh ...", timeout=15)
        result = process_with_vertex_ai(text, BASE_CONFIG)
        assert result == text

    @patch("voice_input.notify")
    @patch("post_processor_configs.subprocess.run")
    def test_timeout_sends_notification(self, mock_run, mock_notify):
        text = "a" * 50
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh ...", timeout=15)
        process_with_vertex_ai(text, BASE_CONFIG)
        mock_notify.assert_called_once()
        assert "timed out" in mock_notify.call_args[0][1]

    @patch("voice_input.notify")
    @patch("post_processor_configs.subprocess.run")
    def test_ssh_error_returns_original_text(self, mock_run, mock_notify):
        text = "a" * 50
        mock_run.return_value = MagicMock(
            returncode=255, stdout="", stderr="Connection refused"
        )
        result = process_with_vertex_ai(text, BASE_CONFIG)
        assert result == text

    @patch("voice_input.notify")
    @patch("post_processor_configs.subprocess.run")
    def test_ssh_error_sends_notification(self, mock_run, mock_notify):
        text = "a" * 50
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Gemini API error"
        )
        process_with_vertex_ai(text, BASE_CONFIG)
        mock_notify.assert_called_once()
        assert "error" in mock_notify.call_args[0][1].lower()


class TestProcessWithVertexAiHallucinationGuard:
    """Tests for hallucination guard (output > 5x input)."""

    @patch("post_processor_configs.subprocess.run")
    def test_hallucination_guard_returns_original(self, mock_run):
        """Output > 5x input length triggers guard."""
        text = "a" * 50  # len 50, so 251+ triggers guard
        mock_run.return_value = MagicMock(
            returncode=0, stdout="a" * 251, stderr=""
        )
        result = process_with_vertex_ai(text, BASE_CONFIG)
        assert result == text

    @patch("post_processor_configs.subprocess.run")
    def test_output_within_5x_accepted(self, mock_run):
        """Output at exactly 5x input length is accepted."""
        text = "a" * 50
        mock_run.return_value = MagicMock(
            returncode=0, stdout="b" * 250, stderr=""
        )
        result = process_with_vertex_ai(text, BASE_CONFIG)
        assert result == "b" * 250


class TestGeminiFixPreset:
    """Verify gemini-fix preset structure."""

    def test_gemini_fix_preset_exists(self):
        assert "gemini-fix" in POST_PROCESSOR_PRESETS

    def test_gemini_fix_framework(self):
        preset = POST_PROCESSOR_PRESETS["gemini-fix"]
        assert preset["framework"] == "vertex-ai"

    def test_gemini_fix_config_fields(self):
        config = POST_PROCESSOR_PRESETS["gemini-fix"]["config"]
        assert config["ssh_host"] == "oracle-cloud"
        assert config["proxy_script"] == "~/vertex_proxy.py"
        assert config["model"] == "gemini-2.5-flash"
        assert config["vertex_region"] == "us-central1"
        assert config["timeout"] == 15
        assert config["min_text_len"] == 45
        assert config["max_text_len"] == 200
        assert config["vocab_min_count"] == 3

    def test_gemini_fix_prompt_file(self):
        config = POST_PROCESSOR_PRESETS["gemini-fix"]["config"]
        assert "system_prompt_file" in config
        assert "gemini-fix" in config["system_prompt_file"]

    def test_gemini_fix_loader_returns_none(self):
        result = PostProcessorLoader.load_post_processor("gemini-fix")
        assert result is None


class TestVertexAiIntegration:
    """Integration tests for vertex-ai in ASRDaemon."""

    def _make_daemon(self):
        """Create a minimal ASRDaemon configured for vertex-ai."""
        with patch("voice_input.ModelLoader"), \
             patch("voice_input.get_current_model", return_value="firered-asr"):
            from voice_input import ASRDaemon
            daemon = ASRDaemon.__new__(ASRDaemon)
            daemon.model = None
            daemon.framework = None
            daemon.extra_data = None
            daemon.current_model_id = "firered-asr"
            daemon.running = False
            daemon.indicator = None
            daemon.gtk_thread = None
            daemon.post_processor_model = None
            daemon.current_post_processor_id = "none"
            daemon.post_processor_framework = "regex"
            daemon.punc_model = None
            daemon._vocab = {}
            return daemon

    def test_gemini_fix_loads_vocab(self, tmp_path):
        daemon = self._make_daemon()
        vocab_data = {"Claude": {"variants": {"克劳的": 5}}}
        vocab_file = tmp_path / "vocab.json"
        vocab_file.write_text(json.dumps(vocab_data), encoding="utf-8")

        with patch("post_processor_configs.VOCAB_PATH", vocab_file):
            daemon.load_post_processor("gemini-fix")

        assert daemon.current_post_processor_id == "gemini-fix"
        assert daemon.post_processor_framework == "vertex-ai"
        assert daemon._vocab == vocab_data
        assert daemon.post_processor_model is None

    def test_pipeline_dispatch_uses_vertex_ai(self):
        daemon = self._make_daemon()
        daemon.current_post_processor_id = "gemini-fix"
        daemon.post_processor_framework = "vertex-ai"
        daemon._vocab = {"Claude": {"variants": {"克劳的": 5}}}

        with patch("post_processor_configs.apply_vocab", return_value="test text") as mock_apply, \
             patch("post_processor_configs.glossary_context", return_value="ctx"), \
             patch("post_processor_configs.process_with_vertex_ai", return_value="corrected text") as mock_vertex, \
             patch("post_processor_configs.process_with_ssh_claude") as mock_ssh, \
             patch("post_processor_configs.diff_to_vocab", return_value=daemon._vocab), \
             patch("post_processor_configs.save_vocab"):
            daemon._post_process("test text")

        # Vertex AI should be called, NOT SSH Claude
        mock_vertex.assert_called_once()
        mock_ssh.assert_not_called()

    def test_ssh_claude_still_works(self):
        """Regression: haiku-fix still uses process_with_ssh_claude."""
        daemon = self._make_daemon()
        daemon.current_post_processor_id = "haiku-fix"
        daemon.post_processor_framework = "ssh-claude"
        daemon._vocab = {}

        with patch("post_processor_configs.apply_vocab", return_value="test text"), \
             patch("post_processor_configs.glossary_context", return_value=""), \
             patch("post_processor_configs.process_with_ssh_claude", return_value="test text") as mock_ssh, \
             patch("post_processor_configs.process_with_vertex_ai") as mock_vertex, \
             patch("post_processor_configs.diff_to_vocab"), \
             patch("post_processor_configs.save_vocab"):
            daemon._post_process("test text")

        # SSH Claude should be called, NOT Vertex AI
        mock_ssh.assert_called_once()
        mock_vertex.assert_not_called()
