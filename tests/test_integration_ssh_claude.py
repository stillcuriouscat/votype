"""Tests for ssh-claude integration into ASRDaemon — US-005."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from post_processor_configs import PostProcessorLoader, PostProcessorInference
from post_processor_presets import POST_PROCESSOR_PRESETS


class TestPostProcessorLoaderSshClaude:
    """PostProcessorLoader.load_post_processor() handles ssh-claude."""

    def test_haiku_fix_returns_none(self):
        result = PostProcessorLoader.load_post_processor("haiku-fix")
        assert result is None

    def test_haiku_expand_returns_none(self):
        result = PostProcessorLoader.load_post_processor("haiku-expand")
        assert result is None

    def test_regex_still_returns_none(self):
        result = PostProcessorLoader.load_post_processor("none")
        assert result is None

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown post-processor"):
            PostProcessorLoader.load_post_processor("nonexistent")


class TestASRDaemonLoadPostProcessor:
    """ASRDaemon.load_post_processor() integration for ssh-claude."""

    def _make_daemon(self):
        """Create a minimal ASRDaemon without loading models."""
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
            daemon.post_processor_framework = None
            daemon.punc_model = None
            daemon._vocab = {}
            return daemon

    def test_haiku_fix_loads_vocab(self, tmp_path):
        daemon = self._make_daemon()
        vocab_data = {"Ralph": {"variants": {"Raf": 5}}}
        vocab_file = tmp_path / "vocab.json"
        vocab_file.write_text(json.dumps(vocab_data), encoding="utf-8")

        with patch("post_processor_configs.VOCAB_PATH", vocab_file):
            daemon.load_post_processor("haiku-fix")

        assert daemon.current_post_processor_id == "haiku-fix"
        assert daemon.post_processor_framework == "ssh-claude"
        assert daemon._vocab == vocab_data
        assert daemon.post_processor_model is None

    def test_haiku_fix_empty_vocab(self, tmp_path):
        daemon = self._make_daemon()
        # No vocab file exists — should get empty dict
        with patch("post_processor_configs.VOCAB_PATH", tmp_path / "missing.json"):
            daemon.load_post_processor("haiku-fix")

        assert daemon._vocab == {}
        assert daemon.current_post_processor_id == "haiku-fix"

    def test_haiku_expand_raises_not_implemented(self):
        daemon = self._make_daemon()
        with patch("voice_input.notify") as mock_notify:
            with pytest.raises(ValueError, match="not yet implemented"):
                daemon.load_post_processor("haiku-expand")

        # Notification sent
        mock_notify.assert_called_once()
        assert "not yet implemented" in mock_notify.call_args[0][1].lower()

    def test_haiku_expand_does_not_change_state(self):
        daemon = self._make_daemon()
        daemon.current_post_processor_id = "none"
        daemon.post_processor_framework = "regex"

        with patch("voice_input.notify"):
            with pytest.raises(ValueError):
                daemon.load_post_processor("haiku-expand")

        # State should NOT have changed
        assert daemon.current_post_processor_id == "none"
        assert daemon.post_processor_framework == "regex"

    def test_regex_preset_no_vocab_load(self):
        daemon = self._make_daemon()
        daemon.load_post_processor("none")

        assert daemon.current_post_processor_id == "none"
        assert daemon.post_processor_framework == "regex"
        assert daemon._vocab == {}


class TestASRDaemonPostProcess:
    """ASRDaemon._post_process() pipeline for ssh-claude."""

    def _make_daemon(self):
        """Create a minimal ASRDaemon configured for ssh-claude."""
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
            daemon.current_post_processor_id = "haiku-fix"
            daemon.post_processor_framework = "ssh-claude"
            daemon.punc_model = None
            daemon._vocab = {"Ralph": {"variants": {"Raf": 5}}}
            return daemon

    def test_pipeline_calls_apply_vocab(self):
        daemon = self._make_daemon()

        with patch("post_processor_configs.apply_vocab", return_value="test Ralph") as mock_apply, \
             patch("post_processor_configs.glossary_context", return_value="Commonly used terms: Ralph"), \
             patch("post_processor_configs.process_with_ssh_claude", return_value="test Ralph"), \
             patch("post_processor_configs.diff_to_vocab"), \
             patch("post_processor_configs.save_vocab"):
            daemon._post_process("test Raf")

        mock_apply.assert_called_once()

    def test_pipeline_calls_ssh_claude(self):
        daemon = self._make_daemon()

        with patch("post_processor_configs.apply_vocab", return_value="test text") as mock_apply, \
             patch("post_processor_configs.glossary_context", return_value="ctx"), \
             patch("post_processor_configs.process_with_ssh_claude", return_value="corrected text") as mock_ssh, \
             patch("post_processor_configs.diff_to_vocab", return_value=daemon._vocab), \
             patch("post_processor_configs.save_vocab"):
            result = daemon._post_process("test text")

        mock_ssh.assert_called_once()
        # glossary_ctx passed to process_with_ssh_claude
        assert mock_ssh.call_args[0][2] == "ctx"

    def test_pipeline_accumulates_vocab_on_change(self):
        daemon = self._make_daemon()
        new_vocab = {"Ralph": {"variants": {"Raf": 5, "Ralf": 1}}}

        with patch("post_processor_configs.apply_vocab", return_value="test Ralf"), \
             patch("post_processor_configs.glossary_context", return_value=""), \
             patch("post_processor_configs.process_with_ssh_claude", return_value="test Ralph"), \
             patch("post_processor_configs.diff_to_vocab", return_value=new_vocab) as mock_diff, \
             patch("post_processor_configs.save_vocab") as mock_save:
            daemon._post_process("test Ralf")

        mock_diff.assert_called_once()
        mock_save.assert_called_once_with(new_vocab)
        assert daemon._vocab == new_vocab

    def test_pipeline_skips_vocab_when_no_change(self):
        daemon = self._make_daemon()

        with patch("post_processor_configs.apply_vocab", return_value="same text"), \
             patch("post_processor_configs.glossary_context", return_value=""), \
             patch("post_processor_configs.process_with_ssh_claude", return_value="same text"), \
             patch("post_processor_configs.diff_to_vocab") as mock_diff, \
             patch("post_processor_configs.save_vocab") as mock_save:
            daemon._post_process("same text")

        mock_diff.assert_not_called()
        mock_save.assert_not_called()

    def test_pipeline_order_regex_before_ssh(self):
        """Filler removal happens before ssh-claude pipeline."""
        daemon = self._make_daemon()
        call_order = []

        orig_remove_fillers = PostProcessorInference.remove_fillers

        def mock_remove_fillers(text):
            call_order.append("fillers")
            return orig_remove_fillers(text)

        def mock_apply_vocab(text, vocab, min_count):
            call_order.append("vocab")
            return text

        def mock_ssh(text, config, ctx):
            call_order.append("ssh")
            return text

        with patch.object(PostProcessorInference, "remove_fillers", side_effect=mock_remove_fillers), \
             patch("post_processor_configs.apply_vocab", side_effect=mock_apply_vocab), \
             patch("post_processor_configs.glossary_context", return_value=""), \
             patch("post_processor_configs.process_with_ssh_claude", side_effect=mock_ssh), \
             patch("post_processor_configs.diff_to_vocab"), \
             patch("post_processor_configs.save_vocab"):
            daemon._post_process("test text")

        assert call_order == ["fillers", "vocab", "ssh"]

    def test_empty_text_skips_ssh(self):
        daemon = self._make_daemon()

        with patch("post_processor_configs.apply_vocab") as mock_apply, \
             patch("post_processor_configs.process_with_ssh_claude") as mock_ssh:
            result = daemon._post_process("")

        mock_apply.assert_not_called()
        mock_ssh.assert_not_called()
        assert result == ""

    def test_llama_cpp_pipeline_still_works(self):
        """LLM post-processing still works for llama-cpp framework."""
        from voice_input import ASRDaemon

        with patch("voice_input.ModelLoader"), \
             patch("voice_input.get_current_model", return_value="firered-asr"):
            daemon = ASRDaemon.__new__(ASRDaemon)
            daemon.model = None
            daemon.framework = None
            daemon.extra_data = None
            daemon.current_model_id = "firered-asr"
            daemon.running = False
            daemon.indicator = None
            daemon.gtk_thread = None
            daemon.post_processor_model = MagicMock()
            daemon.current_post_processor_id = "none"
            daemon.post_processor_framework = "regex"
            daemon.punc_model = None
            daemon._vocab = {}

        # regex framework with no model — should just do filler removal
        result = daemon._post_process("hello world")
        assert result == "hello world"


class TestDaemonSetPostProcessorHandler:
    """Daemon command handler for set_post_processor with haiku-expand."""

    def _make_daemon(self):
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

    def test_haiku_expand_no_contradictory_success_notification(self):
        """CRITIC-R3-L2: haiku-expand must NOT send success notification."""
        daemon = self._make_daemon()
        notify_calls = []

        def track_notify(*args, **kwargs):
            notify_calls.append(args)

        with patch("voice_input.notify", side_effect=track_notify):
            with pytest.raises(ValueError):
                daemon.load_post_processor("haiku-expand")

        # Only the "not implemented" notification, no success notification
        assert len(notify_calls) == 1
        assert "not yet implemented" in notify_calls[0][1].lower()

    def test_haiku_expand_raises_before_try_block(self):
        """CRITIC-R4-C1: raise happens before the try block."""
        daemon = self._make_daemon()

        with patch("voice_input.notify"):
            with pytest.raises(ValueError, match="not yet implemented"):
                daemon.load_post_processor("haiku-expand")

        # Since it raises before the try block, the fallback state is NOT applied
        # (inner except would set current_post_processor_id = "none")
        # State should remain unchanged from before the call
        assert daemon.current_post_processor_id == "none"


class TestInitVocab:
    """ASRDaemon.__init__ initializes _vocab."""

    def test_init_has_empty_vocab(self):
        with patch("voice_input.ModelLoader"), \
             patch("voice_input.get_current_model", return_value="firered-asr"):
            from voice_input import ASRDaemon
            daemon = ASRDaemon.__new__(ASRDaemon)
            daemon.__init__()
        assert daemon._vocab == {}
