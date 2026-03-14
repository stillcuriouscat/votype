"""Tests for dual ASR fusion (faster-whisper + FireRedASR + Gemini merge) — US-007."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_configs import ModelLoader, ModelInference
from post_processor_configs import (
    process_with_gemini_merge,
    PostProcessorLoader,
)
from post_processor_presets import POST_PROCESSOR_PRESETS


# --- Config for gemini-merge tests ---
MERGE_CONFIG = {
    "ssh_host": "oracle-cloud",
    "proxy_script": "~/vertex_proxy.py",
    "model": "gemini-2.5-flash",
    "vertex_region": "us-central1",
    "timeout": 15,
    "min_text_len": 15,
    "vocab_min_count": 3,
    "system_prompt": "You are a merge editor.",
}


# ============================================================
# ModelLoader: faster-whisper
# ============================================================
class TestModelLoaderFasterWhisper:
    """Unit tests for ModelLoader.load_faster_whisper_model()."""

    @patch("model_configs.WhisperModel", create=True)
    def test_load_creates_whisper_model(self, MockWhisper):
        """WhisperModel created with correct params."""
        # Patch the import inside the method
        mock_model = MagicMock()
        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=lambda *a, **kw: mock_model)}):
            result = ModelLoader.load_faster_whisper_model(
                {"model_size": "large-v3-turbo", "compute_type": "int8"}, device="cpu"
            )
        assert result is mock_model

    def test_import_error_gives_install_instructions(self):
        """Missing faster-whisper raises ImportError with install hint."""
        with patch.dict("sys.modules", {"faster_whisper": None}):
            with pytest.raises(ImportError, match="pip install faster-whisper"):
                ModelLoader.load_faster_whisper_model(
                    {"model_size": "large-v3-turbo"}, device="cpu"
                )

    def test_load_model_dispatches_to_faster_whisper(self):
        """load_model() dispatches framework='faster-whisper' correctly."""
        mock_model = MagicMock()
        with patch.object(ModelLoader, "load_faster_whisper_model", return_value=mock_model) as mock_load:
            model, framework, extra = ModelLoader.load_model("faster-whisper")

        assert model is mock_model
        assert framework == "faster-whisper"
        assert extra is None
        mock_load.assert_called_once()


# ============================================================
# ModelInference: faster-whisper
# ============================================================
class TestModelInferenceFasterWhisper:
    """Unit tests for ModelInference.transcribe_faster_whisper()."""

    def test_joins_segment_text(self):
        """Segments are joined into a single string."""
        seg1 = MagicMock(text="Hello ")
        seg2 = MagicMock(text="world")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg1, seg2]), MagicMock())

        result = ModelInference.transcribe_faster_whisper(mock_model, "/tmp/test.wav")
        assert result == "Hello world"

    def test_empty_segments_returns_empty(self):
        """No segments returns empty string."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), MagicMock())

        result = ModelInference.transcribe_faster_whisper(mock_model, "/tmp/test.wav")
        assert result == ""

    def test_transcribe_dispatches_faster_whisper(self):
        """Unified transcribe() dispatches framework='faster-whisper'."""
        seg = MagicMock(text="test")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), MagicMock())

        # Need to patch _trim_leading_clipping to avoid soundfile dependency
        with patch("model_configs._trim_leading_clipping", return_value="/tmp/test.wav"):
            result = ModelInference.transcribe(
                model=mock_model,
                audio_path="/tmp/test.wav",
                model_id="faster-whisper",
                framework="faster-whisper",
            )
        assert result == "test"


# ============================================================
# process_with_gemini_merge: guards
# ============================================================
class TestGeminiMergeGuards:
    """Tests for process_with_gemini_merge() input guards."""

    @patch("post_processor_configs.subprocess.run")
    def test_empty_text_returns_empty(self, mock_run):
        result = process_with_gemini_merge("", None, MERGE_CONFIG)
        assert result == ""
        mock_run.assert_not_called()

    @patch("post_processor_configs.subprocess.run")
    def test_short_text_returns_original(self, mock_run):
        short = "hi"
        result = process_with_gemini_merge(short, "hi", MERGE_CONFIG)
        assert result == short
        mock_run.assert_not_called()

    @patch("post_processor_configs.subprocess.run")
    def test_hallucination_guard(self, mock_run):
        """Output > 2x primary length triggers guard."""
        text = "a" * 50
        mock_run.return_value = MagicMock(returncode=0, stdout="x" * 101, stderr="")
        result = process_with_gemini_merge(text, "secondary", MERGE_CONFIG)
        assert result == text

    @patch("post_processor_configs.subprocess.run")
    def test_question_guard(self, mock_run):
        """Input with ？ but output without triggers guard."""
        text = "a" * 44 + "这是什么？"  # >45 chars
        mock_run.return_value = MagicMock(returncode=0, stdout="a" * 50, stderr="")
        result = process_with_gemini_merge(text, None, MERGE_CONFIG)
        assert result == text


# ============================================================
# process_with_gemini_merge: user input format
# ============================================================
class TestGeminiMergeUserInput:
    """Tests for user input format construction."""

    @patch("post_processor_configs.subprocess.run")
    def test_dual_input_format(self, mock_run):
        """Both texts present: Chinese ASR + English ASR format."""
        primary = "a" * 50
        secondary = "b" * 50
        mock_run.return_value = MagicMock(returncode=0, stdout="merged", stderr="")

        process_with_gemini_merge(primary, secondary, MERGE_CONFIG)

        stdin_data = json.loads(mock_run.call_args.kwargs["input"])
        assert f"Chinese ASR: {primary}" in stdin_data["user_input"]
        assert f"English ASR: {secondary}" in stdin_data["user_input"]

    @patch("post_processor_configs.subprocess.run")
    def test_single_input_format_when_secondary_none(self, mock_run):
        """Secondary is None: only Chinese ASR in user input."""
        primary = "a" * 50
        mock_run.return_value = MagicMock(returncode=0, stdout="polished", stderr="")

        process_with_gemini_merge(primary, None, MERGE_CONFIG)

        stdin_data = json.loads(mock_run.call_args.kwargs["input"])
        assert f"Chinese ASR: {primary}" in stdin_data["user_input"]
        assert "English ASR" not in stdin_data["user_input"]

    @patch("post_processor_configs.subprocess.run")
    def test_glossary_appended_to_system_prompt(self, mock_run):
        primary = "a" * 50
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")

        process_with_gemini_merge(primary, None, MERGE_CONFIG, glossary_ctx="Terms: Claude")

        stdin_data = json.loads(mock_run.call_args.kwargs["input"])
        assert "Terms: Claude" in stdin_data["system_prompt"]


# ============================================================
# process_with_gemini_merge: errors
# ============================================================
class TestGeminiMergeErrors:
    """Tests for timeout and SSH error handling."""

    @patch("voice_input.notify")
    @patch("post_processor_configs.subprocess.run")
    def test_timeout_returns_primary(self, mock_run, mock_notify):
        text = "a" * 50
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh ...", timeout=15)
        result = process_with_gemini_merge(text, "secondary", MERGE_CONFIG)
        assert result == text

    @patch("voice_input.notify")
    @patch("post_processor_configs.subprocess.run")
    def test_ssh_error_returns_primary(self, mock_run, mock_notify):
        text = "a" * 50
        mock_run.return_value = MagicMock(returncode=255, stdout="", stderr="Connection refused")
        result = process_with_gemini_merge(text, "secondary", MERGE_CONFIG)
        assert result == text


# ============================================================
# Preset structure
# ============================================================
class TestGeminiMergePreset:
    """Verify gemini-merge preset structure."""

    def test_preset_exists(self):
        assert "gemini-merge" in POST_PROCESSOR_PRESETS

    def test_framework_is_vertex_ai_merge(self):
        assert POST_PROCESSOR_PRESETS["gemini-merge"]["framework"] == "vertex-ai-merge"

    def test_config_fields(self):
        config = POST_PROCESSOR_PRESETS["gemini-merge"]["config"]
        assert config["ssh_host"] == "oracle-cloud"
        assert config["proxy_script"] == "~/vertex_proxy.py"
        assert config["model"] == "gemini-2.5-flash"
        assert config["min_text_len"] == 15
        assert config["timeout"] == 15

    def test_prompt_file(self):
        config = POST_PROCESSOR_PRESETS["gemini-merge"]["config"]
        assert "system_prompt_file" in config
        assert "gemini-merge" in config["system_prompt_file"]

    def test_loader_returns_none(self):
        result = PostProcessorLoader.load_post_processor("gemini-merge")
        assert result is None


# ============================================================
# ASRDaemon integration: secondary model load/unload
# ============================================================
class TestSecondaryModelLifecycle:
    """Tests for secondary model load/unload on post-processor switch."""

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
            daemon.post_processor_framework = "regex"
            daemon.punc_model = None
            daemon._vocab = {}
            daemon._secondary_model = None
            daemon._last_secondary_text = None
            return daemon

    def test_gemini_merge_loads_secondary(self, tmp_path):
        """Switching to gemini-merge loads secondary model."""
        daemon = self._make_daemon()
        mock_model = MagicMock()

        with patch("post_processor_configs.VOCAB_PATH", tmp_path / "vocab.json"), \
             patch("voice_input.ASRDaemon._load_secondary_model") as mock_load:
            daemon.load_post_processor("gemini-merge")

        mock_load.assert_called_once()
        assert daemon.post_processor_framework == "vertex-ai-merge"

    def test_switching_away_unloads_secondary(self, tmp_path):
        """Switching from gemini-merge to none unloads secondary model."""
        daemon = self._make_daemon()
        daemon._secondary_model = MagicMock()  # Simulate loaded model
        daemon.current_post_processor_id = "gemini-merge"
        daemon.post_processor_framework = "vertex-ai-merge"

        with patch("post_processor_configs.VOCAB_PATH", tmp_path / "vocab.json"):
            daemon.load_post_processor("none")

        assert daemon._secondary_model is None

    def test_gemini_fix_does_not_load_secondary(self, tmp_path):
        """gemini-fix (vertex-ai) should NOT load secondary model."""
        daemon = self._make_daemon()

        with patch("post_processor_configs.VOCAB_PATH", tmp_path / "vocab.json"), \
             patch("voice_input.ASRDaemon._load_secondary_model") as mock_load:
            daemon.load_post_processor("gemini-fix")

        mock_load.assert_not_called()
        assert daemon.post_processor_framework == "vertex-ai"


# ============================================================
# ASRDaemon: pipeline dispatch for vertex-ai-merge
# ============================================================
class TestPipelineDispatch:
    """Tests for _post_process() pipeline dispatch."""

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
            daemon.current_post_processor_id = "gemini-merge"
            daemon.post_processor_framework = "vertex-ai-merge"
            daemon.punc_model = None
            daemon._vocab = {}
            daemon._secondary_model = MagicMock()
            daemon._last_secondary_text = "whisper output"
            return daemon

    def test_vertex_ai_merge_calls_gemini_merge(self):
        """vertex-ai-merge dispatches to process_with_gemini_merge."""
        daemon = self._make_daemon()

        with patch("post_processor_configs.apply_vocab", return_value="test text"), \
             patch("post_processor_configs.glossary_context", return_value=""), \
             patch("post_processor_configs.process_with_gemini_merge", return_value="merged") as mock_merge, \
             patch("post_processor_configs.process_with_vertex_ai") as mock_vertex, \
             patch("post_processor_configs.process_with_ssh_claude") as mock_ssh, \
             patch("post_processor_configs.diff_to_vocab", return_value={}), \
             patch("post_processor_configs.save_vocab"), \
             patch("post_processor_configs.load_vocab", return_value={}):
            result = daemon._post_process("test text")

        mock_merge.assert_called_once()
        mock_vertex.assert_not_called()
        mock_ssh.assert_not_called()
        # Verify secondary text passed to merge
        call_args = mock_merge.call_args
        assert call_args[0][1] == "whisper output"  # secondary_text arg

    def test_fallback_when_secondary_none(self):
        """When _last_secondary_text is None, merge still called with None."""
        daemon = self._make_daemon()
        daemon._last_secondary_text = None

        with patch("post_processor_configs.apply_vocab", return_value="test text"), \
             patch("post_processor_configs.glossary_context", return_value=""), \
             patch("post_processor_configs.process_with_gemini_merge", return_value="polished") as mock_merge, \
             patch("post_processor_configs.diff_to_vocab", return_value={}), \
             patch("post_processor_configs.save_vocab"), \
             patch("post_processor_configs.load_vocab", return_value={}):
            daemon._post_process("test text")

        # secondary_text should be None
        call_args = mock_merge.call_args
        assert call_args[0][1] is None


# ============================================================
# Regression: existing presets unchanged
# ============================================================
class TestExistingPresetsUnchanged:
    """Verify existing presets are not modified."""

    def test_gemini_fix_unchanged(self):
        preset = POST_PROCESSOR_PRESETS["gemini-fix"]
        assert preset["framework"] == "vertex-ai"
        assert preset["config"]["model"] == "gemini-2.5-flash"

    def test_haiku_fix_unchanged(self):
        preset = POST_PROCESSOR_PRESETS["haiku-fix"]
        assert preset["framework"] == "ssh-claude"
        assert preset["config"]["ssh_host"] == "oracle-cloud"

    def test_none_unchanged(self):
        preset = POST_PROCESSOR_PRESETS["none"]
        assert preset["framework"] == "regex"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
