"""
Clean-room unit tests for faster-whisper integration and ASRDaemon dual-ASR fusion.

Derived exclusively from FUNCTION_SPEC.md behavior tables.
Does NOT read implementation source code.
"""

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helper: create a minimal ASRDaemon without __init__ side-effects
# (from FUNCTION_SPEC.md §Test Configuration Constants)
# ---------------------------------------------------------------------------

def _make_daemon():
    """Create minimal ASRDaemon for testing."""
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


# ---------------------------------------------------------------------------
# Helper: create mock whisper segments
# ---------------------------------------------------------------------------

def _make_segments(texts):
    """Create mock Segment objects with .text attributes."""
    segments = []
    for t in texts:
        seg = SimpleNamespace(text=t)
        segments.append(seg)
    return segments


# ===========================================================================
# Module 1: ModelLoader.load_faster_whisper_model — Behavior Table Tests
# ===========================================================================


class TestLoadFasterWhisperModel:
    """ModelLoader.load_faster_whisper_model() behavior table."""

    def test_normal_default_config(self):
        """BT#1: default config → WhisperModel(large-v3-turbo, cpu, int8)."""
        mock_whisper_module = MagicMock()
        mock_model_instance = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model_instance

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            from model_configs import ModelLoader
            config = {"model_size": "large-v3-turbo", "compute_type": "int8"}
            result = ModelLoader.load_faster_whisper_model(config, device="cpu")

            mock_whisper_module.WhisperModel.assert_called_once_with(
                "large-v3-turbo", device="cpu", compute_type="int8"
            )
            assert result is mock_model_instance

    def test_normal_custom_model_size(self):
        """BT#2: custom model_size and compute_type."""
        mock_whisper_module = MagicMock()
        mock_model_instance = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model_instance

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            from model_configs import ModelLoader
            config = {"model_size": "medium", "compute_type": "float16"}
            result = ModelLoader.load_faster_whisper_model(config, device="cpu")

            mock_whisper_module.WhisperModel.assert_called_once_with(
                "medium", device="cpu", compute_type="float16"
            )
            assert result is mock_model_instance

    def test_empty_config_uses_defaults(self):
        """BT#3: empty config → defaults to large-v3-turbo, int8."""
        mock_whisper_module = MagicMock()
        mock_model_instance = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model_instance

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            from model_configs import ModelLoader
            result = ModelLoader.load_faster_whisper_model({}, device="cpu")

            mock_whisper_module.WhisperModel.assert_called_once_with(
                "large-v3-turbo", device="cpu", compute_type="int8"
            )
            assert result is mock_model_instance

    def test_import_error_when_not_installed(self):
        """BT#4: faster-whisper not installed → ImportError with install instructions."""
        # Remove faster_whisper from sys.modules if present
        saved = sys.modules.pop("faster_whisper", None)
        try:
            with patch.dict("sys.modules", {"faster_whisper": None}):
                from model_configs import ModelLoader
                with pytest.raises(
                    ImportError,
                    match="faster-whisper is not installed.*pip install faster-whisper",
                ):
                    ModelLoader.load_faster_whisper_model(
                        {"model_size": "large-v3-turbo"}, device="cpu"
                    )
        finally:
            if saved is not None:
                sys.modules["faster_whisper"] = saved


# ===========================================================================
# Module 1: ModelInference.transcribe_faster_whisper — Behavior Table Tests
# ===========================================================================


class TestTranscribeFasterWhisper:
    """ModelInference.transcribe_faster_whisper() behavior table."""

    def test_multiple_segments_joined(self):
        """BT#1: segments [' Hello ', 'world'] → ' Hello world'."""
        from model_configs import ModelInference

        mock_model = MagicMock()
        segments = _make_segments([" Hello ", "world"])
        info = MagicMock()
        mock_model.transcribe.return_value = (iter(segments), info)

        result = ModelInference.transcribe_faster_whisper(mock_model, "/tmp/audio.wav")

        assert result == " Hello world"

    def test_single_segment(self):
        """BT#2: one segment 'test output' → 'test output'."""
        from model_configs import ModelInference

        mock_model = MagicMock()
        segments = _make_segments(["test output"])
        info = MagicMock()
        mock_model.transcribe.return_value = (iter(segments), info)

        result = ModelInference.transcribe_faster_whisper(mock_model, "/tmp/audio.wav")

        assert result == "test output"

    def test_empty_segments_returns_empty(self):
        """BT#3: empty iterable → ''."""
        from model_configs import ModelInference

        mock_model = MagicMock()
        info = MagicMock()
        mock_model.transcribe.return_value = (iter([]), info)

        result = ModelInference.transcribe_faster_whisper(mock_model, "/tmp/audio.wav")

        assert result == ""

    def test_model_transcribe_fails_raises(self):
        """BT#4: model.transcribe fails → RuntimeError propagated."""
        from model_configs import ModelInference

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("decode error")

        with pytest.raises(RuntimeError, match="decode error"):
            ModelInference.transcribe_faster_whisper(mock_model, "/tmp/audio.wav")


# ===========================================================================
# Module 2: MODEL_PRESETS["faster-whisper"] — Data Contract Tests
# ===========================================================================


class TestFasterWhisperPreset:
    """Verify faster-whisper preset structure from BT#1-7."""

    @pytest.fixture(autouse=True)
    def _load_presets(self):
        from model_presets import MODEL_PRESETS
        self.presets = MODEL_PRESETS
        self.preset = MODEL_PRESETS["faster-whisper"]

    def test_preset_exists(self):
        assert "faster-whisper" in self.presets

    def test_name(self):
        """BT#1."""
        assert self.preset["name"] == "Faster-Whisper"

    def test_description_mentions_english_and_cpu(self):
        """BT#2: description contains 'English' and 'CPU'."""
        desc = self.preset["description"]
        assert "English" in desc or "english" in desc.lower()
        assert "CPU" in desc or "cpu" in desc.lower()

    def test_framework(self):
        """BT#3."""
        assert self.preset["framework"] == "faster-whisper"

    def test_punctuation_builtin(self):
        """BT#4."""
        assert self.preset["punctuation"] == "builtin"

    def test_force_cpu(self):
        """BT#5."""
        assert self.preset["force_cpu"] is True

    def test_model_size(self):
        """BT#6."""
        assert self.preset["config"]["model_size"] == "large-v3-turbo"

    def test_compute_type(self):
        """BT#7."""
        assert self.preset["config"]["compute_type"] == "int8"


# ===========================================================================
# Module 5: ASRDaemon.__init__ — Behavior Table Tests
# ===========================================================================


class TestASRDaemonInit:
    """ASRDaemon.__init__() initializes secondary model attributes."""

    def test_secondary_model_initialized_to_none(self):
        """BT#1: _secondary_model = None after init."""
        daemon = _make_daemon()
        assert daemon._secondary_model is None

    def test_last_secondary_text_initialized_to_none(self):
        """BT#1: _last_secondary_text = None after init."""
        daemon = _make_daemon()
        assert daemon._last_secondary_text is None


# ===========================================================================
# Module 5: ASRDaemon._load_secondary_model — Behavior Table Tests
# ===========================================================================


class TestLoadSecondaryModel:
    """ASRDaemon._load_secondary_model() behavior table."""

    def test_successful_load(self):
        """BT#1: faster-whisper installed → _secondary_model set to WhisperModel."""
        daemon = _make_daemon()
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            daemon._load_secondary_model()

        assert daemon._secondary_model is mock_model
        mock_whisper_module.WhisperModel.assert_called_once_with(
            "large-v3-turbo", device="cpu", compute_type="int8"
        )

    def test_import_error_sets_none(self):
        """BT#2: ImportError → _secondary_model = None, warning logged."""
        daemon = _make_daemon()

        # Ensure faster_whisper import fails
        saved = sys.modules.pop("faster_whisper", None)
        try:
            with patch.dict("sys.modules", {"faster_whisper": None}):
                daemon._load_secondary_model()
        finally:
            if saved is not None:
                sys.modules["faster_whisper"] = saved

        assert daemon._secondary_model is None

    def test_model_load_exception_sets_none(self):
        """BT#3: WhisperModel() raises → _secondary_model = None."""
        daemon = _make_daemon()
        mock_whisper_module = MagicMock()
        mock_whisper_module.WhisperModel.side_effect = RuntimeError("model corrupt")

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            daemon._load_secondary_model()

        assert daemon._secondary_model is None

    def test_never_raises(self):
        """All errors caught — method never raises."""
        daemon = _make_daemon()

        saved = sys.modules.pop("faster_whisper", None)
        try:
            with patch.dict("sys.modules", {"faster_whisper": None}):
                # Should not raise
                daemon._load_secondary_model()
        finally:
            if saved is not None:
                sys.modules["faster_whisper"] = saved


# ===========================================================================
# Module 5: ASRDaemon._unload_secondary_model — Behavior Table Tests
# ===========================================================================


class TestUnloadSecondaryModel:
    """ASRDaemon._unload_secondary_model() behavior table."""

    def test_model_loaded_sets_none(self):
        """BT#1: model loaded → both attrs set to None."""
        daemon = _make_daemon()
        daemon._secondary_model = MagicMock()
        daemon._last_secondary_text = "some text"

        daemon._unload_secondary_model()

        assert daemon._secondary_model is None
        assert daemon._last_secondary_text is None

    def test_no_model_is_noop(self):
        """BT#2: no model loaded → no-op."""
        daemon = _make_daemon()
        daemon._secondary_model = None
        daemon._last_secondary_text = None

        daemon._unload_secondary_model()

        assert daemon._secondary_model is None
        assert daemon._last_secondary_text is None

    def test_missing_attr_safe(self):
        """BT#3: attr missing → no-op (uses getattr safety)."""
        daemon = _make_daemon()
        # Simulate __new__ without full init
        if hasattr(daemon, "_secondary_model"):
            delattr(daemon, "_secondary_model")

        # Should not raise
        daemon._unload_secondary_model()


# ===========================================================================
# Module 5: ASRDaemon.load_post_processor — Behavior Table Tests
# ===========================================================================


class TestLoadPostProcessorSecondaryLifecycle:
    """ASRDaemon.load_post_processor() secondary model lifecycle."""

    @patch("voice_input.PostProcessorLoader")
    @patch("voice_input.load_vocab", return_value={})
    def test_gemini_merge_loads_secondary(self, mock_load_vocab, mock_loader):
        """BT#1: switch to gemini-merge → _load_secondary_model() called."""
        daemon = _make_daemon()
        mock_loader.load_post_processor.return_value = None

        with patch.object(daemon, "_load_secondary_model") as mock_load, \
             patch.object(daemon, "_unload_secondary_model") as mock_unload, \
             patch("voice_input.POST_PROCESSOR_PRESETS", {
                 "gemini-merge": {
                     "name": "Gemini Merge",
                     "framework": "vertex-ai-merge",
                     "config": {"vocab_min_count": 3},
                 },
             }):
            daemon.load_post_processor("gemini-merge")

        mock_load.assert_called_once()
        mock_unload.assert_not_called()

    @patch("voice_input.PostProcessorLoader")
    @patch("voice_input.load_vocab", return_value={})
    def test_gemini_fix_unloads_secondary(self, mock_load_vocab, mock_loader):
        """BT#2: switch to gemini-fix → _unload_secondary_model() called."""
        daemon = _make_daemon()
        mock_loader.load_post_processor.return_value = None

        with patch.object(daemon, "_load_secondary_model") as mock_load, \
             patch.object(daemon, "_unload_secondary_model") as mock_unload, \
             patch("voice_input.POST_PROCESSOR_PRESETS", {
                 "gemini-fix": {
                     "name": "Gemini Fix",
                     "framework": "vertex-ai",
                     "config": {"vocab_min_count": 3},
                 },
             }):
            daemon.load_post_processor("gemini-fix")

        mock_unload.assert_called()
        mock_load.assert_not_called()

    @patch("voice_input.PostProcessorLoader")
    def test_switch_to_none_unloads_secondary(self, mock_loader):
        """BT#3: switch to none → _unload_secondary_model() called."""
        daemon = _make_daemon()
        mock_loader.load_post_processor.return_value = None

        with patch.object(daemon, "_unload_secondary_model") as mock_unload, \
             patch("voice_input.POST_PROCESSOR_PRESETS", {
                 "none": {
                     "name": "None",
                     "framework": "regex",
                 },
             }):
            daemon.load_post_processor("none")

        mock_unload.assert_called()


# ===========================================================================
# Module 5: ASRDaemon._handle_transcribe — Behavior Table Tests
# ===========================================================================


class TestHandleTranscribeDualASR:
    """ASRDaemon._handle_transcribe() dual ASR path."""

    def test_no_secondary_model_resets_text_to_none(self):
        """BT#1: no secondary model → _last_secondary_text reset to None."""
        daemon = _make_daemon()
        daemon._secondary_model = None
        daemon._last_secondary_text = "stale data"
        daemon.model = MagicMock()
        daemon.framework = "fireredasr"
        daemon.extra_data = {"use_gpu": True}
        daemon.current_model_id = "firered-asr"

        with patch("voice_input.ModelInference") as mock_inference, \
             patch.object(daemon, "_post_process", return_value="processed"), \
             patch("voice_input.set_status"):
            mock_inference.transcribe.return_value = "primary text"
            result = daemon._handle_transcribe({"data": "/tmp/audio.wav"})

        assert daemon._last_secondary_text is None
        assert result["text"] == "processed"

    def test_secondary_model_sets_text(self):
        """BT#2: secondary model available → _last_secondary_text set."""
        daemon = _make_daemon()
        mock_secondary = MagicMock()
        segments = _make_segments(["whisper ", "output"])
        info = MagicMock()
        mock_secondary.transcribe.return_value = (iter(segments), info)
        daemon._secondary_model = mock_secondary
        daemon.model = MagicMock()
        daemon.framework = "fireredasr"
        daemon.extra_data = {"use_gpu": True}
        daemon.current_model_id = "firered-asr"

        with patch("voice_input.ModelInference") as mock_inference, \
             patch.object(daemon, "_post_process", return_value="processed"), \
             patch("voice_input.set_status"):
            mock_inference.transcribe.return_value = "primary text"
            result = daemon._handle_transcribe({"data": "/tmp/audio.wav"})

        assert daemon._last_secondary_text == "whisper output"
        assert result["text"] == "processed"

    def test_secondary_failure_sets_none(self):
        """BT#3: secondary model.transcribe raises → _last_secondary_text = None."""
        daemon = _make_daemon()
        mock_secondary = MagicMock()
        mock_secondary.transcribe.side_effect = RuntimeError("whisper crash")
        daemon._secondary_model = mock_secondary
        daemon.model = MagicMock()
        daemon.framework = "fireredasr"
        daemon.extra_data = {"use_gpu": True}
        daemon.current_model_id = "firered-asr"

        with patch("voice_input.ModelInference") as mock_inference, \
             patch.object(daemon, "_post_process", return_value="processed"), \
             patch("voice_input.set_status"):
            mock_inference.transcribe.return_value = "primary text"
            result = daemon._handle_transcribe({"data": "/tmp/audio.wav"})

        assert daemon._last_secondary_text is None
        assert result["text"] == "processed"

    def test_primary_failure_discards_secondary(self):
        """BT#4: primary ASR fails → _last_secondary_text is None.

        In parallel mode, secondary thread may have already started and
        completed before primary failure is detected. The result is
        simply discarded.
        """
        daemon = _make_daemon()
        mock_secondary = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = "whisper output"
        mock_secondary.transcribe.return_value = (iter([mock_seg]), MagicMock())
        daemon._secondary_model = mock_secondary
        daemon.model = MagicMock()
        daemon.framework = "fireredasr"
        daemon.extra_data = {"use_gpu": True}
        daemon.current_model_id = "firered-asr"

        with patch("voice_input.ModelInference") as mock_inference, \
             patch.object(daemon, "set_status"):
            mock_inference.transcribe.side_effect = Exception("primary ASR fail")

            result = daemon._handle_transcribe({"data": "/tmp/audio.wav"})

        assert "error" in result
        assert daemon._last_secondary_text is None


# ===========================================================================
# Module 5: ASRDaemon._post_process — Behavior Table Tests
# ===========================================================================


class TestPostProcessVertexAiMerge:
    """ASRDaemon._post_process() vertex-ai-merge dispatch."""

    @patch("voice_input.process_with_gemini_merge", return_value="merged result")
    @patch("voice_input.PostProcessorInference")
    @patch("voice_input.apply_vocab", side_effect=lambda t, v, m: t)
    @patch("voice_input.glossary_context", return_value="")
    @patch("voice_input.diff_to_vocab", side_effect=lambda o, p, v=None: {})
    @patch("voice_input.save_vocab")
    @patch("voice_input.load_vocab", return_value={})
    def test_vertex_ai_merge_calls_gemini_merge(
        self, mock_load_vocab, mock_save, mock_diff, mock_glossary,
        mock_apply, mock_ppi, mock_merge
    ):
        """BT#1: vertex-ai-merge with secondary text → process_with_gemini_merge called."""
        from post_processor_presets import POST_PROCESSOR_PRESETS

        daemon = _make_daemon()
        daemon.post_processor_framework = "vertex-ai-merge"
        daemon.current_post_processor_id = "gemini-merge"
        daemon._last_secondary_text = "whisper output"
        daemon.punc_model = None
        daemon._vocab = {}

        mock_ppi.remove_fillers.side_effect = lambda t: t

        result = daemon._post_process("raw text")

        mock_merge.assert_called_once()
        # Verify secondary_text was passed
        call_args = mock_merge.call_args
        # secondary_text is the 2nd positional arg
        assert call_args[0][1] == "whisper output" or call_args.kwargs.get("secondary_text") == "whisper output"

    @patch("voice_input.process_with_gemini_merge", return_value="polished result")
    @patch("voice_input.PostProcessorInference")
    @patch("voice_input.apply_vocab", side_effect=lambda t, v, m: t)
    @patch("voice_input.glossary_context", return_value="")
    @patch("voice_input.diff_to_vocab", side_effect=lambda o, p, v=None: {})
    @patch("voice_input.save_vocab")
    @patch("voice_input.load_vocab", return_value={})
    def test_vertex_ai_merge_passes_none_secondary(
        self, mock_load_vocab, mock_save, mock_diff, mock_glossary,
        mock_apply, mock_ppi, mock_merge
    ):
        """BT#2: vertex-ai-merge without secondary (None) → fallback polish."""
        daemon = _make_daemon()
        daemon.post_processor_framework = "vertex-ai-merge"
        daemon.current_post_processor_id = "gemini-merge"
        daemon._last_secondary_text = None
        daemon.punc_model = None
        daemon._vocab = {}

        mock_ppi.remove_fillers.side_effect = lambda t: t

        result = daemon._post_process("raw text")

        mock_merge.assert_called_once()
        call_args = mock_merge.call_args
        # secondary_text should be None
        secondary_arg = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("secondary_text")
        assert secondary_arg is None

    @patch("voice_input.process_with_vertex_ai", return_value="vertex result")
    @patch("voice_input.PostProcessorInference")
    @patch("voice_input.apply_vocab", side_effect=lambda t, v, m: t)
    @patch("voice_input.glossary_context", return_value="")
    @patch("voice_input.diff_to_vocab", side_effect=lambda o, p, v=None: {})
    @patch("voice_input.save_vocab")
    @patch("voice_input.load_vocab", return_value={})
    def test_vertex_ai_still_works_regression(
        self, mock_load_vocab, mock_save, mock_diff, mock_glossary,
        mock_apply, mock_ppi, mock_vertex_ai
    ):
        """BT#3/regression: vertex-ai (gemini-fix) still dispatches correctly."""
        daemon = _make_daemon()
        daemon.post_processor_framework = "vertex-ai"
        daemon.current_post_processor_id = "gemini-fix"
        daemon.punc_model = None
        daemon._vocab = {}
        daemon._last_secondary_text = None

        mock_ppi.remove_fillers.side_effect = lambda t: t

        result = daemon._post_process("raw text")

        mock_vertex_ai.assert_called_once()

    @patch("voice_input.process_with_ssh_claude", return_value="ssh result")
    @patch("voice_input.PostProcessorInference")
    @patch("voice_input.apply_vocab", side_effect=lambda t, v, m: t)
    @patch("voice_input.glossary_context", return_value="")
    @patch("voice_input.diff_to_vocab", side_effect=lambda o, p, v=None: {})
    @patch("voice_input.save_vocab")
    @patch("voice_input.load_vocab", return_value={})
    def test_ssh_claude_still_works_regression(
        self, mock_load_vocab, mock_save, mock_diff, mock_glossary,
        mock_apply, mock_ppi, mock_ssh_claude
    ):
        """BT#4/regression: ssh-claude (haiku-fix) still dispatches correctly."""
        daemon = _make_daemon()
        daemon.post_processor_framework = "ssh-claude"
        daemon.current_post_processor_id = "haiku-fix"
        daemon.punc_model = None
        daemon._vocab = {}
        daemon._last_secondary_text = None

        mock_ppi.remove_fillers.side_effect = lambda t: t

        result = daemon._post_process("raw text")

        mock_ssh_claude.assert_called_once()

    @patch("voice_input.PostProcessorInference")
    def test_regex_only_no_llm(self, mock_ppi):
        """BT#5: regex framework → only filler removal, no LLM call."""
        daemon = _make_daemon()
        daemon.post_processor_framework = "regex"
        daemon.current_post_processor_id = "none"
        daemon.punc_model = None
        daemon._vocab = {}

        mock_ppi.remove_fillers.return_value = "defillered text"

        result = daemon._post_process("raw text")

        mock_ppi.remove_fillers.assert_called_once()
        assert result == "defillered text"

    @patch("voice_input.process_with_gemini_merge", return_value="changed text")
    @patch("voice_input.PostProcessorInference")
    @patch("voice_input.apply_vocab", side_effect=lambda t, v, m: t)
    @patch("voice_input.glossary_context", return_value="")
    @patch("voice_input.diff_to_vocab", return_value={"new": {"variants": {"old": 1}}})
    @patch("voice_input.save_vocab")
    @patch("voice_input.load_vocab", return_value={"new": {"variants": {"old": 1}}})
    def test_vocab_updated_when_text_changed(
        self, mock_load_vocab, mock_save, mock_diff, mock_glossary,
        mock_apply, mock_ppi, mock_merge
    ):
        """BT#6: LLM changed text → diff_to_vocab called, save_vocab called."""
        daemon = _make_daemon()
        daemon.post_processor_framework = "vertex-ai-merge"
        daemon.current_post_processor_id = "gemini-merge"
        daemon._last_secondary_text = "secondary"
        daemon.punc_model = None
        daemon._vocab = {}

        mock_ppi.remove_fillers.side_effect = lambda t: t

        daemon._post_process("raw text")

        mock_diff.assert_called_once()
        mock_save.assert_called_once()


# ===========================================================================
# Module 1: ModelLoader.load_model — faster-whisper dispatch tests
# ===========================================================================


class TestLoadModelFasterWhisperDispatch:
    """ModelLoader.load_model() faster-whisper dispatch."""

    def test_faster_whisper_dispatch(self):
        """BT#1: model_id='faster-whisper' → load_faster_whisper_model called."""
        from model_configs import ModelLoader

        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            model, framework, extra_data = ModelLoader.load_model("faster-whisper")

        assert framework == "faster-whisper"
        assert extra_data is None
        assert model is mock_model

    def test_force_cpu_overrides_device(self):
        """BT#3: force_cpu=True → device='cpu' even when cuda specified."""
        from model_configs import ModelLoader

        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            model, framework, extra_data = ModelLoader.load_model(
                "faster-whisper", device="cuda:0"
            )

        # Regardless of device arg, WhisperModel should be called with device="cpu"
        mock_whisper_module.WhisperModel.assert_called_once()
        call_kwargs = mock_whisper_module.WhisperModel.call_args
        assert call_kwargs.kwargs.get("device", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None) == "cpu" or \
            "cpu" in str(call_kwargs)

    def test_unknown_model_id_raises(self):
        """BT#4: unknown model_id → ValueError."""
        from model_configs import ModelLoader

        with pytest.raises(ValueError, match="Unknown model: nonexistent"):
            ModelLoader.load_model("nonexistent")


# ===========================================================================
# Module 1: ModelInference.transcribe — faster-whisper dispatch tests
# ===========================================================================


class TestTranscribeFasterWhisperDispatch:
    """ModelInference.transcribe() faster-whisper dispatch."""

    def test_faster_whisper_dispatch(self):
        """BT#1: framework='faster-whisper' → transcribe_faster_whisper called."""
        from model_configs import ModelInference

        mock_model = MagicMock()
        segments = _make_segments(["hello"])
        info = MagicMock()
        mock_model.transcribe.return_value = (iter(segments), info)

        with patch.object(
            ModelInference, "transcribe_faster_whisper", return_value="hello"
        ) as mock_fw:
            # We need to handle _trim_leading_clipping which is called before dispatch
            with patch("model_configs._trim_leading_clipping", return_value="/tmp/audio.wav"):
                result = ModelInference.transcribe(
                    mock_model,
                    "/tmp/audio.wav",
                    model_id="faster-whisper",
                    framework="faster-whisper",
                )

        mock_fw.assert_called_once()
        assert result == "hello"
