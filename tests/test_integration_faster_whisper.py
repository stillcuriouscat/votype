"""Clean-room integration tests for faster-whisper and dual ASR pipeline contracts.

Derived from LOW_LEVEL_DESIGN.md sections 1.1, 1.2, 1.5, 2.1, 4.1.
Tests ModelLoader, ModelInference, ASRDaemon secondary model lifecycle,
dual transcription, and pipeline dispatch.
Does NOT read implementation source code.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from model_configs import ModelLoader, ModelInference
from model_presets import MODEL_PRESETS
import voice_input


# ===========================================================================
# 1. MODEL_PRESETS Contract (LLD 1.2)
# ===========================================================================


class TestFasterWhisperPreset:
    """Verify faster-whisper preset structure in MODEL_PRESETS."""

    def test_preset_exists(self):
        assert "faster-whisper" in MODEL_PRESETS

    def test_preset_framework(self):
        assert MODEL_PRESETS["faster-whisper"]["framework"] == "faster-whisper"

    def test_preset_punctuation_builtin(self):
        """LLD 1.2: punctuation='builtin'."""
        assert MODEL_PRESETS["faster-whisper"]["punctuation"] == "builtin"

    def test_preset_force_cpu(self):
        """LLD 1.2: force_cpu=True (must not compete with FireRedASR for VRAM)."""
        assert MODEL_PRESETS["faster-whisper"].get("force_cpu") is True

    def test_preset_config_model_size(self):
        config = MODEL_PRESETS["faster-whisper"]["config"]
        assert config["model_size"] == "large-v3-turbo"

    def test_preset_config_compute_type(self):
        config = MODEL_PRESETS["faster-whisper"]["config"]
        assert config["compute_type"] == "int8"

    def test_preset_has_name_and_description(self):
        preset = MODEL_PRESETS["faster-whisper"]
        assert isinstance(preset.get("name"), str)
        assert isinstance(preset.get("description"), str)

    def test_existing_presets_unchanged(self):
        """Regression: existing presets still present."""
        for preset_id in ("firered-asr", "fun-asr-nano", "paraformer", "sensevoice"):
            assert preset_id in MODEL_PRESETS, f"Missing preset: {preset_id}"


# ===========================================================================
# 2. ModelLoader Contracts (LLD 1.1)
# ===========================================================================


class TestFasterWhisperLoader:
    """ModelLoader.load_faster_whisper_model() contracts."""

    def _mock_faster_whisper(self):
        """Inject a fake faster_whisper module with WhisperModel."""
        mock_module = MagicMock()
        mock_instance = MagicMock()
        mock_module.WhisperModel.return_value = mock_instance
        return mock_module, mock_instance

    def test_load_creates_model_with_correct_params(self):
        """LLD 1.1: WhisperModel(model_size, device='cpu', compute_type='int8')."""
        mock_fw, mock_instance = self._mock_faster_whisper()

        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            config = {"model_size": "large-v3-turbo", "compute_type": "int8"}
            ModelLoader.load_faster_whisper_model(config, device="cpu")

        mock_fw.WhisperModel.assert_called_once_with(
            "large-v3-turbo", device="cpu", compute_type="int8"
        )

    def test_load_returns_whisper_model_instance(self):
        """LLD 1.1: returns WhisperModel instance."""
        mock_fw, mock_instance = self._mock_faster_whisper()

        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            config = {"model_size": "large-v3-turbo", "compute_type": "int8"}
            result = ModelLoader.load_faster_whisper_model(config, device="cpu")

        assert result is mock_instance

    def test_load_raises_import_error_when_not_installed(self):
        """LLD 1.1: ImportError with install instructions when package missing."""
        with patch.dict(sys.modules, {"faster_whisper": None}):
            config = {"model_size": "large-v3-turbo", "compute_type": "int8"}
            with pytest.raises(ImportError):
                ModelLoader.load_faster_whisper_model(config)

    def test_load_model_dispatches_faster_whisper(self):
        """LLD 1.1: load_model() dispatches framework='faster-whisper'."""
        mock_fw, mock_instance = self._mock_faster_whisper()

        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            model, framework, extra = ModelLoader.load_model("faster-whisper", device="cpu")

        assert framework == "faster-whisper"
        assert extra is None


# ===========================================================================
# 3. ModelInference Contracts (LLD 1.1)
# ===========================================================================


class TestFasterWhisperInference:
    """ModelInference.transcribe_faster_whisper() contracts."""

    def test_joins_segments_text(self):
        """LLD 1.1: concatenated text from all segments (no separator)."""
        seg1 = MagicMock()
        seg1.text = "Hello "
        seg2 = MagicMock()
        seg2.text = "world"
        info = MagicMock()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], info)

        result = ModelInference.transcribe_faster_whisper(mock_model, "/tmp/test.wav")
        assert result == "Hello world"

    def test_empty_segments_returns_empty_string(self):
        """LLD 1.1: empty segments -> empty string."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())

        result = ModelInference.transcribe_faster_whisper(mock_model, "/tmp/test.wav")
        assert result == ""

    @patch("model_configs._trim_leading_clipping", side_effect=lambda p: p)
    def test_transcribe_dispatches_faster_whisper(self, mock_trim):
        """LLD 1.1: transcribe() delegates to transcribe_faster_whisper."""
        with patch.object(
            ModelInference, "transcribe_faster_whisper", return_value="test"
        ) as mock_fw:
            result = ModelInference.transcribe(
                model=MagicMock(),
                audio_path="/tmp/test.wav",
                model_id="faster-whisper",
                framework="faster-whisper",
                extra_data=None,
            )
        assert result == "test"
        mock_fw.assert_called_once()

    def test_language_not_forced(self):
        """LLD 1.1: Language auto-detected (not forced) for mixed Chinese-English."""
        seg1 = MagicMock()
        seg1.text = "test"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1], MagicMock())

        ModelInference.transcribe_faster_whisper(mock_model, "/tmp/test.wav")

        call_kwargs = mock_model.transcribe.call_args
        if call_kwargs.kwargs:
            assert "language" not in call_kwargs.kwargs or call_kwargs.kwargs.get("language") is None


# ===========================================================================
# 4. ASRDaemon Secondary Model Lifecycle (LLD 1.5, 4.1)
# ===========================================================================


def _make_bare_daemon():
    """Create an ASRDaemon with bypassed __init__."""
    with patch.object(voice_input.ASRDaemon, "__init__", lambda self, *a, **kw: None):
        daemon = voice_input.ASRDaemon.__new__(voice_input.ASRDaemon)
        daemon._secondary_model = None
        daemon._last_secondary_text = None
        daemon.running = False
        return daemon


class TestSecondaryModelLifecycle:
    """ASRDaemon._load_secondary_model / _unload_secondary_model contracts."""

    def test_init_secondary_model_is_none(self):
        """LLD 1.5: _secondary_model initialized to None in __init__."""
        daemon = _make_bare_daemon()
        assert daemon._secondary_model is None

    def test_init_last_secondary_text_is_none(self):
        """LLD 1.5 (CRITIC-R10-L4): _last_secondary_text initialized to None."""
        daemon = _make_bare_daemon()
        assert daemon._last_secondary_text is None

    def test_load_secondary_model_sets_model(self):
        """LLD 1.5: _load_secondary_model() sets self._secondary_model."""
        mock_fw = MagicMock()
        mock_instance = MagicMock()
        mock_fw.WhisperModel.return_value = mock_instance

        daemon = _make_bare_daemon()
        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            daemon._load_secondary_model()

        assert daemon._secondary_model is mock_instance

    def test_load_failure_sets_none(self):
        """LLD 4.1: Exception from WhisperModel() -> _secondary_model = None."""
        mock_fw = MagicMock()
        mock_fw.WhisperModel.side_effect = RuntimeError("download failed")

        daemon = _make_bare_daemon()
        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            daemon._load_secondary_model()

        assert daemon._secondary_model is None

    def test_import_error_sets_none(self):
        """LLD 4.1: ImportError -> _secondary_model = None, never raises."""
        daemon = _make_bare_daemon()
        with patch.dict(sys.modules, {"faster_whisper": None}):
            daemon._load_secondary_model()

        assert daemon._secondary_model is None

    def test_load_never_raises(self):
        """LLD 1.5: Never raises, all exceptions caught."""
        mock_fw = MagicMock()
        mock_fw.WhisperModel.side_effect = Exception("catastrophic")

        daemon = _make_bare_daemon()
        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            daemon._load_secondary_model()

        assert daemon._secondary_model is None

    def test_unload_sets_both_to_none(self):
        """LLD 1.5: _unload sets _secondary_model and _last_secondary_text to None."""
        daemon = _make_bare_daemon()
        daemon._secondary_model = MagicMock()
        daemon._last_secondary_text = "some text"

        daemon._unload_secondary_model()

        assert daemon._secondary_model is None
        assert daemon._last_secondary_text is None

    def test_unload_is_noop_when_already_none(self):
        """LLD 1.5: No-op if _secondary_model is already None."""
        daemon = _make_bare_daemon()
        daemon._unload_secondary_model()
        assert daemon._secondary_model is None


# ===========================================================================
# 5. Dual Transcription in _handle_transcribe (LLD 1.5, 2.1 rows 1-2)
# ===========================================================================


def _make_transcribe_daemon():
    """Create a daemon with enough state for _handle_transcribe tests."""
    daemon = _make_bare_daemon()
    daemon.running = True
    daemon.model = MagicMock()
    daemon.framework = "fireredasr"
    daemon.extra_data = None
    daemon.current_model_id = "firered-asr"
    daemon.indicator = None
    daemon.post_processor_model = None
    daemon.current_post_processor_id = "none"
    daemon.post_processor_framework = None
    daemon.punc_model = None
    daemon._vocab = {}
    return daemon


class TestDualTranscription:
    """_handle_transcribe() dual ASR path contracts."""

    @patch.object(voice_input.ASRDaemon, "_post_process", return_value="processed text")
    @patch.object(voice_input.ModelInference, "transcribe", return_value="primary text")
    def test_secondary_text_set_when_model_available(self, mock_primary, mock_pp):
        """LLD 2.1 row 2: secondary transcription stored as _last_secondary_text."""
        daemon = _make_transcribe_daemon()
        mock_secondary = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = "secondary text"
        mock_secondary.transcribe.return_value = (iter([mock_seg]), MagicMock())
        daemon._secondary_model = mock_secondary

        with patch.object(daemon, "set_status", create=True):
            daemon._handle_transcribe({"command": "transcribe", "data": "/tmp/audio.wav"})

        assert daemon._last_secondary_text == "secondary text"

    @patch.object(voice_input.ASRDaemon, "_post_process", return_value="processed text")
    @patch.object(voice_input.ModelInference, "transcribe", return_value="primary text")
    def test_secondary_failure_sets_none(self, mock_primary, mock_pp):
        """LLD 4.1: secondary model inference fails -> _last_secondary_text = None."""
        daemon = _make_transcribe_daemon()
        mock_secondary = MagicMock()
        mock_secondary.transcribe.side_effect = RuntimeError("fail")
        daemon._secondary_model = mock_secondary

        with patch.object(daemon, "set_status", create=True):
            result = daemon._handle_transcribe({"command": "transcribe", "data": "/tmp/audio.wav"})

        assert daemon._last_secondary_text is None
        assert "text" in result

    @patch.object(voice_input.ASRDaemon, "_post_process", return_value="processed text")
    @patch.object(voice_input.ModelInference, "transcribe", return_value="primary text")
    def test_no_secondary_model_resets_text_to_none(self, mock_primary, mock_pp):
        """LLD 1.5: when _secondary_model is None, _last_secondary_text reset to None."""
        daemon = _make_transcribe_daemon()
        daemon._secondary_model = None
        daemon._last_secondary_text = "stale from previous"

        with patch.object(daemon, "set_status", create=True):
            daemon._handle_transcribe({"command": "transcribe", "data": "/tmp/audio.wav"})

        assert daemon._last_secondary_text is None


# ===========================================================================
# 6. Pipeline Dispatch for vertex-ai-merge (LLD 1.5 _post_process)
# ===========================================================================


def _make_pipeline_daemon(framework="vertex-ai-merge", post_proc_id="gemini-merge"):
    """Create daemon configured for _post_process testing."""
    daemon = _make_bare_daemon()
    daemon._secondary_model = MagicMock()
    daemon._last_secondary_text = "whisper output"
    daemon.post_processor_framework = framework
    daemon.current_post_processor_id = post_proc_id
    daemon.punc_model = None
    daemon._vocab = {}
    daemon.post_processor_model = None
    return daemon


class TestPipelineDispatch:
    """_post_process() vertex-ai-merge dispatch contracts."""

    @patch("post_processor_configs.process_with_gemini_merge", return_value="merged result")
    @patch("post_processor_configs.save_vocab")
    @patch("post_processor_configs.diff_to_vocab", return_value={})
    @patch("post_processor_configs.apply_vocab", side_effect=lambda text, *a, **kw: text)
    @patch("post_processor_configs.glossary_context", return_value="")
    def test_vertex_ai_merge_calls_gemini_merge(
        self, mock_gc, mock_av, mock_dtv, mock_sv, mock_merge
    ):
        """LLD 1.5: vertex-ai-merge dispatches to process_with_gemini_merge."""
        with patch.object(voice_input.PostProcessorInference, "remove_fillers", side_effect=lambda t: t):
            daemon = _make_pipeline_daemon()
            result = daemon._post_process("raw text from ASR")

        mock_merge.assert_called_once()
        assert result == "merged result"

    @patch("post_processor_configs.process_with_gemini_merge", return_value="merged result")
    @patch("post_processor_configs.save_vocab")
    @patch("post_processor_configs.diff_to_vocab", return_value={})
    @patch("post_processor_configs.apply_vocab", side_effect=lambda text, *a, **kw: text)
    @patch("post_processor_configs.glossary_context", return_value="")
    def test_vertex_ai_merge_passes_secondary_text(
        self, mock_gc, mock_av, mock_dtv, mock_sv, mock_merge
    ):
        """LLD 2.1 row 7: secondary_text passed to merge function."""
        with patch.object(voice_input.PostProcessorInference, "remove_fillers", side_effect=lambda t: t):
            daemon = _make_pipeline_daemon()
            daemon._last_secondary_text = "whisper output"
            daemon._post_process("raw text")

        call_args = mock_merge.call_args
        # secondary_text should be positional arg [1] or kwarg
        all_args = list(call_args.args) if call_args.args else []
        all_kwargs = call_args.kwargs or {}
        secondary = all_kwargs.get("secondary_text", all_args[1] if len(all_args) > 1 else None)
        assert secondary == "whisper output"

    @patch("post_processor_configs.process_with_gemini_merge", return_value="single polish")
    @patch("post_processor_configs.save_vocab")
    @patch("post_processor_configs.diff_to_vocab", return_value={})
    @patch("post_processor_configs.apply_vocab", side_effect=lambda text, *a, **kw: text)
    @patch("post_processor_configs.glossary_context", return_value="")
    def test_fallback_when_secondary_text_none(
        self, mock_gc, mock_av, mock_dtv, mock_sv, mock_merge
    ):
        """LLD 1.5: when secondary_text is None, merge still called (dual-purpose prompt)."""
        with patch.object(voice_input.PostProcessorInference, "remove_fillers", side_effect=lambda t: t):
            daemon = _make_pipeline_daemon()
            daemon._last_secondary_text = None
            daemon._post_process("raw text")

        mock_merge.assert_called_once()
        call_args = mock_merge.call_args
        all_args = list(call_args.args) if call_args.args else []
        all_kwargs = call_args.kwargs or {}
        secondary = all_kwargs.get("secondary_text", all_args[1] if len(all_args) > 1 else None)
        assert secondary is None

    @patch("post_processor_configs.process_with_vertex_ai", return_value="vertex fixed")
    @patch("post_processor_configs.save_vocab")
    @patch("post_processor_configs.diff_to_vocab", return_value={})
    @patch("post_processor_configs.apply_vocab", side_effect=lambda text, *a, **kw: text)
    @patch("post_processor_configs.glossary_context", return_value="")
    def test_vertex_ai_still_works_regression(
        self, mock_gc, mock_av, mock_dtv, mock_sv, mock_fix
    ):
        """Regression: vertex-ai (gemini-fix) pipeline still works."""
        with patch.object(voice_input.PostProcessorInference, "remove_fillers", side_effect=lambda t: t):
            daemon = _make_pipeline_daemon(framework="vertex-ai", post_proc_id="gemini-fix")
            result = daemon._post_process("raw text")

        mock_fix.assert_called_once()

    @patch("post_processor_configs.process_with_ssh_claude", return_value="haiku fixed")
    @patch("post_processor_configs.save_vocab")
    @patch("post_processor_configs.diff_to_vocab", return_value={})
    @patch("post_processor_configs.apply_vocab", side_effect=lambda text, *a, **kw: text)
    @patch("post_processor_configs.glossary_context", return_value="")
    def test_ssh_claude_still_works_regression(
        self, mock_gc, mock_av, mock_dtv, mock_sv, mock_ssh
    ):
        """Regression: ssh-claude (haiku-fix) pipeline still works."""
        with patch.object(voice_input.PostProcessorInference, "remove_fillers", side_effect=lambda t: t):
            daemon = _make_pipeline_daemon(framework="ssh-claude", post_proc_id="haiku-fix")
            result = daemon._post_process("raw text")

        mock_ssh.assert_called_once()


# ===========================================================================
# 7. Pipeline Order Contract (LLD 1.5 _post_process)
# ===========================================================================


class TestPipelineOrder:
    """Verify pipeline step ordering for vertex-ai-merge."""

    @patch("post_processor_configs.process_with_gemini_merge")
    @patch("post_processor_configs.save_vocab")
    @patch("post_processor_configs.diff_to_vocab", return_value={})
    @patch("post_processor_configs.apply_vocab")
    @patch("post_processor_configs.glossary_context", return_value="")
    @patch("post_processor_configs.load_vocab", return_value={})
    def test_filler_removal_before_merge(
        self, mock_lv, mock_gc, mock_av, mock_dtv, mock_sv, mock_merge
    ):
        """LLD 1.5: Pipeline is filler_removal -> punc -> vocab -> merge -> diff -> save."""
        call_order = []

        with patch.object(
            voice_input.PostProcessorInference, "remove_fillers",
            side_effect=lambda t: (call_order.append("remove_fillers"), t)[1],
        ):
            mock_av.side_effect = lambda text, *a, **kw: (call_order.append("apply_vocab"), text)[1]
            # Return DIFFERENT text to ensure diff_to_vocab is reached
            mock_merge.side_effect = lambda *a, **kw: (call_order.append("gemini_merge"), "MERGED TEXT")[1]
            mock_dtv.side_effect = lambda *a, **kw: (call_order.append("diff_to_vocab"), {})[1]
            mock_sv.side_effect = lambda *a, **kw: call_order.append("save_vocab")

            daemon = _make_pipeline_daemon()
            daemon._post_process("raw text")

        # Core ordering: filler removal and vocab applied before merge
        assert call_order.index("remove_fillers") < call_order.index("gemini_merge")
        assert call_order.index("apply_vocab") < call_order.index("gemini_merge")
        # diff_to_vocab and save_vocab follow merge
        if "diff_to_vocab" in call_order:
            assert call_order.index("gemini_merge") < call_order.index("diff_to_vocab")
        if "save_vocab" in call_order:
            assert call_order.index("gemini_merge") < call_order.index("save_vocab")

    @patch("post_processor_configs.process_with_gemini_merge", return_value="merged")
    @patch("post_processor_configs.save_vocab")
    @patch("post_processor_configs.diff_to_vocab", return_value={})
    @patch("post_processor_configs.apply_vocab", side_effect=lambda text, *a, **kw: text)
    @patch("post_processor_configs.glossary_context", return_value="")
    def test_secondary_text_not_processed_through_filler_removal(
        self, mock_gc, mock_av, mock_dtv, mock_sv, mock_merge
    ):
        """LLD US-006: Secondary text does NOT go through filler removal or FireRedPunc."""
        with patch.object(
            voice_input.PostProcessorInference, "remove_fillers",
            side_effect=lambda t: t,
        ) as mock_rf:
            daemon = _make_pipeline_daemon()
            daemon._last_secondary_text = "whisper raw"
            daemon._post_process("raw primary")

            # remove_fillers called exactly once (primary only)
            assert mock_rf.call_count == 1
