"""Tests for SenseVoice use_itn=True (US-001) — ITN enables built-in punctuation."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_configs import ModelInference


class TestSenseVoiceUseItn:
    """Verify that only sensevoice passes use_itn=True to model.generate()."""

    def test_sensevoice_passes_use_itn_true(self):
        """transcribe_funasr with model_id='sensevoice' passes use_itn=True."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "hello world"}]

        result = ModelInference.transcribe_funasr(
            mock_model, "/tmp/test.wav", model_id="sensevoice"
        )

        mock_model.generate.assert_called_once_with(
            input="/tmp/test.wav", language="zh", use_itn=True
        )
        assert result == "hello world"

    def test_sensevoice_preserves_language_zh(self):
        """language='zh' is still passed alongside use_itn=True."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "test"}]

        ModelInference.transcribe_funasr(
            mock_model, "/tmp/test.wav", model_id="sensevoice"
        )

        kwargs = mock_model.generate.call_args
        assert kwargs.kwargs.get("language") == "zh"
        assert kwargs.kwargs.get("use_itn") is True

    def test_fun_asr_nano_does_not_pass_use_itn(self):
        """transcribe_funasr with model_id='fun-asr-nano' does NOT pass use_itn."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "nano result"}]

        ModelInference.transcribe_funasr(
            mock_model, "/tmp/test.wav", model_id="fun-asr-nano"
        )

        kwargs = mock_model.generate.call_args
        # fun-asr-nano uses itn=True (lowercase), NOT use_itn
        assert "use_itn" not in kwargs.kwargs

    def test_paraformer_does_not_pass_use_itn(self):
        """transcribe_funasr with model_id='paraformer' does NOT pass use_itn."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "paraformer result"}]

        ModelInference.transcribe_funasr(
            mock_model, "/tmp/test.wav", model_id="paraformer"
        )

        kwargs = mock_model.generate.call_args
        assert "use_itn" not in kwargs.kwargs

    def test_sensevoice_strips_special_tokens_with_itn(self):
        """SenseVoice special token stripping still works with use_itn enabled."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [
            {"text": "<|zh|><|HAPPY|><|Speech|><|woitn|>punctuated text here"}
        ]

        result = ModelInference.transcribe_funasr(
            mock_model, "/tmp/test.wav", model_id="sensevoice"
        )

        assert result == "punctuated text here"

    def test_sensevoice_empty_result(self):
        """SenseVoice returns empty string when model returns empty."""
        mock_model = MagicMock()
        mock_model.generate.return_value = []

        result = ModelInference.transcribe_funasr(
            mock_model, "/tmp/test.wav", model_id="sensevoice"
        )

        assert result == ""


class TestSenseVoiceItnIntegration:
    """Integration test: mock model.generate() and verify use_itn=True in kwargs."""

    @patch("model_configs._trim_leading_clipping", side_effect=lambda x, **kw: x)
    def test_full_transcribe_path_sensevoice_passes_use_itn(self, mock_trim):
        """Through ModelInference.transcribe(), sensevoice still passes use_itn=True."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "integrated test"}]

        result = ModelInference.transcribe(
            model=mock_model,
            audio_path="/tmp/test.wav",
            model_id="sensevoice",
            framework="funasr",
            extra_data=None,
        )

        mock_model.generate.assert_called_once_with(
            input="/tmp/test.wav", language="zh", use_itn=True
        )
        assert result == "integrated test"

    @patch("model_configs._trim_leading_clipping", side_effect=lambda x, **kw: x)
    def test_full_transcribe_path_paraformer_no_use_itn(self, mock_trim):
        """Through ModelInference.transcribe(), paraformer does NOT pass use_itn."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "paraformer integrated"}]

        result = ModelInference.transcribe(
            model=mock_model,
            audio_path="/tmp/test.wav",
            model_id="paraformer",
            framework="funasr",
            extra_data=None,
        )

        kwargs = mock_model.generate.call_args
        assert "use_itn" not in kwargs.kwargs
        assert result == "paraformer integrated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
