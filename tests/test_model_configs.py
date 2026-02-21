#!/usr/bin/env python3
"""
Test model configuration and loading logic
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_configs import MODEL_PRESETS, ModelLoader, ModelInference


class TestModelPresets:
    """Test model preset configurations"""

    def test_all_models_have_required_fields(self):
        """Test all models have required fields"""
        required_fields = ["name", "description", "framework", "config"]

        for model_id, preset in MODEL_PRESETS.items():
            for field in required_fields:
                assert field in preset, f"Model {model_id} missing field: {field}"

    def test_framework_types(self):
        """Test framework types are valid"""
        valid_frameworks = ["funasr", "transformers", "fireredasr"]

        for model_id, preset in MODEL_PRESETS.items():
            framework = preset["framework"]
            assert framework in valid_frameworks, \
                f"Model {model_id} has invalid framework: {framework}"

    def test_new_models_exist(self):
        """Test new models are added to configuration"""
        assert "qwen2-audio" in MODEL_PRESETS, "Qwen2-Audio model not found"
        assert "firered-asr" in MODEL_PRESETS, "FireRedASR model not found"

    def test_qwen2_audio_config(self):
        """Test Qwen2-Audio configuration is correct"""
        preset = MODEL_PRESETS["qwen2-audio"]

        assert preset["framework"] == "transformers"
        assert "model_id" in preset["config"]
        assert "Qwen/Qwen2-Audio-7B" in preset["config"]["model_id"]
        # Uses chat template format, check generate_kwargs exists
        assert "generate_kwargs" in preset

    def test_firered_asr_config(self):
        """Test FireRedASR configuration is correct"""
        preset = MODEL_PRESETS["firered-asr"]

        assert preset["framework"] == "fireredasr"
        assert "model_type" in preset["config"]
        assert preset["config"]["model_type"] == "aed"
        assert "model_id" in preset["config"]
        assert "FireRedTeam" in preset["config"]["model_id"]

    def test_original_models_still_exist(self):
        """Test original models still exist"""
        original_models = ["fun-asr-nano", "paraformer", "sensevoice"]

        for model_id in original_models:
            assert model_id in MODEL_PRESETS, f"Original model {model_id} not found"
            assert MODEL_PRESETS[model_id]["framework"] == "funasr"


class TestModelLoader:
    """Test model loader"""

    def test_load_model_invalid_id(self):
        """Test loading invalid model ID raises exception"""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelLoader.load_model("invalid_model_id")

    def test_load_model_returns_tuple(self):
        """Test loading model returns correct tuple format"""
        # Note: This test requires model files to exist and may fail
        # Skip if models are not downloaded
        pytest.skip("Requires model files to be downloaded")

        model_id = "fun-asr-nano"
        model, framework, extra_data = ModelLoader.load_model(model_id, "cpu")

        assert model is not None
        assert framework == "funasr"


class TestModelInference:
    """Test model inference engine"""

    def test_transcribe_invalid_framework(self):
        """Test inference with invalid framework raises exception"""
        with pytest.raises(ValueError, match="Unknown framework"):
            ModelInference.transcribe(
                model=None,
                audio_path="dummy.wav",
                model_id="dummy",
                framework="invalid_framework",
                extra_data=None
            )


class TestModelIntegration:
    """Integration tests - require model files"""

    @pytest.mark.skipif(
        not Path.home().joinpath(".cache/modelscope").exists(),
        reason="Model files not downloaded"
    )
    def test_load_and_transcribe_funasr(self):
        """Test loading and inference with FunASR model"""
        # Requires actual audio file and model
        pytest.skip("Requires actual audio file and model")

    @pytest.mark.skipif(
        not Path.home().joinpath(".cache/huggingface").exists(),
        reason="Model files not downloaded"
    )
    def test_load_and_transcribe_qwen2_audio(self):
        """Test loading and inference with Qwen2-Audio model"""
        # Requires actual audio file and model
        pytest.skip("Requires actual audio file and model")

    @pytest.mark.skipif(
        not Path.home().joinpath(".cache/huggingface").exists(),
        reason="Model files not downloaded"
    )
    def test_load_and_transcribe_firered_asr(self):
        """Test loading and inference with FireRedASR model"""
        # Requires actual audio file and model
        pytest.skip("Requires actual audio file and model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
