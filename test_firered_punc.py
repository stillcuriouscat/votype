#!/usr/bin/env python3
"""Tests for FireRedPunc integration: model config, case preservation, and real model inference."""

import sys
import os
import re
import unittest

# Ensure FireRedASR2S is importable
sys.path.insert(0, os.path.expanduser("~/code/FireRedASR2S"))

from post_processor_configs import PostProcessorInference, PostProcessorLoader
from post_processor_presets import POST_PROCESSOR_PRESETS
from model_presets import MODEL_PRESETS


class TestPunctuationArchitecture(unittest.TestCase):
    """Test that punctuation is configured as ASR model property, not post-processor."""

    def test_firered_asr_has_punctuation_config(self):
        """firered-asr declares punctuation='firered-punc' in MODEL_PRESETS."""
        self.assertIn("firered-asr", MODEL_PRESETS)
        self.assertEqual(MODEL_PRESETS["firered-asr"]["punctuation"], "firered-punc")

    def test_firered_asr_has_punc_config(self):
        """firered-asr has punc_config with model_dir."""
        punc_config = MODEL_PRESETS["firered-asr"].get("punc_config")
        self.assertIsNotNone(punc_config)
        self.assertIn("model_dir", punc_config)

    def test_builtin_models_have_punctuation_builtin(self):
        """Models with built-in punctuation declare punctuation='builtin'."""
        for model_id in ("paraformer", "sensevoice", "fun-asr-nano"):
            with self.subTest(model_id=model_id):
                self.assertEqual(MODEL_PRESETS[model_id]["punctuation"], "builtin")

    def test_firered_punc_not_in_post_processor_presets(self):
        """firered-punc is NOT a post-processor preset (it's now automatic)."""
        self.assertNotIn("firered-punc", POST_PROCESSOR_PRESETS)

    def test_all_models_have_punctuation_key(self):
        """Every MODEL_PRESETS entry has a 'punctuation' key."""
        for model_id, preset in MODEL_PRESETS.items():
            with self.subTest(model_id=model_id):
                self.assertIn("punctuation", preset)


class TestCasePreservation(unittest.TestCase):
    """Test the case preservation logic in process_with_firered_punc (no model needed)."""

    def test_uppercase_words_preserved(self):
        """Uppercase English words like WER, DEMO should stay uppercase."""
        text = "也不知道WER到底是多少再录一下这个DEMO"
        # Extract the case mapping logic directly
        orig_words = {w.lower(): w for w in re.findall(r'[a-zA-Z]+', text)}
        self.assertEqual(orig_words["wer"], "WER")
        self.assertEqual(orig_words["demo"], "DEMO")

        # Simulate: lowercase -> model output (with punctuation) -> restore
        lowered = text.lower()
        # Pretend model added punctuation
        fake_punc = "也不知道wer到底是多少，再录一下这个demo。"
        restored = re.sub(
            r'[a-zA-Z]+',
            lambda m: orig_words.get(m.group(), m.group()),
            fake_punc,
        )
        self.assertIn("WER", restored)
        self.assertIn("DEMO", restored)
        self.assertNotIn("wer", restored)
        self.assertNotIn("demo", restored)

    def test_mixed_case_preserved(self):
        """Mixed case words like iPhone should be preserved."""
        text = "我用iPhone拍照"
        orig_words = {w.lower(): w for w in re.findall(r'[a-zA-Z]+', text)}
        self.assertEqual(orig_words["iphone"], "iPhone")

        fake_punc = "我用iphone拍照。"
        restored = re.sub(
            r'[a-zA-Z]+',
            lambda m: orig_words.get(m.group(), m.group()),
            fake_punc,
        )
        self.assertIn("iPhone", restored)

    def test_pure_chinese_unaffected(self):
        """Pure Chinese text has no English words to map."""
        text = "今天天气不错"
        orig_words = {w.lower(): w for w in re.findall(r'[a-zA-Z]+', text)}
        self.assertEqual(orig_words, {})

    def test_empty_string(self):
        """Empty string returns empty from process_with_firered_punc."""
        # Use a dummy model — empty string short-circuits before model call
        result = PostProcessorInference.process_with_firered_punc(None, "")
        self.assertEqual(result, "")


class TestFireRedPuncIntegration(unittest.TestCase):
    """Integration tests using the real FireRedPunc model. No mocks."""

    model = None

    @classmethod
    def setUpClass(cls):
        """Load FireRedPunc model once for all integration tests."""
        model_dir = os.path.expanduser("~/.local/share/voice-input/models/FireRedPunc")
        cls.model = PostProcessorLoader.load_firered_punc({"model_dir": model_dir})

    def test_model_loads(self):
        """FireRedPunc model loads successfully."""
        self.assertIsNotNone(self.model)

    def test_mixed_chinese_english_with_case(self):
        """Chinese+English input gets punctuation and English case preserved."""
        text = "也不知道WER到底是多少再录一下这个DEMO"
        result = PostProcessorInference.process_with_firered_punc(self.model, text)
        # English case preserved
        self.assertIn("WER", result)
        self.assertIn("DEMO", result)
        # Chinese punctuation added
        has_punc = bool(re.search(r'[。，？！、；：]', result))
        self.assertTrue(has_punc, f"Expected Chinese punctuation in: {result}")

    def test_pure_chinese_gets_punctuation(self):
        """Pure Chinese input gets punctuation added."""
        text = "今天天气不错我们出去走走吧"
        result = PostProcessorInference.process_with_firered_punc(self.model, text)
        has_punc = bool(re.search(r'[。，？！、；：]', result))
        self.assertTrue(has_punc, f"Expected Chinese punctuation in: {result}")

    def test_filler_removal_then_punctuation(self):
        """Filler removal + punctuation pipeline (daemon's _post_process order)."""
        text = "嗯今天天气不错我们出去走走吧"
        # Step 1: Remove fillers (as daemon does)
        cleaned = PostProcessorInference.remove_fillers(text)
        self.assertNotIn("嗯", cleaned)
        # Step 2: Apply punctuation (as daemon does when punc_model is set)
        result = PostProcessorInference.process_with_firered_punc(self.model, cleaned)
        has_punc = bool(re.search(r'[。，？！、；：]', result))
        self.assertTrue(has_punc, f"Expected Chinese punctuation in: {result}")


if __name__ == "__main__":
    unittest.main()
