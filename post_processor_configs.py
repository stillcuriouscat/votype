#!/usr/bin/env python3
"""
Post-processor loading and inference logic.
Configuration is imported from post_processor_presets.py.

Pipeline: regex filler removal (always) -> LLM refinement (optional)
"""

import re
import logging
from typing import Any, Optional

from post_processor_presets import POST_PROCESSOR_PRESETS, DEFAULT_POST_PROCESSOR


# Regex patterns for filler words (Chinese + English)
# Chinese fillers don't have spaces, so no word boundary needed
CHINESE_FILLER_PATTERN = re.compile(
    r'(?:'
    r'那个那个|就是说|然后嘛|'  # multi-char fillers first (greedy)
    r'呃+|嗯+|额+|哦+'         # single-char fillers (repeated)
    r')'
    r'[，,、\s]*',              # trailing punctuation/space after filler
    re.UNICODE
)
# Chinese 啊 is tricky: only remove when standalone or sentence-initial, not in words like 天气啊
CHINESE_STANDALONE_AH_PATTERN = re.compile(
    r'(?:^|(?<=[，,。！？\s]))啊+[，,、\s]*',
    re.UNICODE
)
# English fillers need word boundaries
ENGLISH_FILLER_PATTERN = re.compile(
    r'(?:^|(?<=\s))'
    r'(?:[Uu]mm?|[Uu]hh?|[Ee]rr?|[Aa]hh?|[Ll]ike|[Yy]ou know)'
    r'[,\s]*',
    re.UNICODE
)

# Pattern for repeated punctuation cleanup
REPEATED_PUNCT_PATTERN = re.compile(r'[，,、]{2,}')
# Pattern for leading punctuation
LEADING_PUNCT_PATTERN = re.compile(r'^[，,、\s]+')


class PostProcessorLoader:
    """Load post-processor models (llama-cpp GGUF) and punctuation models (FireRedPunc BERT)."""

    @staticmethod
    def load_firered_punc(config: dict) -> Any:
        """Load a FireRedPunc BERT punctuation model.

        Args:
            config: Config dict with model_dir pointing to FireRedPunc model directory.

        Returns:
            FireRedPunc instance
        """
        from fireredasr2s.fireredpunc.punc import FireRedPunc, FireRedPuncConfig

        model_dir = config["model_dir"]
        logging.info(f"Loading FireRedPunc model: {model_dir}")

        punc_config = FireRedPuncConfig(use_gpu=False)
        model = FireRedPunc.from_pretrained(model_dir, punc_config)
        return model

    @staticmethod
    def load_llama_model(config: dict) -> Any:
        """Load a GGUF model via llama-cpp-python.

        Args:
            config: Model config dict with model_path, n_ctx, n_gpu_layers, etc.

        Returns:
            llama_cpp.Llama instance
        """
        from llama_cpp import Llama

        model_path = config["model_path"]
        logging.info(f"Loading GGUF model: {model_path}")

        model = Llama(
            model_path=model_path,
            n_ctx=config.get("n_ctx", 2048),
            n_gpu_layers=config.get("n_gpu_layers", -1),
            verbose=False,
        )
        return model

    @classmethod
    def load_post_processor(cls, preset_id: str) -> Optional[Any]:
        """Load a post-processor by preset ID.

        Args:
            preset_id: Key in POST_PROCESSOR_PRESETS

        Returns:
            Loaded model instance, or None for "none" preset.

        Raises:
            ValueError: If preset_id is unknown.
            RuntimeError: If model loading fails.
        """
        if preset_id not in POST_PROCESSOR_PRESETS:
            raise ValueError(f"Unknown post-processor: {preset_id}")

        preset = POST_PROCESSOR_PRESETS[preset_id]
        framework = preset["framework"]

        if framework == "regex":
            return None  # No model needed

        if framework == "llama-cpp":
            try:
                return cls.load_llama_model(preset["config"])
            except Exception as e:
                raise RuntimeError(f"Failed to load post-processor {preset_id}: {e}")

        raise ValueError(f"Unknown post-processor framework: {framework}")


class PostProcessorInference:
    """Post-processing inference engine."""

    @staticmethod
    def remove_fillers(text: str) -> str:
        """Remove filler words from text using regex.

        Always runs regardless of LLM post-processor selection.

        Args:
            text: Raw transcription text

        Returns:
            Text with filler words removed
        """
        if not text:
            return text

        result = CHINESE_FILLER_PATTERN.sub('', text)
        result = CHINESE_STANDALONE_AH_PATTERN.sub('', result)
        result = ENGLISH_FILLER_PATTERN.sub('', result)
        # Clean up repeated punctuation left by filler removal
        result = REPEATED_PUNCT_PATTERN.sub('，', result)
        # Remove leading punctuation
        result = LEADING_PUNCT_PATTERN.sub('', result)
        return result.strip()

    @staticmethod
    def process_with_firered_punc(model: Any, text: str) -> str:
        """Process text through FireRedPunc for punctuation restoration.

        Preserves original English case by lowercasing input for the model
        (prevents tokenizer [UNK] on uppercase) and restoring case after.

        Args:
            model: FireRedPunc instance
            text: Text to add punctuation to

        Returns:
            Punctuated text with original English case preserved
        """
        if not text:
            return text

        # Build case mapping: lowered -> original for English words
        orig_words = {w.lower(): w for w in re.findall(r'[a-zA-Z]+', text)}

        # FireRedPunc expects lowercase English to avoid [UNK] tokens
        lowered = text.lower() if orig_words else text

        # Run punctuation restoration (batch API: list in, list of dicts out)
        results = model.process([lowered])
        result = results[0]["punc_text"]

        # Restore original English case
        if orig_words:
            result = re.sub(
                r'[a-zA-Z]+',
                lambda m: orig_words.get(m.group(), m.group()),
                result,
            )

        return result

    @staticmethod
    def process_with_llm(model: Any, text: str, prompt_template: str) -> str:
        """Process text through an LLM for refinement.

        Args:
            model: llama_cpp.Llama instance
            text: Text to refine (after filler removal)
            prompt_template: Prompt template with {text} placeholder

        Returns:
            Refined text from LLM
        """
        prompt = prompt_template.format(text=text)

        output = model.create_completion(
            prompt,
            max_tokens=len(text) * 3,  # allow generous output length
            temperature=0.1,           # low temperature for deterministic correction
            stop=["\n\n"],             # stop at double newline
            echo=False,
        )

        result = output["choices"][0]["text"].strip()
        # Guard against empty or obviously hallucinated output
        if not result or len(result) > len(text) * 5:
            logging.warning("LLM post-processor returned suspicious output, using original text")
            return text
        return result

    @classmethod
    def process(cls, text: str, model: Optional[Any], preset_id: str) -> str:
        """Unified post-processing interface.

        Pipeline: regex filler removal -> optional LLM refinement.

        Args:
            text: Raw transcription text
            model: Loaded LLM model (None for regex-only mode)
            preset_id: Post-processor preset ID

        Returns:
            Processed text
        """
        if not text:
            return text

        # Step 1: Always remove fillers
        result = cls.remove_fillers(text)

        if not result:
            return result

        # Step 2: Optional model-based refinement
        preset = POST_PROCESSOR_PRESETS.get(preset_id, {})
        framework = preset.get("framework")

        if framework == "llama-cpp" and model is not None:
            prompt_template = preset.get("config", {}).get("prompt_template", "")
            if prompt_template:
                try:
                    result = cls.process_with_llm(model, result, prompt_template)
                except Exception as e:
                    logging.error(f"LLM post-processing failed, using regex-only result: {e}")

        return result
