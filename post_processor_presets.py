#!/usr/bin/env python3
"""
Post-processor preset configurations.
Defines available text post-processing options for ASR output.

Pipeline: ASR transcription -> regex filler removal (always) -> LLM refinement (optional)
"""

from pathlib import Path

# Directory for GGUF model files
MODELS_DIR = Path.home() / ".local/share/voice-input/models"

# Post-processor preset configurations
POST_PROCESSOR_PRESETS = {
    "none": {
        "name": "Regex Only",
        "description": "Remove filler words only, no LLM refinement",
        "framework": "regex",
    },
    "chinese-text-correction": {
        "name": "Chinese Text Correction 1.5B",
        "description": "Specialized Chinese error correction model (GGUF Q4_K_M)",
        "framework": "llama-cpp",
        "config": {
            "model_path": str(MODELS_DIR / "chinese-text-correction-1.5b.Q4_K_M.gguf"),
            "prompt_template": "请修正以下语音转录文本中的错误，只输出修正后的文本，不要添加任何解释：\n{text}",
            "n_ctx": 2048,
            "n_gpu_layers": 0,  # CPU-only: ASR model occupies all GPU VRAM
        },
    },
    "qwen3-0.6b": {
        "name": "Qwen3 0.6B",
        "description": "General-purpose small model for text refinement (GGUF Q4_K_M)",
        "framework": "llama-cpp",
        "config": {
            "model_path": str(MODELS_DIR / "Qwen3-0.6B-Q4_K_M.gguf"),
            "prompt_template": "请修正以下语音转录文本中的错别字和语法错误，只输出修正后的文本：\n{text}",
            "n_ctx": 2048,
            "n_gpu_layers": 0,  # CPU-only: ASR model occupies all GPU VRAM
        },
    },
    "minicpm4-0.5b": {
        "name": "MiniCPM4 0.5B",
        "description": "Best Chinese 0.5B model for text refinement (GGUF Q4_K_M)",
        "framework": "llama-cpp",
        "config": {
            "model_path": str(MODELS_DIR / "MiniCPM4-0.5B.Q4_K_M.gguf"),
            "prompt_template": "请修正以下语音转录文本中的错别字和语法错误，只输出修正后的文本：\n{text}",
            "n_ctx": 2048,
            "n_gpu_layers": 0,  # CPU-only: ASR model occupies all GPU VRAM
        },
    },
}

# Default post-processor (regex only, no LLM overhead)
DEFAULT_POST_PROCESSOR = "none"
