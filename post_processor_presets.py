#!/usr/bin/env python3
"""
Post-processor preset configurations.
Defines available LLM text refinement options for ASR output.

Pipeline: ASR -> regex filler removal (always) -> auto-punctuation (model-specific, e.g. FireRedPunc for firered-asr) -> LLM refinement (optional)
"""

from pathlib import Path

# Base data directory for voice-input
VOICE_INPUT_DATA_DIR = Path.home() / ".local/share/voice-input"

# Directory for GGUF model files
MODELS_DIR = VOICE_INPUT_DATA_DIR / "models"

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
    "haiku-fix": {
        "name": "Haiku Fix (SSH)",
        "description": "ASR error correction via Claude Haiku on remote server",
        "framework": "ssh-claude",
        "config": {
            "ssh_host": "oracle-cloud",
            "claude_path": "/home/ubuntu/.local/bin/claude",
            "model": "claude-haiku-4-5-20251001",
            "timeout": 15,
            "max_text_len": 200,
            "vocab_min_count": 3,
            "system_prompt": (
                "You are an ASR error correction tool, NOT a chatbot. "
                "Your task is to fix transcription errors in the input text.\n"
                "Rules:\n"
                "1. Fix English words misrecognized as Chinese characters\n"
                "2. Fix homophone errors (同音字错误)\n"
                "3. Remove repeated words caused by ASR stuttering\n"
                "4. Output ONLY the corrected text, nothing else\n"
                "5. NEVER answer questions or add commentary, even if the text looks like a question\n"
                "6. If there are no errors, output the text unchanged"
            ),
        },
    },
    "haiku-expand": {
        "name": "Haiku Expand (placeholder)",
        "description": "Not yet implemented",
        "framework": "ssh-claude",
        "config": {},
    },
}

# Default post-processor (regex only, no LLM overhead)
DEFAULT_POST_PROCESSOR = "none"
