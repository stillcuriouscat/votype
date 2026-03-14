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
            "min_text_len": 45,

            "vocab_min_count": 3,
            # Prompt files adapted from voxy (github.com/hahagood/voxy)
            # Custom terms injected at runtime via glossary_context()
            "system_prompt_file": "prompts/haiku-fix-system.txt",
            "user_prompt_template_file": "prompts/haiku-fix-user.txt",
        },
    },
    "haiku-expand": {
        "name": "Haiku Expand (placeholder)",
        "description": "Not yet implemented",
        "framework": "ssh-claude",
        "config": {},
    },
    "gemini-fix": {
        "name": "Gemini Fix (Vertex AI)",
        "description": "ASR error correction via Gemini 2.5 Flash (~5-7s latency)",
        "framework": "vertex-ai",
        "config": {
            "ssh_host": "oracle-cloud",
            "proxy_script": "~/vertex_proxy.py",
            "model": "gemini-2.5-flash",
            "vertex_region": "us-central1",
            "timeout": 15,
            "min_text_len": 45,

            "vocab_min_count": 3,
            # Prompt file: copy of haiku-fix-system.txt without /no_think (Claude-specific)
            # Gemini needs larger initial glossary for proper nouns (e.g. 克劳德→Claude not Cloud)
            "system_prompt_file": "prompts/gemini-fix-system.txt",
            "user_prompt_template_file": "prompts/haiku-fix-user.txt",
        },
    },
    "gemini-merge": {
        "name": "Gemini Merge (Dual ASR)",
        "description": "Merge SenseVoice + faster-whisper via Gemini 2.5 Flash for best Chinese-English accuracy",
        "framework": "vertex-ai-merge",
        "config": {
            "ssh_host": "oracle-cloud",
            "proxy_script": "~/vertex_proxy.py",
            "model": "gemini-2.5-flash",
            "vertex_region": "us-central1",
            "timeout": 15,
            "min_text_len": 45,

            "vocab_min_count": 3,
            "system_prompt_file": "prompts/gemini-merge-system.txt",
        },
    },
}

# Default post-processor (regex only, no LLM overhead)
DEFAULT_POST_PROCESSOR = "none"
