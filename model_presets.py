#!/usr/bin/env python3
"""
Voice recognition model configuration
Contains configuration for all supported ASR models

Note: The following models have been confirmed unable to run locally and removed:
- glm-asr-nano: Requires Transformers 5.x, not supported by current version
- qwen3-asr-flash: API only, no open weights available
- qwen2-audio: 7B model requires 14GB+ VRAM, 8GB GPU will timeout
"""

# ASR model preset configurations (only verified working models)
MODEL_PRESETS = {
    # ===== FunASR framework models =====
    "fun-asr-nano": {
        "name": "Fun-ASR-Nano",
        "description": "Latest end-to-end model, 31 languages, high accuracy low latency",
        "framework": "funasr",
        "punctuation": "builtin",
        "config": {
            "model": "FunAudioLLM/Fun-ASR-Nano-2512",
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
        },
    },
    "paraformer": {
        "name": "Paraformer-zh",
        "description": "Fast and lightweight, Chinese-optimized (CPU mode)",
        "framework": "funasr",
        "punctuation": "builtin",
        "force_cpu": True,  # RTX 5060 (sm_120) incompatible with current PyTorch, requires CPU
        "config": {
            "model": "paraformer-zh",
            "vad_model": "fsmn-vad",
            "punc_model": "ct-punc",
        },
    },
    "sensevoice": {
        "name": "SenseVoice",
        "description": "Chinese-English mixed, multilingual support",
        "framework": "funasr",
        "punctuation": "builtin",
        "config": {
            "model": "iic/SenseVoiceSmall",
        },
    },

    # ===== FireRedASR framework models =====
    "firered-asr": {
        "name": "FireRedASR-AED",
        "description": "Xiaohongshu AED model, Chinese SOTA, Chinese-English mixed optimization",
        "framework": "fireredasr",
        "punctuation": "firered-punc",
        "punc_config": {
            "model_dir": "~/.local/share/voice-input/models/FireRedPunc",
        },
        "config": {
            "model_type": "aed",
            "model_id": "FireRedTeam/FireRedASR-AED-L",
            "modelscope_id": "pengzhendong/FireRedASR-AED-L",
        },
    },
}

# Default model
DEFAULT_MODEL = "firered-asr"

# Device configuration (prefer CUDA)
DEVICE = "cuda:0"

# Hotword configuration (improve recognition accuracy for technical terms)
HOTWORDS = "software engineer machine learning artificial intelligence Python Claude API React TypeScript GitHub Docker Kubernetes AWS Azure"
