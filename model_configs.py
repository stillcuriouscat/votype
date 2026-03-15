#!/usr/bin/env python3
"""
ASR model loading and inference logic.
Configuration is imported from model_presets.py.
"""

from typing import Dict, Any, Optional, List
import logging
import os

# Import from configuration file
from model_presets import MODEL_PRESETS, DEFAULT_MODEL, DEVICE, HOTWORDS


class ModelLoader:
    """Model loader - loads different models based on framework type."""

    @staticmethod
    def _find_modelscope_cache(modelscope_id: Optional[str]) -> Optional[str]:
        """
        Find the ModelScope cache path.

        Args:
            modelscope_id: ModelScope repository ID (e.g. "Qwen/Qwen2-Audio")

        Returns:
            Cache path if it exists, otherwise None.
        """
        if not modelscope_id:
            return None

        try:
            cache_root = os.environ.get('MODELSCOPE_CACHE', os.path.expanduser('~/.cache/modelscope'))
            repo_parts = modelscope_id.split('/')
            if len(repo_parts) == 2:
                potential_path = os.path.join(cache_root, 'hub', 'models', repo_parts[0], repo_parts[1])
                if os.path.exists(potential_path):
                    logging.info(f"Found ModelScope cache at: {potential_path}")
                    return potential_path
        except Exception as e:
            logging.debug(f"Failed to check ModelScope cache: {e}")

        return None

    @staticmethod
    def load_funasr_model(config: Dict[str, Any], device: str) -> Any:
        """Load a FunASR framework model."""
        from funasr import AutoModel

        # Manually import FunASRNano class to trigger registration (fix for FunASR bug #2757)
        try:
            from funasr.models.fun_asr_nano.model import FunASRNano
        except ImportError:
            pass

        model_config = config.copy()
        model_config["device"] = device
        model_config["disable_update"] = True

        model = AutoModel(**model_config)
        logging.info("[MODEL] FunASR model loaded: %s", config.get("model", config.get("model_id", "unknown")))
        return model

    @staticmethod
    def load_transformers_model(config: Dict[str, Any], device: str = "cpu") -> tuple:
        """
        Load a Transformers framework model (e.g. Qwen2-Audio).
        Prefers loading from ModelScope cache.
        Automatically uses 4-bit quantization when GPU VRAM is insufficient.
        """
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        hf_id = config["model_id"]
        model_path = ModelLoader._find_modelscope_cache(config.get("modelscope_id"))
        if model_path is None:
            model_path = hf_id
            logging.info(f"Loading from HuggingFace ID: {model_path}")

        device_map = "cpu" if device == "cpu" else config.get("device_map", "auto")

        # Check GPU VRAM; use 4-bit quantization if less than 16GB
        quantization_config = None
        if device != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    if gpu_memory < 16:
                        logging.info(f"GPU VRAM {gpu_memory:.1f}GB < 16GB, enabling 4-bit quantization")
                        try:
                            from transformers import BitsAndBytesConfig
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                            )
                        except ImportError:
                            logging.warning("bitsandbytes is not installed, cannot use 4-bit quantization, falling back to CPU offload")
            except Exception as e:
                logging.debug(f"Failed to check GPU VRAM: {e}")

        load_kwargs = {
            "device_map": device_map,
            "trust_remote_code": config.get("trust_remote_code", True),
        }
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config

        model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=config.get("trust_remote_code", True)
        )

        logging.info("[MODEL] Transformers model loaded: %s on %s", hf_id, model.device)
        return model, processor

    @staticmethod
    def load_glmasr_model(config: Dict[str, Any], device: str) -> tuple:
        """
        Load GLM-ASR model (using Transformers 5.x).
        Prefers loading from ModelScope cache.
        """
        from transformers import AutoProcessor, AutoModelForSeq2SeqLM

        hf_id = config["model_id"]
        model_path = ModelLoader._find_modelscope_cache(config.get("modelscope_id"))
        if model_path is None:
            model_path = hf_id
            logging.info(f"Loading from HuggingFace ID: {model_path}")

        # Configure device_map based on device
        device_map = "cpu" if device == "cpu" else "auto"

        processor = AutoProcessor.from_pretrained(model_path)
        logging.info("[MODEL] GLM-ASR processor loaded")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map=device_map
        )

        logging.info(f"GLM-ASR model loaded on device: {model.device}")
        return model, processor

    @staticmethod
    def load_fireredasr_model(config: Dict[str, Any], device: str = "cpu") -> tuple:
        """Load a FireRedASR framework model.

        Returns:
            (model, use_gpu) tuple
        """
        import argparse
        import torch

        # PyTorch 2.6+ safety restriction: add argparse.Namespace to the safe globals whitelist
        torch.serialization.add_safe_globals([argparse.Namespace])

        from fireredasr.models.fireredasr import FireRedAsr

        model = FireRedAsr.from_pretrained(config["model_type"])
        logging.info("[MODEL] FireRedASR loaded: %s", config["model_type"])
        use_gpu = device.startswith("cuda") and torch.cuda.is_available()
        return model, use_gpu

    @staticmethod
    def load_faster_whisper_model(config: Dict[str, Any], device: str = "cpu") -> Any:
        """Load a faster-whisper model (WhisperModel).

        Returns:
            WhisperModel instance (CPU-only, int8 quantization).
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. Install it with: "
                "pip install faster-whisper"
            )

        model_size = config.get("model_size", "large-v3-turbo")
        compute_type = config.get("compute_type", "int8")

        logging.info(f"Loading faster-whisper model: {model_size} (compute_type={compute_type}, device={device})")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logging.info("[MODEL] faster-whisper loaded: %s (%s, %s)", model_size, compute_type, device)
        return model

    @classmethod
    def load_model(cls, model_id: str, device: str = DEVICE) -> tuple:
        """
        Load the corresponding model by model ID.

        Returns:
            (model, framework, extra_data) tuple.
        """
        if model_id not in MODEL_PRESETS:
            raise ValueError(f"Unknown model: {model_id}")

        preset = MODEL_PRESETS[model_id]
        framework = preset["framework"]
        config = preset["config"]

        # Some models are incompatible with the current GPU/PyTorch version; force CPU mode
        force_cpu = preset.get("force_cpu", False)
        actual_device = "cpu" if force_cpu else device
        if force_cpu:
            logging.info(f"Model {model_id} forced to use CPU mode")

        logging.info(f"Loading model: {preset['name']} (framework: {framework}, device: {actual_device})")

        if framework == "funasr":
            logging.info("[MODEL] loading model %s (framework=%s, device=%s)", model_id, framework, actual_device)
            model = cls.load_funasr_model(config, actual_device)
            logging.info("[MODEL] model ready: %s", model_id)
            return model, framework, None

        elif framework == "glmasr":
            logging.info("[MODEL] loading model %s (framework=%s, device=%s)", model_id, framework, actual_device)
            model, processor = cls.load_glmasr_model(config, actual_device)
            logging.info("[MODEL] model ready: %s", model_id)
            return model, framework, {"processor": processor}

        elif framework == "fireredasr":
            logging.info("[MODEL] loading model %s (framework=%s, device=%s)", model_id, framework, actual_device)
            model, use_gpu = cls.load_fireredasr_model(config, actual_device)
            logging.info("[MODEL] model ready: %s", model_id)
            return model, framework, {"use_gpu": use_gpu}

        elif framework == "faster-whisper":
            logging.info("[MODEL] loading model %s (framework=%s, device=%s)", model_id, framework, actual_device)
            model = cls.load_faster_whisper_model(config, actual_device)
            logging.info("[MODEL] model ready: %s", model_id)
            return model, framework, None

        else:
            raise ValueError(f"Unknown framework: {framework}")


def _trim_leading_clipping(audio_path: str, threshold: float = 30000,
                           chunk_ms: int = 10, min_clean_ms: int = 30) -> str:
    """Trim leading clipping burst from recorder startup.

    Scans from the start in chunk_ms-sized windows. Clipping ends when
    min_clean_ms consecutive chunks are all below threshold (handles brief
    dips in clipping). Returns a temp file path with the trimmed audio,
    or the original path if no clipping is detected.
    """
    import soundfile as sf
    import numpy as np
    import tempfile

    try:
        data, sr = sf.read(audio_path, dtype="int16")
    except Exception:
        return audio_path
    logging.info("[AUDIO] loaded %s: %.1fs, %dHz", audio_path, len(data)/sr, sr)
    chunk_size = int(chunk_ms * sr / 1000)
    total_chunks = len(data) // chunk_size
    min_clean_chunks = max(1, int(min_clean_ms / chunk_ms))

    if total_chunks == 0:
        return audio_path

    # Check if first chunk is already clean — no clipping to trim
    first_chunk = data[:chunk_size]
    first_rms = np.sqrt(np.mean(first_chunk.astype(np.float64) ** 2))
    if first_rms < threshold:
        logging.info("[AUDIO] no clipping detected, using original")
        return audio_path

    # Find where clipping ends: need min_clean_chunks consecutive clean chunks
    consecutive_clean = 0
    clip_end = total_chunks  # default: all clipping
    for i in range(total_chunks):
        chunk = data[i * chunk_size:(i + 1) * chunk_size]
        rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
        if rms < threshold:
            consecutive_clean += 1
            if consecutive_clean >= min_clean_chunks:
                # Clipping ended at the start of the first clean chunk in this run
                clip_end = i - min_clean_chunks + 1
                break
        else:
            consecutive_clean = 0

    if clip_end >= total_chunks:
        # All chunks are clipping — return original unchanged
        logging.info("[AUDIO] entire audio is clipping, using original")
        return audio_path

    trim_sample = clip_end * chunk_size
    trimmed = data[trim_sample:]

    if len(trimmed) == 0:
        return audio_path

    trimmed_ms = trim_sample * 1000 / sr
    logging.info(f"Trimmed {trimmed_ms:.0f}ms of leading clipping")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, trimmed, sr, subtype="PCM_16")
    logging.info("[AUDIO] trimmed %.0fms clipping → %s", trimmed_ms, tmp.name)
    return tmp.name


def _chunk_audio(audio_path: str, chunk_sec: int = 30) -> List[str]:
    """Split audio into fixed-length chunks, return list of temp file paths.

    Uses a unique ID per call to avoid collisions under concurrency.
    Cleans up all written chunks if an error occurs mid-way.
    """
    import soundfile as sf
    import uuid
    from pathlib import Path

    call_id = uuid.uuid4().hex[:8]
    chunk_paths = []
    try:
        with sf.SoundFile(audio_path) as f:
            frames_per_chunk = int(chunk_sec * f.samplerate)
            i = 0
            while True:
                data = f.read(frames=frames_per_chunk)
                if len(data) == 0:
                    break
                chunk_path = f"/tmp/firered_chunk_{call_id}_{i}.wav"
                sf.write(chunk_path, data, f.samplerate)
                chunk_paths.append(chunk_path)
                i += 1
    except Exception as e:
        logging.info("[CHUNK] chunking failed: %s", e)
        for p in chunk_paths:
            Path(p).unlink(missing_ok=True)
        raise
    logging.info("[CHUNK] split audio into %d chunks (%.1fs each)", len(chunk_paths), chunk_sec)
    return chunk_paths


def _parse_firered_result(result) -> str:
    """Extract text from a FireRedASR transcription result."""
    if isinstance(result, list) and len(result) > 0:
        item = result[0]
        if isinstance(item, dict):
            return item.get('text', '')
        return str(item)
    elif isinstance(result, dict):
        return result.get('text', '')
    return str(result)


class ModelInference:
    """Model inference engine - executes different inference logic based on framework type."""

    @staticmethod
    def transcribe_funasr(model: Any, audio_path: str, model_id: str, hotwords: str = HOTWORDS) -> str:
        """FunASR framework inference."""
        import re

        if model_id == "fun-asr-nano":
            result = model.generate(
                input=[audio_path],
                cache={},
                batch_size=1,
                language="中文",
                itn=True
            )
        elif model_id == "sensevoice":
            result = model.generate(input=audio_path, language="zh")
        else:
            try:
                result = model.generate(input=audio_path, hotword=hotwords)
            except TypeError:
                logging.info("[ASR] FunASR hotword not supported, retrying without")
                result = model.generate(input=audio_path)

        if result and len(result) > 0:
            text = result[0].get("text", "")
            # SenseVoice outputs special tokens: <|zh|><|HAPPY|><|Speech|><|woitn|>
            # Strip all <|...|> tags to get clean transcription text
            raw_text = text
            text = re.sub(r'<\|[^|]*\|>', '', text).strip()
            if raw_text != text:
                logging.info("[ASR] stripped special tokens from FunASR output")
            logging.info("[ASR] FunASR transcription complete (%d chars)", len(text))
            return text
        return ""

    @staticmethod
    def transcribe_transformers(
        model: Any,
        processor: Any,
        audio_path: str,
        model_id: str,
        prompt_template: Optional[str] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        """Transformers framework inference (Qwen2-Audio Instruct)."""
        import librosa

        audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

        # Use chat template format (required by Instruct models)
        conversation = [
            {'role': 'user', 'content': [
                {'type': 'audio', 'audio': audio},
                {'type': 'text', 'text': '请将这段语音转录为文字'},
            ]}
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=[audio], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_kwargs = generate_kwargs or MODEL_PRESETS.get(model_id, {}).get("generate_kwargs", {})
        generated_ids = model.generate(**inputs, **gen_kwargs)

        generated_ids = generated_ids[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        logging.info("[ASR] Transformers transcription complete (%d chars)", len(response))
        return response

    @staticmethod
    def transcribe_glmasr(model: Any, processor: Any, audio_path: str) -> str:
        """GLM-ASR framework inference (Transformers 5.x)."""
        import librosa

        # Load audio and resample to the model's required sample rate
        target_sr = processor.feature_extractor.sampling_rate
        audio, _ = librosa.load(audio_path, sr=target_sr)

        # Process input using apply_transcription_request
        inputs = processor.apply_transcription_request(audio)
        inputs = inputs.to(model.device, dtype=model.dtype)

        # Generate transcription result
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        # Decode output (skip the input portion)
        decoded = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        text = decoded[0] if decoded else ""
        logging.info("[ASR] GLM-ASR transcription complete (%d chars)", len(text))
        return text

    @staticmethod
    def _transcribe_firered_single(model, audio_path: str, use_gpu: bool) -> str:
        """Transcribe a single audio file with FireRedASR."""
        import torch

        if use_gpu:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                result = model.transcribe(['utt1'], [audio_path], args={"use_gpu": use_gpu})
        else:
            result = model.transcribe(['utt1'], [audio_path], args={"use_gpu": use_gpu})
        return _parse_firered_result(result)

    @staticmethod
    def transcribe_fireredasr(model: Any, audio_path: str, use_gpu: bool = False) -> str:
        """FireRedASR framework inference with automatic chunking for long audio.

        Audio longer than 30s is split into 30s chunks to avoid CUDA OOM on 8GB VRAM.
        """
        import soundfile as sf
        from pathlib import Path

        info = sf.info(audio_path)
        duration = info.duration

        if duration <= 30:
            text = ModelInference._transcribe_firered_single(model, audio_path, use_gpu)
            logging.info("[ASR] FireRedASR single-file transcription complete (%d chars)", len(text))
            return text

        # Long audio: chunk into 30s segments to avoid OOM
        logging.info(f"FireRedASR: audio {duration:.1f}s > 30s, chunking...")
        chunks = _chunk_audio(audio_path, chunk_sec=30)
        texts = []
        try:
            for chunk_path in chunks:
                text = ModelInference._transcribe_firered_single(model, chunk_path, use_gpu)
                texts.append(text)
        finally:
            # Always clean up chunk files
            for chunk_path in chunks:
                Path(chunk_path).unlink(missing_ok=True)
        result = "".join(texts)
        logging.info("[ASR] FireRedASR chunked transcription: %d chunks → %d chars", len(chunks), len(result))
        return result

    @staticmethod
    def transcribe_faster_whisper(model: Any, audio_path: str) -> str:
        """Faster-whisper framework inference.

        model.transcribe() returns (Iterable[Segment], TranscriptionInfo).
        Language auto-detected (not forced) to handle mixed Chinese-English.
        """
        segments, _info = model.transcribe(audio_path)
        text = "".join(segment.text for segment in segments)
        logging.info("[ASR] faster-whisper transcription complete (%d chars)", len(text))
        return text

    @classmethod
    def transcribe(
        cls,
        model: Any,
        audio_path: str,
        model_id: str,
        framework: str,
        extra_data: Optional[Dict[str, Any]] = None,
        hotwords: str = HOTWORDS
    ) -> str:
        """Unified transcription interface."""
        original_path = audio_path
        try:
            # Trim leading clipping from arecord startup
            audio_path = _trim_leading_clipping(audio_path)
            logging.info("[AUDIO] clipping check: %s", "trimmed" if audio_path != original_path else "clean")

            logging.info("[ASR] transcribing with framework=%s", framework)

            if framework == "funasr":
                return cls.transcribe_funasr(model, audio_path, model_id, hotwords)

            elif framework == "transformers":
                processor = extra_data.get("processor") if extra_data else None
                if processor is None:
                    raise ValueError("Processor is required for transformers models")
                return cls.transcribe_transformers(model, processor, audio_path, model_id)

            elif framework == "glmasr":
                processor = extra_data.get("processor") if extra_data else None
                if processor is None:
                    raise ValueError("Processor is required for GLM-ASR models")
                return cls.transcribe_glmasr(model, processor, audio_path)

            elif framework == "fireredasr":
                use_gpu = extra_data.get("use_gpu", False) if extra_data else False
                return cls.transcribe_fireredasr(model, audio_path, use_gpu=use_gpu)

            elif framework == "faster-whisper":
                return cls.transcribe_faster_whisper(model, audio_path)

            else:
                raise ValueError(f"Unknown framework: {framework}")

        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            raise
        finally:
            # Clean up trimmed temp file if one was created
            if audio_path != original_path:
                from pathlib import Path
                Path(audio_path).unlink(missing_ok=True)
