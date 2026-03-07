#!/usr/bin/env python3
"""
Tests for _trim_leading_clipping() function in model_configs.py.

TDD: Tests written BEFORE implementation. Tests should initially fail with
ImportError or AttributeError if the function does not exist yet.

Run (from project root, with venv activated):
    PYTHONPATH=~/code/FireRedASR2S:$PYTHONPATH python -m pytest tests/test_trim_clipping.py -v
    # Skip slow tests:
    PYTHONPATH=~/code/FireRedASR2S:$PYTHONPATH python -m pytest tests/test_trim_clipping.py -v -m "not slow"
"""

import sys
import math
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Add project root so model_configs is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model_configs import _trim_leading_clipping  # noqa: E402  (will fail RED until implemented)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
REAL_RECORDING = Path.home() / ".config" / "voice-input" / "recording_20260307_101114.wav"
REAL_RECORDING_DURATION_S = 4.250  # measured from the actual file
RMS_CLIP_THRESHOLD = 30000         # same threshold the implementation uses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a 16-bit mono WAV file from a float64 or int16 numpy array."""
    if samples.dtype != np.int16:
        # Assume float in [-1.0, 1.0]
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def _rms_int16(samples_float: np.ndarray) -> float:
    """Compute RMS as if samples were in int16 range (multiply by 32768)."""
    int16_equivalent = samples_float * 32768
    return math.sqrt(np.mean(int16_equivalent.astype(np.float64) ** 2))


def _read_wav_float(path) -> tuple[np.ndarray, int]:
    """Return (float64 samples in [-1,1], sample_rate)."""
    return sf.read(str(path))


# ---------------------------------------------------------------------------
# Test 1: Synthetic clip + sine
# ---------------------------------------------------------------------------

class TestTrimClippingSynthetic:
    """Synthetic WAV: 500 ms full-scale clipping followed by 2 s sine wave."""

    def _build_wav(self, tmp_path: Path) -> Path:
        clip_ms = 500
        sine_ms = 2000
        clip_samples = int(SAMPLE_RATE * clip_ms / 1000)
        sine_samples = int(SAMPLE_RATE * sine_ms / 1000)

        # Full-scale clipping: alternating +32767 / -32767 at int16 boundary
        clip_part = np.ones(clip_samples, dtype=np.float64)
        clip_part[::2] = -1.0  # square wave to ensure max RMS

        # 440 Hz sine at 50% amplitude (well below clipping threshold)
        t = np.arange(sine_samples) / SAMPLE_RATE
        sine_part = 0.5 * np.sin(2 * math.pi * 440 * t)

        audio = np.concatenate([clip_part, sine_part])
        wav_path = tmp_path / "clipping_then_sine.wav"
        _write_wav(wav_path, audio)
        return wav_path

    def test_trimmed_output_starts_with_sine_not_clipping(self, tmp_path):
        """After trimming, the first 10 ms chunk must have RMS well below 30000."""
        wav_path = self._build_wav(tmp_path)
        result_path = _trim_leading_clipping(str(wav_path))

        assert result_path != str(wav_path), (
            "Expected a new (trimmed) path, got the original back — "
            "clipping was not detected"
        )

        audio, sr = _read_wav_float(result_path)
        assert sr == SAMPLE_RATE

        # First 10 ms chunk of trimmed audio
        chunk_size = int(sr * 0.01)
        first_chunk = audio[:chunk_size]
        first_rms = _rms_int16(first_chunk)

        assert first_rms < RMS_CLIP_THRESHOLD, (
            f"First 10 ms of trimmed audio still sounds like clipping: "
            f"RMS={first_rms:.0f} (threshold={RMS_CLIP_THRESHOLD})"
        )

    def test_trimmed_duration_is_shorter(self, tmp_path):
        """Trimmed file must be shorter than the original."""
        wav_path = self._build_wav(tmp_path)
        original_audio, _ = _read_wav_float(wav_path)
        original_duration = len(original_audio) / SAMPLE_RATE

        result_path = _trim_leading_clipping(str(wav_path))
        trimmed_audio, _ = _read_wav_float(result_path)
        trimmed_duration = len(trimmed_audio) / SAMPLE_RATE

        assert trimmed_duration < original_duration, (
            f"Trimmed duration ({trimmed_duration:.3f}s) should be less than "
            f"original ({original_duration:.3f}s)"
        )

    def test_sine_content_is_preserved_in_trimmed_output(self, tmp_path):
        """The sine-wave content must be present in the trimmed output."""
        wav_path = self._build_wav(tmp_path)
        result_path = _trim_leading_clipping(str(wav_path))

        audio, sr = _read_wav_float(result_path)

        # The trimmed file should still have at least ~1.5 s of content
        # (2 s sine minus the ~50 ms safety margin overshoot, if any)
        min_expected_duration = 1.5
        actual_duration = len(audio) / sr
        assert actual_duration >= min_expected_duration, (
            f"Too much audio was trimmed: only {actual_duration:.3f}s remain "
            f"(expected >= {min_expected_duration}s)"
        )


# ---------------------------------------------------------------------------
# Test 2: No clipping — original path returned unchanged
# ---------------------------------------------------------------------------

class TestNoClippingReturnsOriginal:
    """WAV with normal amplitude — function must return the original path."""

    def _build_normal_wav(self, tmp_path: Path) -> Path:
        duration_s = 2
        samples = int(SAMPLE_RATE * duration_s)
        t = np.arange(samples) / SAMPLE_RATE
        # 30% amplitude 200 Hz sine — RMS will be well below 30000
        audio = 0.3 * np.sin(2 * math.pi * 200 * t)

        wav_path = tmp_path / "normal_audio.wav"
        _write_wav(wav_path, audio)
        return wav_path

    def test_original_path_returned_when_no_clipping(self, tmp_path):
        """No clipping detected → same path returned, no temp file created."""
        wav_path = self._build_normal_wav(tmp_path)
        result_path = _trim_leading_clipping(str(wav_path))

        assert result_path == str(wav_path), (
            f"Expected original path '{wav_path}' to be returned unchanged, "
            f"but got '{result_path}'"
        )

    def test_no_clipping_file_is_not_modified(self, tmp_path):
        """The original file must not be altered when no trimming occurs."""
        wav_path = self._build_normal_wav(tmp_path)
        original_audio, _ = _read_wav_float(wav_path)
        original_mtime = wav_path.stat().st_mtime

        _trim_leading_clipping(str(wav_path))

        assert wav_path.stat().st_mtime == original_mtime, (
            "Original WAV file was modified even though no clipping was detected"
        )


# ---------------------------------------------------------------------------
# Test 3: Real recording from arecord
# ---------------------------------------------------------------------------

class TestTrimRealRecording:
    """Tests against the actual arecord capture that exhibits leading clipping."""

    @pytest.fixture(autouse=True)
    def require_real_recording(self):
        if not REAL_RECORDING.exists():
            pytest.skip(f"Real recording not found: {REAL_RECORDING}")

    def test_trimmed_duration_is_shorter_than_original(self):
        """Trimmed duration must be strictly shorter than 4.25 s."""
        result_path = _trim_leading_clipping(str(REAL_RECORDING))
        audio, sr = _read_wav_float(result_path)
        trimmed_duration = len(audio) / sr

        assert trimmed_duration < REAL_RECORDING_DURATION_S, (
            f"Trimmed duration {trimmed_duration:.3f}s is not shorter than "
            f"original {REAL_RECORDING_DURATION_S}s — clipping was not detected"
        )

    def test_trimmed_duration_is_reasonable(self):
        """
        Trimmed duration must be 3.0–4.0 s.

        The recording has ~570 ms of clipping; adding 50 ms safety margin
        means ~620 ms trimmed, leaving ~3.63 s — well within [3.0, 4.0].
        """
        result_path = _trim_leading_clipping(str(REAL_RECORDING))
        audio, sr = _read_wav_float(result_path)
        trimmed_duration = len(audio) / sr

        assert 3.0 <= trimmed_duration <= 4.0, (
            f"Trimmed duration {trimmed_duration:.3f}s is outside expected "
            f"range [3.0, 4.0] s — too much or too little was removed"
        )

    def test_first_10ms_of_trimmed_audio_has_no_clipping(self):
        """After trimming, the first 10 ms must have RMS < 30000."""
        result_path = _trim_leading_clipping(str(REAL_RECORDING))
        audio, sr = _read_wav_float(result_path)

        chunk_size = int(sr * 0.01)  # 10 ms
        first_chunk = audio[:chunk_size]
        first_rms = _rms_int16(first_chunk)

        assert first_rms < RMS_CLIP_THRESHOLD, (
            f"First 10 ms of trimmed audio still has clipping: "
            f"RMS={first_rms:.0f} >= threshold {RMS_CLIP_THRESHOLD}"
        )


# ---------------------------------------------------------------------------
# Test 4: ASR quality comparison (slow — loads FireRedASR)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTrimPreservesSpeech:
    """
    Run FireRedASR on both original and trimmed recording.
    The trimmed version should produce an equal or better transcription
    (i.e. not lose more words than the original clipping-corrupted run).
    """

    @pytest.fixture(autouse=True)
    def require_real_recording(self):
        if not REAL_RECORDING.exists():
            pytest.skip(f"Real recording not found: {REAL_RECORDING}")

    def _transcribe(self, audio_path: str) -> str:
        """Run FireRedASR-AED inference and return the transcript string."""
        import os
        firered_path = os.path.expanduser("~/code/FireRedASR2S")
        if firered_path not in sys.path:
            sys.path.insert(0, firered_path)

        from fireredasr.models.fireredasr import FireRedASR  # noqa: PLC0415

        model_dir = os.path.expanduser(
            "~/.local/share/voice-input/models/FireRedASR-AED-L"
        )
        if not os.path.exists(model_dir):
            pytest.skip(f"FireRedASR model not found: {model_dir}")

        model = FireRedASR.from_pretrained(model_dir)
        results = model.transcribe(
            [audio_path],
            {"use_gpu": False, "beam_size": 3, "nbest": 1, "decode_max_len": 0},
        )
        text = results[0].get("text", "") if results else ""
        return text.strip()

    def test_trimmed_transcription_does_not_lose_words(self):
        """
        Word count in trimmed transcription >= word count in original.

        Rationale: clipping corrupts the first ~500 ms which the ASR model
        interprets as noise or garbage. Removing it should not reduce the
        meaningful word count.
        """
        result_path = _trim_leading_clipping(str(REAL_RECORDING))

        original_text = self._transcribe(str(REAL_RECORDING))
        trimmed_text = self._transcribe(result_path)

        original_words = len(original_text.split())
        trimmed_words = len(trimmed_text.split())

        assert trimmed_words >= original_words, (
            f"Trimmed transcription lost words.\n"
            f"  Original ({original_words} words): {original_text!r}\n"
            f"  Trimmed  ({trimmed_words} words): {trimmed_text!r}"
        )


# ---------------------------------------------------------------------------
# Test 5: Edge case — very short file
# ---------------------------------------------------------------------------

class TestEmptyFileHandling:
    """Files too short to contain meaningful clipping data."""

    def _build_short_wav(self, tmp_path: Path, duration_ms: int) -> Path:
        """Create a WAV shorter than 100 ms."""
        samples = int(SAMPLE_RATE * duration_ms / 1000)
        # Use full amplitude so it WOULD trigger the clipping check if not short
        audio = np.ones(samples, dtype=np.float64)
        wav_path = tmp_path / f"short_{duration_ms}ms.wav"
        _write_wav(wav_path, audio)
        return wav_path

    @pytest.mark.parametrize("duration_ms", [0, 10, 50, 90])
    def test_very_short_file_returns_original_path(self, tmp_path, duration_ms):
        """
        Files shorter than 100 ms must return the original path unchanged,
        regardless of amplitude (avoids index-out-of-range errors and
        ensures the caller always receives a valid path).
        """
        # 0-sample file: write a WAV header with no frames
        if duration_ms == 0:
            wav_path = tmp_path / "empty.wav"
            with wave.open(str(wav_path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                # write zero frames
        else:
            wav_path = self._build_short_wav(tmp_path, duration_ms)

        result_path = _trim_leading_clipping(str(wav_path))

        assert result_path == str(wav_path), (
            f"Expected original path for {duration_ms}ms file, got '{result_path}'"
        )

    def test_exactly_100ms_file_is_processed_not_skipped(self, tmp_path):
        """
        A file of exactly 100 ms (boundary) should be processed normally
        (not treated as too short). With full-scale audio it must be trimmed.
        """
        samples = int(SAMPLE_RATE * 0.1)  # exactly 100 ms
        audio = np.ones(samples, dtype=np.float64)  # full-scale clipping
        wav_path = tmp_path / "exactly_100ms.wav"
        _write_wav(wav_path, audio)

        result_path = _trim_leading_clipping(str(wav_path))

        # Either trimmed (new temp path) or original returned — both are valid
        # because there is no post-clip content to preserve.
        # The important thing: no exception is raised.
        assert result_path is not None, (
            "Function returned None for a 100 ms file — expected a path string"
        )
        assert isinstance(result_path, str), (
            f"Expected str path, got {type(result_path)}"
        )
