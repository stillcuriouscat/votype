"""Regression test: long audio (>30s) must not be truncated.

Background (2026-03-14):
  SenseVoice Small has an architectural limit of ~30s (attention trained on <=30s).
  Without VAD segmentation, a 37s recording was truncated to 3 characters ("我一下").
  Fix: add fsmn-vad + max_single_segment_time=30000 to sensevoice preset.

This test uses a real 37s recording (tests/fixtures/long_speech_37s.wav) that
triggered the original bug.  It loads the actual SenseVoice model with the preset
config and asserts the transcription is reasonably long.

Regression (2026-03-14) — 187s VAD truncation:
  A 187-second (3m7s) recording was truncated to ~50 chars (one sentence) because
  fsmn-vad was absent from the sensevoice preset.  With VAD the correct output is
  ~772 chars.  TestLongAudio187sRegression covers this case.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_WAV = PROJECT_ROOT / "tests" / "fixtures" / "long_speech_37s.wav"

# The recording is 37s of continuous speech.  Any reasonable transcription
# should produce at least 30 Chinese characters (~30 bytes in len()).
MIN_EXPECTED_CHARS = 30


@pytest.mark.slow
@pytest.mark.real_model
class TestLongAudioRegression:
    """Ensure >30s audio is not silently truncated by ASR models."""

    @pytest.fixture(scope="class")
    def sensevoice_model(self):
        """Load SenseVoice with the exact preset config (including VAD)."""
        try:
            from model_presets import MODEL_PRESETS
            from funasr import AutoModel
        except ImportError:
            pytest.skip("FunASR not installed")

        preset = MODEL_PRESETS["sensevoice"]
        config = preset["config"]

        model = AutoModel(
            model=config["model"],
            vad_model=config.get("vad_model"),
            vad_kwargs=config.get("vad_kwargs"),
            disable_update=True,
        )
        yield model

    @pytest.fixture(scope="class")
    def faster_whisper_model(self):
        """Load faster-whisper with the same config used in production."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            pytest.skip("faster-whisper not installed")

        model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
        yield model

    def test_sensevoice_long_audio_not_truncated(self, sensevoice_model):
        """Primary ASR (SenseVoice) must transcribe >30s audio fully.

        Without VAD, SenseVoice returns ~3 chars for 37s audio.
        With VAD (fsmn-vad, max_single_segment_time=30000), it should
        segment the audio and return a full transcription.
        """
        assert FIXTURE_WAV.exists(), f"Fixture not found: {FIXTURE_WAV}"

        import re
        result = sensevoice_model.generate(input=str(FIXTURE_WAV))
        # FunASR returns list of dicts with "text" key
        raw_text = result[0]["text"] if result else ""
        # Strip SenseVoice special tokens
        text = re.sub(r'<\|[^|]*\|>', '', raw_text).strip()

        assert len(text) >= MIN_EXPECTED_CHARS, (
            f"SenseVoice transcription too short for 37s audio: "
            f"got {len(text)} chars ({text!r}), expected >= {MIN_EXPECTED_CHARS}. "
            f"Check that VAD (fsmn-vad) is configured in sensevoice preset."
        )

    def test_faster_whisper_long_audio_not_truncated(self, faster_whisper_model):
        """Secondary ASR (faster-whisper) must transcribe >30s audio fully."""
        assert FIXTURE_WAV.exists(), f"Fixture not found: {FIXTURE_WAV}"

        segments, _info = faster_whisper_model.transcribe(
            str(FIXTURE_WAV), language="zh"
        )
        text = "".join(s.text for s in segments).strip()

        assert len(text) >= MIN_EXPECTED_CHARS, (
            f"faster-whisper transcription too short for 37s audio: "
            f"got {len(text)} chars ({text!r}), expected >= {MIN_EXPECTED_CHARS}."
        )

    def test_sensevoice_preset_has_vad(self):
        """Preset config must include VAD to prevent long audio truncation."""
        from model_presets import MODEL_PRESETS

        preset = MODEL_PRESETS["sensevoice"]
        config = preset["config"]

        assert "vad_model" in config, (
            "sensevoice preset missing 'vad_model' — long audio (>30s) will be truncated. "
            "Add: \"vad_model\": \"fsmn-vad\""
        )
        assert "vad_kwargs" in config, (
            "sensevoice preset missing 'vad_kwargs' — add: "
            "\"vad_kwargs\": {\"max_single_segment_time\": 30000}"
        )
        assert config["vad_kwargs"].get("max_single_segment_time", 0) <= 30000, (
            "max_single_segment_time should be <= 30000 (30s) for SenseVoice"
        )


@pytest.mark.slow
@pytest.mark.real_model
class TestLongAudio187sRegression:
    """Regression: 187s audio must not be truncated to a single sentence.

    Root cause: fsmn-vad missing from sensevoice preset → SenseVoice's
    internal 30s attention window silently drops everything after the first
    segment, producing ~50 chars instead of ~772 chars.

    Fix: "vad_model": "fsmn-vad" + "vad_kwargs": {"max_single_segment_time": 30000}
    in model_presets.py sensevoice entry.
    """

    FIXTURE_WAV = PROJECT_ROOT / "tests" / "fixtures" / "long_speech_187s_truncated.wav"

    # With VAD the reference output for this recording is ~772 chars.
    # We require at least 500 chars to allow for minor model variation while
    # still catching the truncation-to-one-sentence regression (~50 chars).
    MIN_EXPECTED_CHARS = 500

    # fsmn-vad on 187s of speech produces ~38 segments at max_single_segment_time=30s.
    # Requiring at least 5 segments rules out the no-VAD case (1 segment) and
    # is resilient to future tuning of max_single_segment_time.
    MIN_EXPECTED_SEGMENTS = 5

    @pytest.fixture(scope="class")
    def sensevoice_model_187(self):
        """Load SenseVoice with the exact production preset (including VAD)."""
        try:
            from model_presets import MODEL_PRESETS
            from funasr import AutoModel
        except ImportError:
            pytest.skip("FunASR not installed")

        preset = MODEL_PRESETS["sensevoice"]
        config = preset["config"]

        model = AutoModel(
            model=config["model"],
            vad_model=config.get("vad_model"),
            vad_kwargs=config.get("vad_kwargs"),
            disable_update=True,
        )
        yield model

    def test_sensevoice_187s_produces_full_transcription(self, sensevoice_model_187):
        """SenseVoice + VAD must return >= 500 chars for the 187s fixture.

        Without VAD: ~50 chars (one sentence, first 30s only).
        With VAD:    ~772 chars (full recording, correct output).
        """
        import re

        assert self.FIXTURE_WAV.exists(), f"Fixture not found: {self.FIXTURE_WAV}"

        result = sensevoice_model_187.generate(input=str(self.FIXTURE_WAV), language="zh")
        raw_text = result[0]["text"] if result else ""
        text = re.sub(r'<\|[^|]*\|>', '', raw_text).strip()

        assert len(text) >= self.MIN_EXPECTED_CHARS, (
            f"SenseVoice transcription too short for 187s audio: "
            f"got {len(text)} chars ({text[:80]!r}...), expected >= {self.MIN_EXPECTED_CHARS}. "
            f"This likely means VAD (fsmn-vad) is missing from the sensevoice preset — "
            f"audio was truncated to the first ~30s only."
        )

    def test_sensevoice_187s_vad_produces_multiple_segments(self, sensevoice_model_187):
        """VAD must split 187s audio into multiple segments (expected ~38).

        Without VAD: generate() returns 1 result entry (no segmentation).
        With VAD:    generate() returns multiple result entries (one per VAD chunk).
        Requiring >= 5 confirms segmentation is active and the bug cannot regress.
        """
        assert self.FIXTURE_WAV.exists(), f"Fixture not found: {self.FIXTURE_WAV}"

        result = sensevoice_model_187.generate(input=str(self.FIXTURE_WAV), language="zh")

        assert len(result) >= self.MIN_EXPECTED_SEGMENTS, (
            f"SenseVoice returned only {len(result)} segment(s) for 187s audio, "
            f"expected >= {self.MIN_EXPECTED_SEGMENTS}. "
            f"This indicates VAD segmentation is not active — check that "
            f"'vad_model': 'fsmn-vad' is set in the sensevoice preset."
        )
