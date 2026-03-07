"""
E2E test: verify pw-record has no startup clipping after warmup.

arecord (ALSA) intermittently produces 0-590ms of full-scale clipping
(RMS >= 30000) at recording start due to device initialization.
pw-record (PipeWire native) eliminates this except on the very first
recording when the audio source wakes from SUSPENDED state.

The test does a warmup recording first (waking the device), then verifies
5 subsequent trials have zero clipping.

Requires: pw-record available, PipeWire running, audio input device.
"""

import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pytest
import soundfile as sf


CLIPPING_THRESHOLD = 30000  # RMS threshold for clipping detection
CHUNK_MS = 10  # Analysis window size in ms
CHECK_FIRST_MS = 200  # Check first 200ms for clipping
RECORD_DURATION_S = 1.0  # Record 1 second per trial
NUM_TRIALS = 5


@pytest.fixture(autouse=True)
def require_pw_record():
    if not shutil.which("pw-record"):
        pytest.skip("pw-record not available")


def _record_and_analyze(duration_s: float) -> dict:
    """Record audio with pw-record and analyze startup clipping."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    try:
        proc = subprocess.Popen(
            ["pw-record", "--format=s16", "--rate=16000", "--channels=1", tmp.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(duration_s)
        proc.terminate()
        proc.wait()
        time.sleep(0.1)

        data, sr = sf.read(tmp.name, dtype="int16")
        chunk_size = int(CHUNK_MS * sr / 1000)
        check_chunks = int(CHECK_FIRST_MS / CHUNK_MS)
        total_chunks = min(check_chunks, len(data) // chunk_size)

        clipping_chunks = 0
        max_rms = 0.0
        for i in range(total_chunks):
            chunk = data[i * chunk_size : (i + 1) * chunk_size]
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            max_rms = max(max_rms, rms)
            if rms >= CLIPPING_THRESHOLD:
                clipping_chunks += 1

        duration_ms = len(data) * 1000 / sr
        return {
            "duration_ms": duration_ms,
            "clipping_chunks": clipping_chunks,
            "max_rms": max_rms,
        }
    finally:
        os.unlink(tmp.name)


class TestPwRecordNoClipping:
    """Verify pw-record produces no startup clipping after device warmup."""

    def test_no_clipping_in_first_200ms(self):
        """Warmup + 5 trials: no chunk in first 200ms should have RMS >= 30000.

        The first recording after PipeWire source wakes from SUSPENDED may
        have clipping. A warmup trial wakes the device; subsequent trials
        must be clean.
        """
        # Warmup: wake the PipeWire source from SUSPENDED state
        warmup = _record_and_analyze(0.5)
        print(
            f"  Warmup: {warmup['duration_ms']:.0f}ms, "
            f"clip_chunks={warmup['clipping_chunks']} (not counted)"
        )

        results = []
        for trial in range(NUM_TRIALS):
            result = _record_and_analyze(RECORD_DURATION_S)
            results.append(result)
            print(
                f"  Trial {trial + 1}: {result['duration_ms']:.0f}ms, "
                f"clip_chunks={result['clipping_chunks']}, "
                f"max_rms={result['max_rms']:.0f}"
            )

        total_clipping = sum(r["clipping_chunks"] for r in results)
        assert total_clipping == 0, (
            f"{total_clipping} clipping chunks found across {NUM_TRIALS} trials "
            f"(threshold={CLIPPING_THRESHOLD})"
        )
