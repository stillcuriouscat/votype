#!/usr/bin/env python3
"""E2E test for voice-input → Kitty terminal pipeline.

Tests the full flow: daemon health → voice recording → ASR transcription → text
appears in Kitty scrollback. Monitors system audio via parecord to verify the
recording pipeline captured sound.

Requirements:
    - voice-input daemon running (`voice-input daemon --start`)
    - Kitty terminal open with remote control enabled
    - PulseAudio/PipeWire with parecord available
    - Background noise or speech during recording

Usage:
    python test_e2e_kitty.py --verbose
    python test_e2e_kitty.py --duration 10
"""

import argparse
import glob
import json
import math
import socket
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_DIR = Path.home() / ".config" / "voice-input"
DAEMON_PID_FILE = CONFIG_DIR / "daemon.pid"
SOCKET_PATH = CONFIG_DIR / "daemon.sock"
KITTY_SOCKET_GLOB = "/tmp/kitty-socket*"
MONITOR_WAV = Path("/tmp/test-e2e-monitor.wav")
TEST_AUDIO = Path(__file__).parent / "local" / "demo.wav"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

results = []
verbose = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    """Print an indented log line (only in verbose mode)."""
    if verbose:
        print(f"    {msg}")


def record_result(name, passed, detail=""):
    """Record and display a test result."""
    entry = {"name": name, "passed": passed, "detail": detail}
    results.append(entry)
    tag = "\033[32m[PASS]\033[0m" if passed else "\033[31m[FAIL]\033[0m"
    suffix = f" — {detail}" if detail else ""
    print(f"  {tag} {name}{suffix}")


# ---------------------------------------------------------------------------
# Daemon communication
# ---------------------------------------------------------------------------

def send_to_daemon(command, data=None, timeout=10):
    """Send a JSON command to the voice-input daemon via unix socket.

    Returns the parsed JSON response dict, or None on error.
    """
    if not SOCKET_PATH.exists():
        log(f"Socket not found: {SOCKET_PATH}")
        return None

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(timeout)
        client.connect(str(SOCKET_PATH))

        payload = {"command": command}
        if data is not None:
            payload["data"] = data
        client.sendall(json.dumps(payload).encode())

        response_data = client.recv(65536)
        return json.loads(response_data.decode())
    except (socket.error, json.JSONDecodeError, OSError) as exc:
        log(f"Daemon communication error: {exc}")
        return None
    finally:
        client.close()


def check_daemon():
    """Ping the daemon and record the result."""
    response = send_to_daemon("ping")
    if response and response.get("status") == "ok":
        model = response.get("model", "unknown")
        record_result("Daemon ping", True, f"model={model}")
        return True
    record_result("Daemon ping", False, "No response or bad status")
    return False


# ---------------------------------------------------------------------------
# Kitty helpers
# ---------------------------------------------------------------------------

def find_kitty_socket():
    """Find the first available Kitty remote-control socket.

    Returns the socket path string, or None.
    """
    sockets = sorted(glob.glob(KITTY_SOCKET_GLOB))
    if sockets:
        chosen = sockets[0]
        record_result("Kitty socket", True, chosen)
        return chosen
    record_result("Kitty socket", False, "No /tmp/kitty-socket* found")
    return None


def get_kitty_text(kitty_socket):
    """Capture current Kitty scrollback text.

    Returns the captured text string, or empty string on failure.
    """
    try:
        result = subprocess.run(
            ["kitty", "@", "--to", f"unix:{kitty_socket}",
             "get-text", "--extent", "all"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        log(f"kitty get-text failed: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Audio monitor (parecord + RMS)
# ---------------------------------------------------------------------------

def get_monitor_source():
    """Detect a PulseAudio/PipeWire monitor source.

    Returns the source name string, or None.
    """
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "monitor" in line.lower():
                source = line.split()[1]
                record_result("Monitor source", True, source)
                return source
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        log(f"pactl failed: {exc}")

    record_result("Monitor source", False, "No monitor source found")
    return None


def start_monitor(source):
    """Start recording system audio with parecord.

    Returns the Popen process.
    """
    # Remove stale file
    if MONITOR_WAV.exists():
        MONITOR_WAV.unlink()

    proc = subprocess.Popen(
        [
            "parecord",
            "--device", source,
            "--format=s16le",
            "--rate=16000",
            "--channels=1",
            "--file-format=wav",
            str(MONITOR_WAV),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log(f"Monitor recording started (PID {proc.pid})")
    return proc


def stop_monitor(proc):
    """Stop parecord and return the wav path if it exists."""
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)

    if MONITOR_WAV.exists() and MONITOR_WAV.stat().st_size > 44:
        log(f"Monitor file: {MONITOR_WAV} ({MONITOR_WAV.stat().st_size} bytes)")
        return MONITOR_WAV
    log("Monitor wav missing or empty")
    return None


def compute_rms(wav_path):
    """Compute RMS of a 16-bit mono WAV file.

    Returns the RMS value as a float.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                return 0.0
            raw = wf.readframes(n_frames)

        n_samples = len(raw) // 2
        if n_samples == 0:
            return 0.0

        samples = struct.unpack(f"<{n_samples}h", raw)
        sum_sq = sum(s * s for s in samples)
        return math.sqrt(sum_sq / n_samples)
    except (wave.Error, struct.error, OSError) as exc:
        log(f"RMS computation error: {exc}")
        return 0.0


# ---------------------------------------------------------------------------
# Main test flow
# ---------------------------------------------------------------------------

def run_test(duration=8):
    """Execute the full E2E test sequence."""
    print("\n=== E2E Kitty Voice Input Test ===\n")

    # -- Pre-checks ----------------------------------------------------------
    print("[Pre-checks]")
    daemon_ok = check_daemon()
    kitty_socket = find_kitty_socket()
    monitor_source = get_monitor_source()

    if not daemon_ok:
        print("\nAborting: daemon not reachable.")
        return False
    if not kitty_socket:
        print("\nAborting: no Kitty socket found.")
        return False
    if not monitor_source:
        print("\nAborting: no monitor source found.")
        return False

    # -- Baseline capture ----------------------------------------------------
    print("\n[Baseline]")
    baseline = get_kitty_text(kitty_socket)
    baseline_len = len(baseline)
    log(f"Baseline length: {baseline_len} chars")
    print(f"  Captured baseline ({baseline_len} chars)")

    # -- Start monitor -------------------------------------------------------
    print("\n[Recording]")
    monitor_proc = start_monitor(monitor_source)

    try:
        # -- Toggle: start recording -----------------------------------------
        print(f"  Starting voice-input recording...")
        try:
            subprocess.run(
                ["voice-input", "toggle"],
                timeout=30,
                capture_output=True,
            )
            log("voice-input toggle (start) returned")
        except subprocess.TimeoutExpired:
            record_result("Toggle start", False, "Timed out (30s)")
            return False
        except FileNotFoundError:
            record_result("Toggle start", False, "voice-input command not found")
            return False

        record_result("Toggle start", True)

        # -- Play test audio so microphone picks up sound --------------------
        audio_proc = None
        if TEST_AUDIO.exists():
            log(f"Playing test audio: {TEST_AUDIO}")
            audio_proc = subprocess.Popen(
                ["paplay", str(TEST_AUDIO)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # -- Wait for recording duration -------------------------------------
        print(f"  Recording for {duration}s...")
        time.sleep(duration)

        # -- Stop test audio playback ----------------------------------------
        if audio_proc is not None:
            audio_proc.terminate()
            try:
                audio_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                audio_proc.kill()
                audio_proc.wait(timeout=3)

        # -- Toggle: stop + transcribe (synchronous, blocks) -----------------
        print(f"  Stopping voice-input (transcribing, may take a while)...")
        try:
            subprocess.run(
                ["voice-input", "toggle"],
                timeout=120,
                capture_output=True,
            )
            log("voice-input toggle (stop+transcribe) returned")
        except subprocess.TimeoutExpired:
            record_result("Toggle stop", False, "Timed out (120s)")
            return False
        except FileNotFoundError:
            record_result("Toggle stop", False, "voice-input command not found")
            return False

        record_result("Toggle stop", True)
    finally:
        # -- Stop monitor (always, even on failure/interrupt) ----------------
        print("\n[Audio verification]")
        wav_path = stop_monitor(monitor_proc)
    if wav_path:
        rms = compute_rms(wav_path)
        rms_pass = rms > 100
        record_result("Audio RMS", rms_pass,
                       f"RMS={rms:.1f} (threshold=100)"
                       + ("" if rms_pass else " [non-critical]"))
    else:
        record_result("Audio RMS", False, "No monitor recording captured [non-critical]")

    # -- Check Kitty scrollback for new content ------------------------------
    print("\n[Output verification]")
    # Small delay for any pending output
    time.sleep(0.5)
    final_text = get_kitty_text(kitty_socket)
    final_len = len(final_text)
    new_content = final_text[baseline_len:].strip() if final_len > baseline_len else ""
    has_new_text = len(new_content) > 0

    if has_new_text:
        preview = new_content[:120].replace("\n", " ")
        record_result("New text in Kitty", True, f"'{preview}'")
    else:
        record_result("New text in Kitty", False,
                       f"No new content (baseline={baseline_len}, final={final_len})")

    # -- Summary -------------------------------------------------------------
    print("\n=== Summary ===")
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    # Audio RMS is non-critical (environment-dependent)
    critical_failures = [
        r for r in results
        if not r["passed"] and "[non-critical]" not in r.get("detail", "")
    ]
    print(f"  {passed}/{total} checks passed")
    if critical_failures:
        print(f"  Critical failures: {[r['name'] for r in critical_failures]}")
    print()

    return len(critical_failures) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global verbose

    parser = argparse.ArgumentParser(
        description="E2E test for voice-input → Kitty terminal pipeline",
    )
    parser.add_argument(
        "--duration", type=int, default=8,
        help="Recording duration in seconds (default: 8)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed log output",
    )
    args = parser.parse_args()
    verbose = args.verbose

    success = run_test(duration=args.duration)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
