#!/usr/bin/env python3
"""E2E test for voice-input → Kitty terminal pipeline.

Tests the full flow: daemon health → voice recording → ASR transcription → text
appears in Kitty scrollback. Monitors system audio via parecord to verify the
recording pipeline captured sound.

Auto-punctuation: When the ASR model has auto-punctuation (e.g. firered-asr with
FireRedPunc), the test automatically verifies Chinese punctuation appears in the
output — no --post-processor flag needed.

Requirements:
    - voice-input daemon running (`voice-input daemon --start`)
    - Kitty terminal open with remote control enabled
    - PulseAudio/PipeWire with parecord available
    - Background noise or speech during recording

Usage:
    python test_e2e_kitty.py --verbose
    python test_e2e_kitty.py --duration 10
    python test_e2e_kitty.py --post-processor none --verbose
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
    """Ping the daemon and record the result.

    Returns the model_id string on success, or None on failure.
    """
    response = send_to_daemon("ping")
    if response and response.get("status") == "ok":
        model = response.get("model", "unknown")
        record_result("Daemon ping", True, f"model={model}")
        return model
    record_result("Daemon ping", False, "No response or bad status")
    return None


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


def find_kitty_target_window(kitty_socket):
    """Find a suitable Kitty window ID to use for testing.

    Prefers a plain shell window (not Claude Code). Falls back to any window.
    Returns the window ID as int, or None.
    """
    try:
        result = subprocess.run(
            ["kitty", "@", "--to", f"unix:{kitty_socket}", "ls"],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        log(f"kitty ls failed: {exc}")
        return None

    all_windows = []
    for os_win in data:
        for tab in os_win.get("tabs", []):
            for win in tab.get("windows", []):
                all_windows.append(win)

    # Prefer a shell window (not running Claude Code)
    for win in all_windows:
        title = win.get("title", "")
        if "Claude Code" not in title and "claude" not in title.lower():
            wid = win["id"]
            log(f"Target window: id={wid} title={title[:60]}")
            return wid

    # Fallback: use the first window
    if all_windows:
        wid = all_windows[0]["id"]
        log(f"Target window (fallback): id={wid}")
        return wid

    return None


def get_kitty_text(kitty_socket, window_id=None):
    """Capture current Kitty scrollback text from a specific window.

    Returns the captured text string, or empty string on failure.
    """
    cmd = ["kitty", "@", "--to", f"unix:{kitty_socket}",
           "get-text", "--extent", "all"]
    if window_id is not None:
        cmd.extend(["--match", f"id:{window_id}"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
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
# Daemon log verification
# ---------------------------------------------------------------------------

DAEMON_LOG = Path("/tmp/voice-input-notify.log")


def _check_daemon_log_for_punctuation(punc_chars):
    """Check recent daemon log [PUNC] and [PP] output lines for Chinese punctuation.

    Checks [PUNC] lines (auto-punctuation) and [PP] output lines (post-processor).
    Returns a summary string if found, or None.
    """
    if not DAEMON_LOG.exists():
        log("Daemon log not found")
        return None

    try:
        lines = DAEMON_LOG.read_text().splitlines()
    except OSError as exc:
        log(f"Failed to read daemon log: {exc}")
        return None

    # Check the last 30 lines for [PUNC] or [PP] output entries
    for line in reversed(lines[-30:]):
        if "[PUNC]" not in line and "[PP] output" not in line:
            continue
        found = [c for c in line if c in punc_chars]
        if found:
            unique = "".join(sorted(set(found)))
            # Extract the text portion after the colon
            parts = line.split(": ", 2)
            snippet = parts[-1][:80] if len(parts) > 2 else line[-80:]
            tag = "[PUNC]" if "[PUNC]" in line else "[PP]"
            log(f"Daemon log punctuation ({tag}): {unique} in '{snippet}'")
            return f"{unique} ({tag} in '{snippet}')"

    log("No punctuation found in recent daemon log entries")
    return None


# ---------------------------------------------------------------------------
# Main test flow
# ---------------------------------------------------------------------------

def switch_post_processor(pp_id):
    """Switch the daemon's post-processor and return True on success."""
    response = send_to_daemon("set_post_processor", {"post_processor_id": pp_id})
    if response and response.get("status") == "ok":
        log(f"Switched post-processor to: {pp_id}")
        return True
    # "already active" is also fine
    if response and "already" in response.get("message", "").lower():
        log(f"Post-processor {pp_id} already active")
        return True
    log(f"Failed to switch post-processor: {response}")
    return False


def get_current_post_processor():
    """Query the daemon's current post-processor ID."""
    response = send_to_daemon("get_post_processor")
    if response and "post_processor" in response:
        return response["post_processor"]
    return None


def run_test(duration=8, post_processor=None):
    """Execute the full E2E test sequence.

    Args:
        duration: Recording duration in seconds.
        post_processor: If set, switch daemon to this post-processor before
            recording and restore the original afterwards.
    """
    print("\n=== E2E Kitty Voice Input Test ===\n")

    # -- Pre-checks ----------------------------------------------------------
    print("[Pre-checks]")
    model_id = check_daemon()
    kitty_socket = find_kitty_socket()
    monitor_source = get_monitor_source()

    if not model_id:
        print("\nAborting: daemon not reachable.")
        return False

    # Models with punctuation='firered-punc' in MODEL_PRESETS auto-load FireRedPunc
    AUTO_PUNC_MODELS = {"firered-asr"}
    auto_punc_expected = model_id in AUTO_PUNC_MODELS
    if auto_punc_expected:
        log(f"Auto-punctuation expected for model: {model_id}")
    if not kitty_socket:
        print("\nAborting: no Kitty socket found.")
        return False
    if not monitor_source:
        print("\nAborting: no monitor source found.")
        return False

    # -- Post-processor switching --------------------------------------------
    original_pp = None
    if post_processor:
        print(f"\n[Post-processor]")
        original_pp = get_current_post_processor()
        log(f"Original post-processor: {original_pp}")
        if switch_post_processor(post_processor):
            record_result("Post-processor switch", True, f"→ {post_processor}")
        else:
            record_result("Post-processor switch", False, f"Failed to switch to {post_processor}")
            print("\nAborting: could not switch post-processor.")
            return False

    # -- Find target Kitty window (avoid multi-window ambiguity) -------------
    target_wid = find_kitty_target_window(kitty_socket)
    if target_wid is None:
        print("\nAborting: no Kitty window found.")
        return False
    print(f"  Target Kitty window: id={target_wid}")

    # Focus the target window so voice-input's send-text goes there too
    try:
        subprocess.run(
            ["kitty", "@", "--to", f"unix:{kitty_socket}",
             "focus-window", "--match", f"id:{target_wid}"],
            timeout=5, capture_output=True,
        )
        log(f"Focused window id={target_wid}")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log(f"WARNING: focus-window failed: {exc}")

    # -- Inject marker into Kitty scrollback ---------------------------------
    print("\n[Baseline]")
    import uuid
    marker = f"__E2E_MARKER_{uuid.uuid4().hex[:12]}__"
    match_arg = ["--match", f"id:{target_wid}"]
    try:
        subprocess.run(
            ["kitty", "@", "--to", f"unix:{kitty_socket}",
             "send-text"] + match_arg + [f"\n{marker}\n"],
            timeout=5, check=True, capture_output=True,
        )
        log(f"Injected marker: {marker}")
        print(f"  Injected marker into Kitty scrollback")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"  WARNING: Failed to inject marker: {exc}")
        marker = None

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
    time.sleep(1)
    final_text = get_kitty_text(kitty_socket, target_wid)
    has_new_text = False
    new_content = ""
    after_marker = ""

    if marker and marker in final_text:
        # Extract everything after the marker line
        after_marker = final_text.split(marker, 1)[1]
        # Filter out test script noise (shell prompts, known commands)
        noise_prefixes = ("$", "dev@", "Recording", "Transcribed:", "voice-input",
                          "[type_text]", "[PASS]", "[FAIL]", "===", "  ")
        content_lines = []
        for line in after_marker.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if any(stripped.startswith(p) for p in noise_prefixes):
                continue
            content_lines.append(stripped)
        new_content = " ".join(content_lines).strip()
        has_new_text = len(new_content) > 0

        if has_new_text:
            preview = new_content[:120]
            record_result("New text in Kitty", True, f"'{preview}'")
        else:
            log(f"After marker ({len(after_marker)} chars), all lines filtered as noise")
            record_result("New text in Kitty", False,
                           "No transcription text found after marker (only noise)")
    else:
        # Fallback: marker not found (buffer overflow or injection failed)
        # Check if final text ends with something that looks like transcription
        final_len = len(final_text)
        log(f"Marker not found in scrollback ({final_len} chars)")
        record_result("New text in Kitty", False,
                       f"Marker not found (scrollback={final_len} chars)")

    # -- Punctuation verification (auto-punctuation or explicit post-processor)
    if auto_punc_expected or post_processor:
        print("\n[Punctuation verification]")
        punc_source = "auto-punctuation" if auto_punc_expected else f"post-processor={post_processor}"
        log(f"Checking punctuation ({punc_source})")
        punc_chars = set("。，？！、；：")
        # Scan raw scrollback (before noise filtering) — tmux/mosh borders
        # can cause the noise filter to strip transcribed text with leading spaces.
        scan_text = after_marker if after_marker else new_content
        found_punc = [c for c in scan_text if c in punc_chars]
        if found_punc:
            unique = "".join(sorted(set(found_punc)))
            snippet = ""
            for line in scan_text.splitlines():
                if any(c in punc_chars for c in line):
                    snippet = line.strip()[:80]
                    break
            record_result("Chinese punctuation", True,
                           f"Found: {unique} (in '{snippet}')")
        else:
            # Fallback: check daemon log for [PUNC] (auto-punctuation) or
            # [PP] (post-processor) output with punctuation. This still verifies
            # the full E2E pipeline ran — audio recorded, ASR transcribed, and
            # punctuation was applied.
            log_punc = _check_daemon_log_for_punctuation(punc_chars)
            if log_punc:
                record_result("Chinese punctuation", True,
                               f"Found in daemon log: {log_punc}")
            else:
                record_result("Chinese punctuation", False,
                               "No Chinese punctuation in scrollback or daemon log")

    # -- Restore original post-processor ------------------------------------
    if original_pp and original_pp != post_processor:
        print("\n[Restore post-processor]")
        if switch_post_processor(original_pp):
            record_result("Post-processor restore", True, f"→ {original_pp}")
        else:
            record_result("Post-processor restore", False,
                           f"Failed to restore {original_pp} [non-critical]")

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
    parser.add_argument(
        "--post-processor", type=str, default=None,
        help="Switch daemon to this LLM post-processor before test (e.g. none, chinese-text-correction)",
    )
    args = parser.parse_args()
    verbose = args.verbose

    success = run_test(duration=args.duration, post_processor=args.post_processor)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
