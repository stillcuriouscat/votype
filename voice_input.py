#!/usr/bin/env python3
"""
Global Voice Input Tool - Multi-ASR-framework support
Supports FunASR, Transformers, FireRedASR and more
Punctuation is built-in for default model (SenseVoice); firered-asr uses FireRedPunc.

Usage:
    voice-input start    # Start recording
    voice-input stop     # Stop and transcribe
    voice-input toggle   # Toggle state (recommended: bind to hotkey)
    voice-input daemon   # Start background service (model stays in memory)
    voice-input kill     # Stop background service
    voice-input status   # Show current status and model
    voice-input models   # List available models
    voice-input model <name>  # Switch model (fun-asr-nano/paraformer/sensevoice/firered-asr)
    voice-input post-processors    # List available post-processors (LLM text refinement)
    voice-input post-processor <id>  # Switch LLM post-processor (none/chinese-text-correction/qwen3-0.6b/minicpm4-0.5b)
"""

import sys
import os
import fcntl
import subprocess
import signal
import socket
import json
import shutil
import threading
import time
import logging
from pathlib import Path

# Import model configs
from model_configs import MODEL_PRESETS, DEFAULT_MODEL, DEVICE, HOTWORDS, ModelLoader, ModelInference
from post_processor_presets import POST_PROCESSOR_PRESETS, DEFAULT_POST_PROCESSOR
from post_processor_configs import PostProcessorLoader, PostProcessorInference
from state_db import init_db, get_state, update_state

# AppIndicator (system tray icon)
try:
    import gi
    gi.require_version('Gtk', '3.0')
    gi.require_version('AyatanaAppIndicator3', '0.1')
    from gi.repository import Gtk, AyatanaAppIndicator3, GLib
    HAS_INDICATOR = True
except (ImportError, ValueError):
    HAS_INDICATOR = False

# Suppress unnecessary logs
logging.getLogger("modelscope").setLevel(logging.WARNING)
logging.getLogger("funasr").setLevel(logging.WARNING)
logging.getLogger("jieba").setLevel(logging.WARNING)

# Configuration
CONFIG_DIR = Path.home() / ".config" / "voice-input"
PID_FILE = CONFIG_DIR / "recording.pid"
AUDIO_FILE = CONFIG_DIR / "recording.wav"  # Legacy, kept for compatibility
AUDIO_PATH_FILE = CONFIG_DIR / "recording_path.txt"  # Stores current recording file path
DAEMON_PID_FILE = CONFIG_DIR / "daemon.pid"
DAEMON_LOCK_FILE = CONFIG_DIR / "daemon.lock"
SOCKET_PATH = CONFIG_DIR / "daemon.sock"
PROCESSING_FILE = CONFIG_DIR / "processing.flag"

# Recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1

# Recorder selection: prefer pw-record (PipeWire native, no startup clipping)
# Fall back to arecord (ALSA) if pw-record is unavailable
_RECORDER = "pw-record" if shutil.which("pw-record") else "arecord"

# Current model state file
MODEL_STATE_FILE = CONFIG_DIR / "current_model.txt"
POST_PROCESSOR_STATE_FILE = CONFIG_DIR / "current_post_processor.txt"

# SQLite state database path (mirrors state_db.DEFAULT_DB_PATH)
STATE_DB_PATH = Path.home() / ".config" / "voice-input" / "state.db"


def ensure_config_dir():
    """Ensure config directory exists and initialize state DB."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    init_db(STATE_DB_PATH)
    _log("INIT", f"config dir ensured: {CONFIG_DIR}")


def _cleanup_old_recordings(max_age_hours=2):
    """Delete recording WAV files older than max_age_hours."""
    cutoff = time.time() - max_age_hours * 3600
    for f in CONFIG_DIR.glob("recording_*.wav"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                _log("CLEANUP", f"deleted stale recording: {f.name}")
        except OSError:
            pass


def get_current_model():
    """Get the currently selected model - always returns the default model."""
    return DEFAULT_MODEL


def set_current_model(model_id):
    """Set the current model - no longer persisted to file."""
    # Single-model architecture: do not remember model selection
    pass


NOTIFY_LOG_FILE = Path("/tmp/voice-input-notify.log")


def _log(tag, message):
    """Write a structured log line to the notify log file."""
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(NOTIFY_LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] [{tag}] {message}\n")
    except Exception:
        pass


def notify(title, message, urgency="normal"):
    """Send a desktop notification and write to the log file."""
    # Always log to file so we can debug notification issues
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_level = "ERROR" if urgency == "critical" else "WARN" if urgency == "low" else "INFO"
        log_line = f"[{timestamp}] [{log_level}] {title}: {message}\n"
        with open(NOTIFY_LOG_FILE, "a") as f:
            f.write(log_line)
    except Exception:
        pass

    try:
        subprocess.Popen(
            ["notify-send", "-u", urgency, "-t", "2000", title, message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        pass


def is_process_running(pid_file):
    """Check whether the process referenced by a PID file is running."""
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        pid_file.unlink(missing_ok=True)
        _log("PID", f"cleaned stale PID file: {pid_file.name}")
        return False


def is_recording():
    """Check whether recording is in progress via DB status + PID liveness."""
    state = get_state(STATE_DB_PATH)
    if state["status"] != "recording":
        return False
    pid = state["recording_pid"]
    if pid is None:
        update_state(STATE_DB_PATH, status="idle", recording_pid=None, recording_path=None)
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        update_state(STATE_DB_PATH, status="idle", recording_pid=None, recording_path=None)
        return False


def _cleanup_daemon_files():
    """Clean up stale daemon PID and socket files."""
    DAEMON_PID_FILE.unlink(missing_ok=True)
    SOCKET_PATH.unlink(missing_ok=True)
    _log("DAEMON", "cleaned up daemon files (PID, socket)")


def _is_daemon_lock_held():
    """Check if the daemon lock file is held by another process.

    Uses fcntl.flock to probe the lock. If we can't acquire it,
    a daemon is running (even if PID file hasn't been written yet).
    """
    if not DAEMON_LOCK_FILE.exists():
        return False
    try:
        fd = open(DAEMON_LOCK_FILE, 'r')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Got the lock — no daemon holding it
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except (IOError, OSError):
            return True  # Lock held by another process
        finally:
            fd.close()
    except (IOError, OSError):
        return False


def is_daemon_running():
    """Check whether the daemon is running.

    Tries DB daemon_pid first, falls back to flock + PID file for
    backward compatibility during transition.
    """
    # Fast path: check flock (covers startup race before PID/DB is written)
    if _is_daemon_lock_held():
        return True

    # Try DB first: daemon writes daemon_pid on startup
    state = get_state(STATE_DB_PATH)
    db_pid = state.get("daemon_pid")
    if db_pid is not None:
        try:
            os.kill(db_pid, 0)
            # Verify this is actually our daemon
            cmdline_path = Path(f"/proc/{db_pid}/cmdline")
            if cmdline_path.exists():
                cmdline = cmdline_path.read_text()
                if "voice_input" not in cmdline and "voice-input" not in cmdline:
                    update_state(STATE_DB_PATH, daemon_pid=None)
                    _cleanup_daemon_files()
                    return False
            return True
        except ProcessLookupError:
            # Dead PID in DB — clean up
            update_state(STATE_DB_PATH, daemon_pid=None)
            _cleanup_daemon_files()
            return False

    # Fallback: PID file (backward compat — daemon may not have written to DB yet)
    if not DAEMON_PID_FILE.exists():
        return False
    try:
        pid = int(DAEMON_PID_FILE.read_text().strip())
        os.kill(pid, 0)
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        if cmdline_path.exists():
            cmdline = cmdline_path.read_text()
            if "voice_input" not in cmdline and "voice-input" not in cmdline:
                _cleanup_daemon_files()
                return False
        return True
    except (ProcessLookupError, ValueError):
        _cleanup_daemon_files()
        return False


def is_daemon_ready():
    """Check whether the daemon is truly ready (can respond to ping)."""
    if not is_daemon_running():
        return False
    response = send_to_daemon("ping")
    return response and response.get("status") == "ok"


def get_daemon_paths():
    """Get the Python interpreter and script path for the daemon."""
    venv_python = Path.home() / ".local" / "share" / "voice-input" / "venv" / "bin" / "python"
    script_path = Path.home() / ".local" / "share" / "voice-input" / "voice_input.py"

    if not venv_python.exists():
        venv_python = Path(sys.executable)
    if not script_path.exists():
        script_path = Path(__file__).resolve()

    return venv_python, script_path


def send_to_daemon(command, data=None, timeout=60):
    """Send a command to the daemon."""
    if not SOCKET_PATH.exists():
        return None

    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        client.connect(str(SOCKET_PATH))
        _log("IPC", f"connected to daemon socket")

        msg = json.dumps({"command": command, "data": data})
        client.send(msg.encode())

        response = client.recv(65536).decode()
        client.close()

        return json.loads(response)
    except Exception as e:
        return {"error": str(e)}


def start_recording():
    """Start recording with a timestamped filename."""
    from datetime import datetime
    ensure_config_dir()

    if is_recording():
        # Abnormal: toggle should have called stop_recording, should not reach here
        notify("⚠️ Voice Input", "Abnormal state: already recording\nPress the hotkey again to stop recording", "critical")
        return

    # Generate timestamped filename for this recording
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = CONFIG_DIR / f"recording_{ts}.wav"

    # Use Popen to start recording (better process management)
    try:
        if _RECORDER == "pw-record":
            cmd = [
                "pw-record",
                f"--format=s16",
                f"--rate={SAMPLE_RATE}",
                f"--channels={CHANNELS}",
                str(audio_file)
            ]
        else:
            cmd = [
                "arecord",
                "-f", "S16_LE",
                "-r", str(SAMPLE_RATE),
                "-c", str(CHANNELS),
                "-t", "wav",
                str(audio_file)
            ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # Create new session to decouple from parent process
        )
        _log("PROC", f"recorder spawned: {_RECORDER} (PID {proc.pid})")

        update_state(STATE_DB_PATH, status="recording", recording_pid=proc.pid, recording_path=str(audio_file))
        _log("PROC", f"recording state saved: PID={proc.pid}, path={audio_file.name}")
        print(f"Recording started (PID: {proc.pid}, recorder: {_RECORDER}, file: {audio_file.name})")
    except (FileNotFoundError, OSError) as e:
        _log("ERROR", f"start_recording failed: {e}")
        update_state(STATE_DB_PATH, status="idle", recording_pid=None, recording_path=None)
        notify("❌ Voice Input", f"Failed to start recording: {e}", "critical")


def stop_recording():
    """Stop recording and transcribe. Recording files are kept for 2 hours."""
    ensure_config_dir()

    if not is_recording():
        # Abnormal: toggle should have called start_recording, should not reach here
        notify("⚠️ Voice Input", "Abnormal state: no recording in progress", "critical")
        return

    # CRITIC-R2-C1: Read recording_pid AND recording_path from DB BEFORE the kill sequence
    state = get_state(STATE_DB_PATH)
    rec_pid = state["recording_pid"]
    rec_path = state["recording_path"]

    # Mark processing state immediately — prevents concurrent toggle from starting new recording
    update_state(STATE_DB_PATH, status="processing")
    _log("STATE", "status set to processing in DB")
    _log_to_notify_file("processing started (recording stopped, transcription pending)")

    # Stop recording
    if rec_pid is not None:
        try:
            os.kill(rec_pid, signal.SIGTERM)
            _log("PROC", f"SIGTERM sent to recorder PID {rec_pid}")
            time.sleep(0.3)
        except (ProcessLookupError, ValueError) as e:
            _log("ERROR", f"kill recorder failed: {e}")

    # Clear recording fields in DB
    update_state(STATE_DB_PATH, recording_pid=None, recording_path=None)

    # Determine the audio file path from DB (with legacy fallback)
    audio_file = None
    if rec_path:
        audio_file = Path(rec_path)
    if audio_file is None or not audio_file.exists():
        # Fallback to legacy path
        audio_file = AUDIO_FILE

    daemon_running = is_daemon_running()

    if not audio_file.exists():
        update_state(STATE_DB_PATH, status="idle")
        _log("STATE", "status set to idle (no audio)")
        notify("❌ Voice Input", "Recording file not found", "critical")
        return

    # If the daemon is running, use it for transcription
    if daemon_running:
        response = send_to_daemon("transcribe", str(audio_file))
        _log("IPC", "sent transcribe request to daemon")
        if response and "text" in response:
            text = response["text"]
            if text:
                type_text(text)
                print(f"Transcribed: {text}")
        else:
            error = response.get("error", "Unknown error") if response else "Daemon not responding"
            notify("❌ Voice Input", f"Transcription failed: {error}", "critical")
            # On failure, the current recording is preserved for recovery
    else:
        # No daemon is an abnormal situation (normally toggle starts the daemon first)
        notify("❌ Voice Input", "Service error\nRun voice-input daemon to start\nor check /tmp/voice-input-daemon.log", "critical")

    # Set idle AFTER text has been pasted (icon stays orange during entire pipeline)
    update_state(STATE_DB_PATH, status="idle")
    _log("STATE", "status set to idle")
    _log_to_notify_file("processing complete")

    # Always clean up old recordings (>2h), regardless of transcription result
    _cleanup_old_recordings()


def transcribe_audio_direct(audio_path):
    """Directly load and transcribe with model (used when no daemon is running)."""
    model_id = get_current_model()
    model, framework, extra_data = ModelLoader.load_model(model_id, DEVICE)
    text = ModelInference.transcribe(model, audio_path, model_id, framework, extra_data, HOTWORDS)
    return text


def clipboard_get():
    """Get clipboard contents."""
    result = subprocess.run(
        ["xclip", "-selection", "clipboard", "-o"],
        capture_output=True, text=True
    )
    return result.stdout


def clipboard_set(text):
    """Set clipboard contents."""
    proc = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
    proc.communicate(input=text.encode('utf-8'))


def is_terminal_window():
    """Detect whether the active window is a terminal."""
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True,
            text=True,
            check=True
        )
        window_name = result.stdout.strip().lower()

        # Common terminal names
        terminals = [
            "terminal",
            "gnome-terminal",
            "konsole",
            "xterm",
            "alacritty",
            "tilix",
            "terminator",
            "kitty",
            "urxvt",
            "rxvt",
            "xfce4-terminal",
            "mate-terminal",
        ]

        return any(term in window_name for term in terminals)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # If detection fails, assume it is not a terminal (use clipboard method)
        return False


KITTY_SOCKET_GLOB = "/tmp/kitty-socket*"


def _get_kitty_socket():
    """Return kitty socket path if available, else None.

    Kitty appends PID to socket path (e.g. /tmp/kitty-socket-12345).
    """
    import glob
    sockets = sorted(glob.glob(KITTY_SOCKET_GLOB))
    return sockets[0] if sockets else None


def _is_kitty_window():
    """Check if the active window belongs to Kitty terminal.

    Uses WM_CLASS (stable) instead of window title (changes with running program).
    """
    try:
        wid = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        wm_class = subprocess.run(
            ["xprop", "-id", wid, "WM_CLASS"],
            capture_output=True, text=True, check=True
        ).stdout.strip().lower()
        return "kitty" in wm_class
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _send_paste_key(terminal):
    """Send paste keystroke via xdotool (for non-Kitty windows)."""
    if terminal:
        subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"], check=True)
    else:
        subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+v"], check=True)


def type_text(text):
    """Smart text input: Kitty uses native send-text, others use clipboard paste."""
    if not text:
        return

    try:
        # Try Kitty native send-text first (works where xdotool can't)
        kitty_socket = _get_kitty_socket()
        if kitty_socket and _is_kitty_window():
            try:
                _log("INPUT", "attempting kitty send-text")
                subprocess.run(
                    ["kitty", "@", "--to", f"unix:{kitty_socket}", "send-text", text],
                    check=True
                )
                print(f"[type_text] Used kitty send-text")
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                _log("INPUT", "kitty failed, falling back to clipboard")
                pass  # Fall through to clipboard method

        # Clipboard paste method (for non-Kitty windows or Kitty fallback)
        old_clipboard = clipboard_get()
        clipboard_set(text)
        time.sleep(0.1)

        terminal = is_terminal_window()
        _send_paste_key(terminal)
        _log("INPUT", f"paste key sent to {'terminal' if terminal else 'GUI'}")
        paste_key = "ctrl+shift+v" if terminal else "ctrl+v"
        print(f"[type_text] Used clipboard paste (xdotool: {paste_key})")

        time.sleep(0.2)
        if old_clipboard:
            clipboard_set(old_clipboard)
    except FileNotFoundError:
        print("Error: xdotool or xclip not found")
    except subprocess.CalledProcessError as e:
        print(f"Error typing text: {e}")


def _log_to_notify_file(msg):
    """Write a trace message to the notify log for debugging client-side flow."""
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(NOTIFY_LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] [TRACE] {msg}\n")
    except Exception:
        pass


def toggle_recording():
    """Toggle recording state."""
    _log_to_notify_file("toggle_recording() called")

    # Check if processing is in progress (ASR/Gemini/pasting) via DB
    state = get_state(STATE_DB_PATH)
    if state["status"] == "processing":
        updated_at = state["updated_at"]
        age = None
        if updated_at:
            try:
                from datetime import datetime, timezone
                ts = datetime.fromisoformat(updated_at)
                age = (datetime.now(timezone.utc) - ts).total_seconds()
            except (ValueError, TypeError):
                pass
        if age is not None and age < 120:
            _log_to_notify_file(f"processing in progress ({age:.1f}s), ignoring toggle")
            notify("⏳ Voice Input", "Processing in progress, please wait...")
            return
        else:
            _log_to_notify_file(f"stale processing status ({age}s), cleaning up")
            update_state(STATE_DB_PATH, status="idle")

    # Check whether the daemon is truly ready
    if not is_daemon_ready():
        daemon_running = is_daemon_running()
        _log_to_notify_file(f"daemon not ready, daemon_running={daemon_running}, pid_file={DAEMON_PID_FILE.exists()}, socket={SOCKET_PATH.exists()}")

        # Daemon not ready
        if daemon_running:
            # PID exists but ping fails = still starting up
            notify("🎙️ Voice Input", "Service is starting up, please wait...\nRecording unavailable until startup completes")
            return

        # Daemon not running, start it
        notify("🎙️ Voice Input", "Starting service, please wait...\nRecording unavailable until startup completes")

        venv_python, script_path = get_daemon_paths()
        cmd = f'nohup "{venv_python}" "{script_path}" _daemon > /tmp/voice-input-daemon.log 2>&1 &'
        subprocess.run(["bash", "-c", cmd])

        # Wait for daemon to be ready (up to 30 seconds)
        for _ in range(30):
            time.sleep(1)
            if is_daemon_ready():
                notify("✅ Voice Input", "Service started!\nPress the hotkey again to start recording")
                return

        notify("❌ Voice Input", "Service failed to start\nCheck /tmp/voice-input-daemon.log", "critical")
        return

    # Daemon is running, toggle recording state normally
    if is_recording():
        stop_recording()
    else:
        start_recording()


# ============ Daemon Mode ============

# Status icon config: {status: (icon_name, tooltip, menu_label)}
STATUS_CONFIG = {
    "idle": ("mic-idle", "Idle", "Status: Idle"),
    "recording": ("mic-recording", "Recording", "Status: 🔴 Recording..."),
    "processing": ("mic-processing", "Processing", "Status: ⏳ Processing..."),
}

def get_icons_dir():
    """Get the absolute path to the icons directory."""
    # Prefer the installed directory
    install_icons = os.path.expanduser("~/.local/share/voice-input/icons")
    if os.path.isdir(install_icons):
        return install_icons
    # Otherwise use the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "icons")


class ASRDaemon:
    """ASR daemon - loads a single model to save memory (~800MB-1GB)."""

    def __init__(self, model_id=None):
        """
        Initialize the daemon.

        Args:
            model_id: Model ID to load. If None, uses the default model.
        """
        self.model = None  # Single model instance
        self.framework = None  # Model framework type
        self.extra_data = None  # Extra data (e.g. processor)
        self.current_model_id = model_id or get_current_model()
        self.running = False
        self.indicator = None
        self.gtk_thread = None
        self.post_processor_model = None
        # Read post_processor from DB (replaces _restore_post_processor_id file read)
        db_state = get_state(STATE_DB_PATH)
        saved_pp = db_state.get("post_processor", DEFAULT_POST_PROCESSOR)
        self.current_post_processor_id = saved_pp if saved_pp in POST_PROCESSOR_PRESETS else DEFAULT_POST_PROCESSOR
        self.post_processor_framework = None
        self.punc_model = None  # Auto-punctuation model (separate from post-processor)
        self._vocab = {}  # Glossary vocab for ssh-claude (loaded in load_post_processor)
        self._secondary_model = None  # Secondary ASR model (faster-whisper for dual fusion)
        self._last_secondary_text = None  # Last secondary ASR transcription result
        self._current_db_status = "idle"  # Tracks last DB status for polling delta

    @staticmethod
    def _restore_post_processor_id():
        """Restore post-processor ID from state file, or return default."""
        try:
            saved = POST_PROCESSOR_STATE_FILE.read_text().strip()
            if saved in POST_PROCESSOR_PRESETS:
                return saved
        except (FileNotFoundError, OSError):
            pass
        return DEFAULT_POST_PROCESSOR

    @staticmethod
    def _persist_post_processor_id(preset_id):
        """Save post-processor ID to state file for daemon restart recovery."""
        try:
            POST_PROCESSOR_STATE_FILE.write_text(preset_id)
            _log("CONFIG", f"post-processor state saved: {preset_id}")
        except OSError:
            pass

    def setup_indicator(self):
        """Set up the system tray icon."""
        if not HAS_INDICATOR:
            print("Warning: AppIndicator not available, no tray icon")
            return
        
        self.indicator = AyatanaAppIndicator3.Indicator.new(
            "voice-input",
            STATUS_CONFIG["idle"][0],
            AyatanaAppIndicator3.IndicatorCategory.APPLICATION_STATUS
        )
        
        # Set custom icon directory
        icons_dir = get_icons_dir()
        print(f"[Indicator] Using icons from: {icons_dir}")
        self.indicator.set_icon_theme_path(icons_dir)
        
        self.indicator.set_status(AyatanaAppIndicator3.IndicatorStatus.ACTIVE)
        self.indicator.set_title("Voice Input")

        # Create menu
        menu = Gtk.Menu()

        # Status display
        item_status = Gtk.MenuItem(label="Status: Idle")
        item_status.set_sensitive(False)
        self.status_item = item_status
        menu.append(item_status)

        menu.append(Gtk.SeparatorMenuItem())

        # Settings
        item_settings = Gtk.MenuItem(label="Settings...")
        item_settings.connect("activate", self._on_settings_clicked)
        menu.append(item_settings)

        menu.append(Gtk.SeparatorMenuItem())

        # Quit
        item_quit = Gtk.MenuItem(label="Quit")
        item_quit.connect("activate", self.on_quit)
        menu.append(item_quit)

        menu.show_all()
        self.indicator.set_menu(menu)
    
    def on_quit(self, widget):
        """Quit menu item callback."""
        self.running = False
        Gtk.main_quit()

    def _on_settings_clicked(self, widget):
        """Settings menu item callback - supports model switching, hotword config, and log viewing."""
        try:
            from settings_dialog import show_settings_dialog, save_hotwords
            result = show_settings_dialog(
                parent=None,
                model_presets=MODEL_PRESETS,
                current_model_id=self.current_model_id,
                post_processor_presets=POST_PROCESSOR_PRESETS,
                current_post_processor_id=self.current_post_processor_id,
            )

            if not result:
                return

            # Handle post-processor switching (no restart needed)
            if result.get("pp_changed"):
                new_pp_id = result.get("new_post_processor")
                if new_pp_id and new_pp_id != self.current_post_processor_id:
                    try:
                        self.load_post_processor(new_pp_id)
                        notify("✅ Voice Input", f"Post-processor: {POST_PROCESSOR_PRESETS[new_pp_id]['name']}")
                    except Exception as e:
                        notify("❌ Voice Input", f"Failed to switch post-processor: {e}", urgency="critical")

            # Handle model switching (requires daemon restart)
            if result.get("model_changed"):
                new_model_id = result.get("new_model")
                if new_model_id and new_model_id != self.current_model_id:
                    self._switch_model(new_model_id)
                    
        except ImportError as e:
            notify("❌ Voice Input", f"Settings dialog unavailable\n{e}", urgency="critical")
        except Exception as e:
            notify("❌ Voice Input", f"Failed to open settings\n{e}", urgency="critical")
    
    def _switch_model(self, new_model_id):
        """Switch to a new model (requires daemon restart)."""
        preset = MODEL_PRESETS.get(new_model_id, {})
        model_name = preset.get("name", new_model_id)
        
        notify("🔄 Voice Input", f"Switching to {model_name}...\nDaemon will restart, please wait")
        
        # Get startup paths
        venv_python, script_path = get_daemon_paths()
        
        # Start the new daemon process (in background), then exit the current process
        # Note: must start before exiting, otherwise threads will terminate with the process
        pp_arg = f' --post-processor {self.current_post_processor_id}' if self.current_post_processor_id != "none" else ''
        cmd = f'nohup "{venv_python}" "{script_path}" _daemon --model {new_model_id}{pp_arg} > /tmp/voice-input-daemon.log 2>&1 &'
        subprocess.run(["bash", "-c", cmd])
        
        # Delay briefly before exiting current process, giving the new process time to start
        def delayed_quit():
            time.sleep(0.5)
            self.running = False
            GLib.idle_add(Gtk.main_quit)
        
        threading.Thread(target=delayed_quit, daemon=True).start()

    def set_status(self, status):
        """Set status (idle/recording/processing)"""
        _log("STATE", f"icon status: {status}")
        print(f"[Indicator] Setting status to: {status}")
        if not self.indicator:
            print("[Indicator] No indicator available")
            return

        config = STATUS_CONFIG.get(status, STATUS_CONFIG["idle"])
        icon_name, tooltip, label = config

        def update():
            try:
                self.indicator.set_icon_full(icon_name, tooltip)
                self.status_item.set_label(label)
                print(f"[Indicator] Icon updated to: {status}")
            except Exception as e:
                print(f"[Indicator] Error updating icon: {e}")
            return False

        GLib.idle_add(update)
    
    def load_model(self, model_id=None):
        """
        Load a single model (single-model architecture, saves memory)

        Args:
            model_id: Model ID to load. If None, uses the currently configured model.

        Raises:
            RuntimeError: If model loading fails.
        """
        if model_id is None:
            model_id = self.current_model_id or get_current_model()

        if model_id not in MODEL_PRESETS:
            raise RuntimeError(f"Unknown model: {model_id}")

        preset = MODEL_PRESETS[model_id]

        print("\n" + "="*60)
        print(f"🚀 Loading ASR model: {preset['name']}")
        print("="*60)
        print(f"  Model ID: {model_id}")
        print(f"  Framework: {preset['framework']}")
        print(f"  Description: {preset['description']}")
        print(f"  Device: {DEVICE}")
        print(f"  ⏳ Estimated 20-60 seconds (depending on model size)...")
        print()

        try:
            # Use ModelLoader to load the model
            self.model, self.framework, self.extra_data = ModelLoader.load_model(model_id, DEVICE)
            self.current_model_id = model_id
            set_current_model(model_id)
            _log("MODEL", f"primary ASR loaded: {model_id} ({preset['name']})")

            print(f"  ✓ {preset['name']} loaded successfully!")
            print("="*60)
            print()
        except Exception as e:
            error_msg = f"Model loading failed: {e}"
            print(f"  ✗ {error_msg}")
            notify("❌ Voice Input", error_msg, urgency="critical")
            raise RuntimeError(error_msg)

    def load_punctuation_model(self):
        """Auto-load punctuation model based on current ASR model config."""
        preset = MODEL_PRESETS.get(self.current_model_id, {})
        punc_type = preset.get("punctuation", "none")

        if punc_type == "firered-punc":
            punc_config = preset.get("punc_config", {})
            model_dir = os.path.expanduser(punc_config.get("model_dir", ""))
            try:
                self.punc_model = PostProcessorLoader.load_firered_punc({"model_dir": model_dir})
                _log("PUNC", f"auto-loaded FireRedPunc for {self.current_model_id}")
                print(f"  Auto-punctuation: FireRedPunc loaded")
            except Exception as e:
                _log("PUNC", f"FAILED to load FireRedPunc: {e}")
                print(f"  Auto-punctuation failed: {e}")
                self.punc_model = None
        elif punc_type == "builtin":
            self.punc_model = None
            _log("PUNC", f"built-in punctuation for {self.current_model_id}")
            print(f"  Auto-punctuation: built-in ({self.current_model_id})")
        else:
            self.punc_model = None
            _log("PUNC", f"no punctuation for {self.current_model_id}")

    def _load_secondary_model(self):
        """Load faster-whisper as secondary ASR model for dual ASR fusion.

        Non-fatal: if loading fails, secondary_model stays None and fusion
        degrades gracefully to single-ASR mode.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logging.warning(
                "faster-whisper not installed, secondary ASR unavailable. "
                "Install with: pip install faster-whisper"
            )
            self._secondary_model = None
            return

        try:
            _log("SECONDARY", "loading secondary ASR model (faster-whisper)...")
            logging.info("Loading secondary ASR model (faster-whisper large-v3-turbo, GPU, int8_float16)...")
            print("  Loading secondary ASR: faster-whisper (GPU, int8_float16)...")
            self._secondary_model = WhisperModel(
                "large-v3-turbo", device="cuda", compute_type="int8_float16"
            )
            _log("SECONDARY", "faster-whisper loaded successfully")
            print("  Secondary ASR ready: faster-whisper")
        except Exception as e:
            logging.warning(f"Failed to load secondary ASR model: {e}")
            print(f"  Secondary ASR failed: {e}")
            self._secondary_model = None

    def _unload_secondary_model(self):
        """Unload secondary ASR model to free memory."""
        if getattr(self, '_secondary_model', None) is not None:
            self._secondary_model = None
            self._last_secondary_text = None
            _log("SECONDARY", "unloaded secondary ASR model")
            print("  Secondary ASR unloaded")

    def _sync_status_from_db(self):
        """Poll DB for status changes and update GTK icon if changed.

        Called every 1 second in socket_server() timeout loop.
        Never raises — all errors are caught and logged.
        """
        try:
            state = get_state(STATE_DB_PATH)
            new_status = state.get("status", "idle")
            if new_status != self._current_db_status:
                _log("SYNC", f"DB status changed: {self._current_db_status} → {new_status}")
                self.set_status(new_status)
                self._current_db_status = new_status
        except Exception as e:
            _log("ERROR", f"_sync_status_from_db failed: {e}")

    def load_post_processor(self, preset_id=None):
        """Load a post-processor model."""
        if preset_id is None:
            preset_id = self.current_post_processor_id

        if preset_id not in POST_PROCESSOR_PRESETS:
            raise RuntimeError(f"Unknown post-processor: {preset_id}")

        preset = POST_PROCESSOR_PRESETS[preset_id]

        print(f"\n{'='*60}")
        print(f"Loading post-processor: {preset['name']}")
        print(f"{'='*60}")

        # haiku-expand: not yet implemented — check BEFORE try block (CRITIC-R4-C1)
        # so the raise propagates to the daemon command handler (inner except swallows all)
        if preset.get("framework") == "ssh-claude" and not preset.get("config"):
            notify("Votype", "Haiku Expand is not yet implemented", urgency="normal")
            raise ValueError("Haiku Expand is not yet implemented")

        try:
            self.post_processor_model = PostProcessorLoader.load_post_processor(preset_id)
            self.current_post_processor_id = preset_id
            update_state(STATE_DB_PATH, post_processor=preset_id)
            self.post_processor_framework = preset["framework"]
            # Load vocab once for SSH-based frameworks (cached on instance)
            if self.post_processor_framework in ("ssh-claude", "vertex-ai", "vertex-ai-merge"):
                from post_processor_configs import load_vocab
                self._vocab = load_vocab()
                _log("PP-LOAD", f"vocab loaded: {len(self._vocab)} terms")
            # Load/unload secondary ASR model based on post-processor
            if self.post_processor_framework == "vertex-ai-merge":
                self._load_secondary_model()
                _log("PP-LOAD", "secondary ASR loaded for merge mode")
            else:
                self._unload_secondary_model()
                _log("PP-LOAD", f"secondary ASR unloaded (not needed for {preset_id})")
            _log("PP-LOAD", f"loaded: {preset['name']} ({preset_id})")
            print(f"  Post-processor ready: {preset['name']}")
            print(f"{'='*60}\n")
        except Exception as e:
            error_msg = f"Post-processor loading failed: {e}"
            _log("PP-LOAD", f"FAILED: {error_msg}")
            print(f"  {error_msg}")
            # Fall back to regex-only
            _log("PP-LOAD", "post-processor load failed, falling back to regex-only")
            self.post_processor_model = None
            self.current_post_processor_id = "none"
            self.post_processor_framework = "regex"
            self._unload_secondary_model()
            print("  Falling back to regex-only mode")

    def _post_process(self, text):
        """Apply post-processing to transcribed text.

        Pipeline: regex filler removal -> auto-punctuation (if needed)
                  -> glossary regex (ssh-claude/vertex-ai) -> LLM polish (ssh-claude/vertex-ai)
                  -> vocab accumulation (ssh-claude/vertex-ai) -> LLM refinement (llama-cpp).
        """
        import time
        _log("PP", f"input ({self.current_post_processor_id}): {text[:120]}")
        t0 = time.time()

        # Step 1: Always remove fillers (regex)
        result = PostProcessorInference.remove_fillers(text)

        # Step 2: Auto-punctuation (model-config driven, separate from post-processor)
        if self.punc_model is not None and result:
            try:
                result = PostProcessorInference.process_with_firered_punc(
                    self.punc_model, result
                )
                _log("PUNC", f"applied punctuation: {result[:120]}")
            except Exception as e:
                _log("PUNC", f"FAILED: {e}")

        # Step 3: SSH-based post-processing (glossary regex + LLM polish + vocab accumulation)
        if result and self.post_processor_framework in ("ssh-claude", "vertex-ai", "vertex-ai-merge"):
            from post_processor_configs import (
                apply_vocab, glossary_context, load_vocab,
                process_with_ssh_claude,
                process_with_vertex_ai, process_with_gemini_merge,
                diff_to_vocab, save_vocab,
            )

            preset = POST_PROCESSOR_PRESETS[self.current_post_processor_id]
            config = preset["config"]
            min_count = config.get("vocab_min_count", 3)

            # Glossary regex replacement + vocab context
            result = apply_vocab(result, self._vocab, min_count)
            glossary_ctx = glossary_context(self._vocab)
            before_polish = result

            if self.post_processor_framework == "vertex-ai-merge":
                # Dual ASR fusion: primary processed text + raw secondary text → Gemini merge
                secondary = getattr(self, '_last_secondary_text', None)
                result = process_with_gemini_merge(result, secondary, config, glossary_ctx)
            else:
                # Single-ASR polish (ssh-claude or vertex-ai)
                process_fn = {
                    "ssh-claude": process_with_ssh_claude,
                    "vertex-ai": process_with_vertex_ai,
                }[self.post_processor_framework]
                result = process_fn(result, config, glossary_ctx)

            # Vocab accumulation (only if LLM changed something)
            if before_polish != result:
                self._vocab = diff_to_vocab(before_polish, result, self._vocab)
                _log("VOCAB", "terms extracted from LLM diff")
                save_vocab(self._vocab)
                _log("VOCAB", f"vocab saved ({len(self._vocab)} entries)")
                # Reload merged vocab (save_vocab merges with disk)
                self._vocab = load_vocab()

        # Step 4: Optional LLM post-processing
        if result and self.post_processor_model is not None:
            preset = POST_PROCESSOR_PRESETS.get(self.current_post_processor_id, {})
            framework = preset.get("framework")
            if framework == "llama-cpp":
                prompt_template = preset.get("config", {}).get("prompt_template", "")
                if prompt_template:
                    try:
                        result = PostProcessorInference.process_with_llm(
                            self.post_processor_model, result, prompt_template
                        )
                    except Exception as e:
                        logging.error(f"LLM post-processing failed: {e}")

        elapsed = time.time() - t0
        _log("PP", f"output ({elapsed:.2f}s): {result[:120]}")
        return result

    def transcribe(self, audio_path):
        """Transcribe audio (with hotword support)"""
        # Check if current model is available
        if self.model is None:
            return {"error": "Model not loaded"}

        try:
            # Use ModelInference.transcribe for unified inference
            text = ModelInference.transcribe(
                self.model,
                audio_path,
                self.current_model_id,
                self.framework,
                self.extra_data,
                HOTWORDS
            )
            return {"text": text}
        except Exception as e:
            return {"error": str(e)}
    
    def _handle_transcribe(self, msg):
        """Handle transcription request with parallel dual ASR.

        Note: does NOT set idle status — the CLI process sends set_idle
        after type_text() completes, so the icon stays orange during the
        entire pipeline (ASR → post-process → paste).

        When secondary ASR model is loaded (gemini-merge mode), runs
        secondary ASR in a background thread while primary runs in the
        main thread. Both release the GIL via C extensions (FunASR CUDA /
        CTranslate2), enabling true CPU+GPU parallelism.
        """
        self.set_status("processing")
        audio_path = msg.get("data")
        t_start = time.time()

        # Reset stale data at start of every call
        self._last_secondary_text = None
        _log("SECONDARY", "secondary text reset")

        # Start secondary ASR in background thread BEFORE primary
        secondary_thread = None
        secondary_result = {}

        if getattr(self, '_secondary_model', None) is not None and audio_path:
            def _run_secondary(model, path, result):
                t0 = time.time()
                try:
                    segments, _info = model.transcribe(path, language="zh")
                    result["text"] = "".join(seg.text for seg in segments)
                    result["elapsed"] = time.time() - t0
                except Exception as e:
                    logging.warning(f"Secondary ASR failed: {e}")
                    result["error"] = str(e)

            secondary_thread = threading.Thread(
                target=_run_secondary,
                args=(self._secondary_model, audio_path, secondary_result),
                daemon=True,
            )
            secondary_thread.start()
            _log("SECONDARY", "secondary ASR thread started")

        # Primary ASR in main thread (GPU)
        t_primary = time.time()
        response = self.transcribe(audio_path)
        t_primary_done = time.time()
        _log("TIMING", f"primary ASR: {t_primary_done - t_primary:.2f}s")

        # Collect secondary result after primary completes
        if secondary_thread is not None:
            _log("SECONDARY", "waiting for secondary thread")
            secondary_thread.join(timeout=30)
            t_join = time.time()
            sec_elapsed = secondary_result.get("elapsed", -1)
            _log("TIMING", f"secondary ASR: {sec_elapsed:.2f}s, join wait: {t_join - t_primary_done:.2f}s, total: {t_join - t_start:.2f}s")

            primary_text = (response or {}).get("text", "")
            if not primary_text:
                self._last_secondary_text = None
            elif secondary_thread.is_alive():
                logging.warning("Secondary ASR timed out after 30s")
                _log("SECONDARY", "secondary thread timed out")
                self._last_secondary_text = None
            elif "error" in secondary_result:
                _log("ERROR", f"secondary ASR error: {secondary_result.get('error', 'unknown')}")
                self._last_secondary_text = None
            elif "text" in secondary_result:
                self._last_secondary_text = secondary_result["text"]
                _log("ASR-2", f"secondary: {self._last_secondary_text[:120]}")
            else:
                self._last_secondary_text = None

        if response and "text" in response and response["text"]:
            _log("ASR", f"raw: {response['text'][:120]}")
            response["text"] = self._post_process(response["text"])
        elif response and "error" in response:
            _log("ASR", f"error: {response['error']}")
        return response

    def _handle_stop(self, msg):
        """Handle stop request"""
        self.running = False
        if HAS_INDICATOR:
            GLib.idle_add(Gtk.main_quit)
        return {"status": "stopping"}

    def handle_client(self, client):
        """Handle client request"""
        try:
            data = client.recv(4096).decode()
            msg = json.loads(data)
            command = msg.get("command")

            if command == "transcribe":
                response = self._handle_transcribe(msg)
            elif command == "stop":
                response = self._handle_stop(msg)
            elif command == "ping":
                response = {"status": "ok", "model": self.current_model_id}
            elif command == "get_model":
                preset = MODEL_PRESETS.get(self.current_model_id, {})
                response = {
                    "model": self.current_model_id,
                    "name": preset.get("name", "Unknown"),
                    "description": preset.get("description", ""),
                }
            elif command == "list_models":
                response = {
                    "models": {
                        mid: {"name": p["name"], "description": p["description"]}
                        for mid, p in MODEL_PRESETS.items()
                    },
                    "current": self.current_model_id,
                }
            elif command == "get_post_processor":
                preset = POST_PROCESSOR_PRESETS.get(self.current_post_processor_id, {})
                response = {
                    "post_processor": self.current_post_processor_id,
                    "name": preset.get("name", "Unknown"),
                    "description": preset.get("description", ""),
                }
            elif command == "list_post_processors":
                response = {
                    "post_processors": {
                        pid: {"name": p["name"], "description": p["description"]}
                        for pid, p in POST_PROCESSOR_PRESETS.items()
                    },
                    "current": self.current_post_processor_id,
                }
            elif command == "set_post_processor":
                new_pp_id = (msg.get("data") or {}).get("post_processor_id")
                if new_pp_id == "firered-punc":
                    response = {"error": "firered-punc is now automatic with firered-asr, not a post-processor. Use 'voice-input post-processors' to see available options."}
                elif not new_pp_id or new_pp_id not in POST_PROCESSOR_PRESETS:
                    response = {"error": f"Unknown post-processor: {new_pp_id}"}
                elif new_pp_id == self.current_post_processor_id:
                    response = {"status": "ok", "message": "Already using this post-processor"}
                else:
                    try:
                        _log("SOCKET", f"post-processor switch requested: {new_pp_id}")
                        self.load_post_processor(new_pp_id)
                        preset = POST_PROCESSOR_PRESETS[new_pp_id]
                        notify("✅ Voice Input", f"Post-processor: {preset['name']}")
                        response = {"status": "ok", "post_processor": new_pp_id, "name": preset["name"]}
                    except Exception as e:
                        response = {"error": f"Failed to switch post-processor: {e}"}
            else:
                response = {"error": f"Unknown command: {command}"}

            client.send(json.dumps(response).encode())
        except Exception as e:
            _log("ERROR", f"socket handler error: {e}")
            try:
                client.send(json.dumps({"error": str(e)}).encode())
            except:
                pass
        finally:
            client.close()
    
    def socket_server(self):
        """Socket server thread"""
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(SOCKET_PATH))
        _log("SOCKET", f"daemon socket bound: {SOCKET_PATH}")
        server.listen(5)
        server.settimeout(1)
        
        while self.running:
            try:
                client, _ = server.accept()
                threading.Thread(target=self.handle_client, args=(client,)).start()
            except socket.timeout:
                self._sync_status_from_db()
                continue
            except Exception as e:
                if self.running:
                    print(f"Error: {e}")
        
        server.close()
    
    def run(self):
        """Run the daemon"""
        ensure_config_dir()

        # Acquire exclusive file lock — prevents two daemons from running simultaneously.
        # The lock is held for the daemon's lifetime and auto-released on crash/exit.
        self._lock_fd = open(DAEMON_LOCK_FILE, 'w')
        _log("LOCK", "daemon lock file opened")
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _log("LOCK", "daemon lock acquired")
        except (IOError, OSError):
            _log("ERROR", "daemon lock held by another process")
            print("Another daemon instance is already running (lock held). Exiting.")
            self._lock_fd.close()
            return

        # Write PID file and DB
        DAEMON_PID_FILE.write_text(str(os.getpid()))
        update_state(STATE_DB_PATH, daemon_pid=os.getpid())
        _log("DAEMON", f"daemon PID written: {os.getpid()}")

        # Remove old socket file
        SOCKET_PATH.unlink(missing_ok=True)

        # Load a single model (saves memory)
        try:
            self.load_model()
        except Exception as e:
            print(f"Failed to load model: {e}")
            DAEMON_PID_FILE.unlink(missing_ok=True)
            sys.exit(1)

        # Auto-load punctuation model based on ASR model config
        self.load_punctuation_model()

        # Load post-processor (non-fatal: falls back to regex-only)
        # Backward compat: firered-punc is no longer a valid post-processor
        if self.current_post_processor_id == "firered-punc":
            print("Note: firered-punc is now automatic with firered-asr, using 'none' instead.")
            self.current_post_processor_id = "none"
        try:
            self.load_post_processor()
        except Exception as e:
            print(f"Post-processor loading failed, using regex-only: {e}")

        print(f"Daemon started (PID: {os.getpid()})")
        print(f"Model ready: {MODEL_PRESETS[self.current_model_id]['name']}")
        pp_name = POST_PROCESSOR_PRESETS.get(self.current_post_processor_id, {}).get('name', 'None')
        print(f"Post-processor: {pp_name}")
        print("Use 'voice-input toggle' to start/stop recording.")
        
        self.running = True
        _log("DAEMON", "daemon startup complete")

        # Set up system tray icon first (before starting socket server)
        if HAS_INDICATOR:
            self.setup_indicator()
            print("Tray icon initialized")
        
        # Start socket server thread
        server_thread = threading.Thread(target=self.socket_server, daemon=True)
        server_thread.start()
        
        # Run main loop
        if HAS_INDICATOR:
            logging.info("Background service started, tray icon displayed")
            # GTK main loop (blocking)
            _log("DAEMON", "GTK main loop started")
            Gtk.main()
        else:
            logging.info("Background service started, model loaded successfully")
            # Without GTK, wait for running to become False
            while self.running:
                time.sleep(1)
        
        # Cleanup
        _log("DAEMON", "daemon shutdown initiated")
        self.running = False
        update_state(STATE_DB_PATH, daemon_pid=None, status="idle")
        SOCKET_PATH.unlink(missing_ok=True)
        DAEMON_PID_FILE.unlink(missing_ok=True)
        if hasattr(self, '_lock_fd'):
            self._lock_fd.close()
        _log("DAEMON", "daemon cleanup complete (socket, PID, lock, DB)")
        print("Daemon stopped")


def stop_daemon():
    """Stop the daemon"""
    if not is_daemon_running():
        print("Daemon is not running")
        return
    
    response = send_to_daemon("stop")
    if response:
        print("Daemon stopped")
    else:
        # Force kill
        try:
            pid = int(DAEMON_PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            print("Daemon killed")
        except:
            pass
    
    DAEMON_PID_FILE.unlink(missing_ok=True)
    SOCKET_PATH.unlink(missing_ok=True)


def show_status():
    """Show status"""
    print(f"Recording: {'Yes' if is_recording() else 'No'}")
    print(f"Daemon: {'Running' if is_daemon_running() else 'Not running'}")

    if is_daemon_running():
        response = send_to_daemon("get_model")
        if response and "model" in response:
            print(f"Model: {response.get('name', 'Unknown')} ({response.get('model')})")
            print(f"  {response.get('description', '')}")
        else:
            print("Daemon: Not responsive")
        pp_response = send_to_daemon("get_post_processor")
        if pp_response and "post_processor" in pp_response:
            print(f"Post-processor: {pp_response.get('name', 'Unknown')} ({pp_response.get('post_processor')})")
    else:
        # Show current model from config file
        model_id = get_current_model()
        preset = MODEL_PRESETS.get(model_id, {})
        print(f"Configured Model: {preset.get('name', 'Unknown')} ({model_id})")


def set_post_processor():
    """Set post-processor via CLI"""
    if len(sys.argv) < 3:
        print("Usage: voice-input post-processor <id>")
        print(f"Available: {', '.join(POST_PROCESSOR_PRESETS.keys())}")
        sys.exit(1)

    pp_id = sys.argv[2].lower()
    if pp_id == "firered-punc":
        print("firered-punc is now automatic with firered-asr, not a post-processor.")
        print("Punctuation is auto-applied when using firered-asr model.")
        print(f"Available post-processors: {', '.join(POST_PROCESSOR_PRESETS.keys())}")
        sys.exit(1)
    if pp_id not in POST_PROCESSOR_PRESETS:
        print(f"Unknown post-processor: {pp_id}")
        print(f"Available: {', '.join(POST_PROCESSOR_PRESETS.keys())}")
        sys.exit(1)

    if not is_daemon_running():
        print("Daemon is not running")
        sys.exit(1)

    response = send_to_daemon("set_post_processor", {"post_processor_id": pp_id})
    if response and response.get("status") == "ok":
        name = response.get("name", pp_id)
        msg = response.get("message", f"Switched to: {name}")
        print(msg)
    elif response and "error" in response:
        print(f"Error: {response['error']}")
    else:
        print("Daemon not responsive")


def list_post_processors():
    """List available post-processors"""
    print("Available post-processors:")
    print("-" * 50)

    response = send_to_daemon("list_post_processors") if is_daemon_running() else None
    current = response.get("current") if response else None

    for pp_id, preset in POST_PROCESSOR_PRESETS.items():
        marker = "→" if pp_id == current else " "
        print(f"  {marker} {pp_id}")
        print(f"      {preset['name']}: {preset['description']}")
    print()


def list_models():
    """List available models"""
    print("Available models:")
    print("-" * 50)

    current = get_current_model()
    for model_id, preset in MODEL_PRESETS.items():
        marker = "→" if model_id == current else " "
        print(f"  {marker} {model_id}")
        print(f"      Name: {preset['name']}")
        print(f"      {preset['description']}")
        print()


def run_daemon(model_id=None, post_processor_id=None):
    """Run the daemon (internal command)"""
    daemon = ASRDaemon(model_id=model_id)
    if post_processor_id:
        # Backward compat: firered-punc is no longer a post-processor
        if post_processor_id == "firered-punc":
            print("Note: firered-punc is now automatic with firered-asr, ignoring post-processor setting.")
        else:
            daemon.current_post_processor_id = post_processor_id
    daemon.run()


def start_daemon_with_model(model_id=None, post_processor_id=None):
    """Start the daemon (supports specifying a model and post-processor)"""
    if is_daemon_running():
        print("Daemon is already running")
        return

    venv_python, script_path = get_daemon_paths()

    # Build command with optional --model and --post-processor
    cmd_parts = [f'nohup "{venv_python}" "{script_path}" _daemon']
    if model_id:
        cmd_parts.append(f'--model {model_id}')
    if post_processor_id:
        cmd_parts.append(f'--post-processor {post_processor_id}')
    cmd_parts.append('> /tmp/voice-input-daemon.log 2>&1 &')
    cmd = ' '.join(cmd_parts)

    subprocess.run(["bash", "-c", cmd])

    model_name = MODEL_PRESETS.get(model_id or DEFAULT_MODEL, {}).get('name', 'default model')
    print(f"Starting daemon with {model_name}... (loading model, please wait)")

    for i in range(60):
        time.sleep(1)
        if is_daemon_running():
            print("Daemon started successfully!")
            print("Tray icon should appear in system tray.")
            return
        if i % 10 == 9:
            print(f"  Still loading... ({i+1}s)")

    print("Failed to start daemon. Check /tmp/voice-input-daemon.log for errors.")


def main():
    import argparse

    # Basic commands (no arguments)
    simple_commands = {
        "start": start_recording,
        "stop": stop_recording,
        "toggle": toggle_recording,
        "kill": stop_daemon,
        "status": show_status,
        "models": list_models,
        "post-processors": list_post_processors,
        "post-processor": set_post_processor,
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h", "help"):
        print(__doc__)
        print("Commands:")
        print("  start                  Start recording")
        print("  stop                   Stop recording and transcribe")
        print("  toggle                 Toggle recording (bind to hotkey)")
        print("  daemon                 Start background service")
        print("  kill                   Stop background service")
        print("  status                 Show current status and model")
        print("  models                 List available ASR models")
        print("  post-processors        List available LLM post-processors")
        print("  post-processor <id>    Switch LLM post-processor")
        print("\nPunctuation is automatic per ASR model (firered-asr uses FireRedPunc).")
        print("\nOptions for daemon:")
        print("  --model <id>           Specify model to load")
        print("  --post-processor <id>  Specify LLM post-processor")
        sys.exit(0)

    command = sys.argv[1].lower()

    # Both daemon / _daemon commands support --model argument
    daemon_handlers = {"daemon": start_daemon_with_model, "_daemon": run_daemon}
    if command in daemon_handlers:
        parser = argparse.ArgumentParser(prog=f'voice-input {command}')
        parser.add_argument('--model', '-m', choices=list(MODEL_PRESETS.keys()),
                            help='Model to load on startup')
        parser.add_argument('--post-processor', '-p', choices=list(POST_PROCESSOR_PRESETS.keys()),
                            default=None, help='Post-processor to use')
        args = parser.parse_args(sys.argv[2:])
        daemon_handlers[command](args.model, getattr(args, 'post_processor', None))
        return

    handler = simple_commands.get(command)

    if handler:
        handler()
    else:
        print(f"Unknown command: {command}")
        print(f"Available commands: {', '.join(simple_commands.keys())}, daemon")
        sys.exit(1)


if __name__ == "__main__":
    main()
