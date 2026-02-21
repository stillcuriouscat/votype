"""
pytest fixtures for voice_input tests.

Provides isolated test environments to avoid affecting real configuration.
"""

import os
import sys
import json
import socket
import signal
import threading
import time
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import voice_input and save defaults (before any test modifications)
import voice_input
_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "voice-input"
_DEFAULT_PID_FILE = _DEFAULT_CONFIG_DIR / "recording.pid"
_DEFAULT_AUDIO_FILE = _DEFAULT_CONFIG_DIR / "recording.wav"
_DEFAULT_DAEMON_PID_FILE = _DEFAULT_CONFIG_DIR / "daemon.pid"
_DEFAULT_SOCKET_PATH = _DEFAULT_CONFIG_DIR / "daemon.sock"
_DEFAULT_MODEL_STATE_FILE = voice_input.MODEL_STATE_FILE

# Save original function references (prevent mock leaks)
_ORIGINAL_IS_RECORDING = voice_input.is_recording
_ORIGINAL_IS_DAEMON_RUNNING = voice_input.is_daemon_running
_ORIGINAL_SEND_TO_DAEMON = voice_input.send_to_daemon
_ORIGINAL_TYPE_TEXT = voice_input.type_text


# ============ Global Test Isolation ============

def _get_all_arecord_pids():
    """Get all arecord process PIDs (not limited to test processes).

    The previous implementation used `pgrep -f "arecord.*pytest"` which was wrong:
    the arecord command is `arecord -f S16_LE ... /path/to/file.wav`,
    which doesn't contain "pytest" in the command line, so it couldn't match.

    Fix: match all arecord processes, then only clean up processes added during testing.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-x", "arecord"],  # -x for exact process name match
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return [int(pid) for pid in result.stdout.strip().split('\n') if pid]
        return []
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        return []


def _kill_test_arecord_processes(pids_to_keep=None):
    """Clean up arecord processes added during testing.

    Args:
        pids_to_keep: Set of arecord process PIDs that existed before testing; these should not be cleaned up.
    """
    if pids_to_keep is None:
        pids_to_keep = set()

    current_pids = set(_get_all_arecord_pids())
    pids_to_kill = current_pids - pids_to_keep

    for pid in pids_to_kill:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[cleanup] Killed orphan arecord process: {pid}")
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"[cleanup] No permission to kill process: {pid}")


@pytest.fixture(autouse=True)
def cleanup_test_processes(request):
    """
    Global autouse fixture to ensure arecord processes created by tests are cleaned up.
    Cleanup runs even if the test is interrupted (Ctrl+C).
    """
    # Record arecord processes before the test (these should be kept)
    initial_pids = set(_get_all_arecord_pids())

    # Register finalizer (runs even if the test fails)
    def cleanup():
        _kill_test_arecord_processes(pids_to_keep=initial_pids)

    request.addfinalizer(cleanup)

    yield

    # Also clean up when the test ends normally
    cleanup()


@pytest.fixture(autouse=True)
def reset_voice_input_state(request):
    """
    Global autouse fixture to restore voice_input module state after each test.
    This prevents state leakage between tests (including path constants and function references).
    """
    yield
    # Restore default paths after test
    voice_input.CONFIG_DIR = _DEFAULT_CONFIG_DIR
    voice_input.PID_FILE = _DEFAULT_PID_FILE
    voice_input.AUDIO_FILE = _DEFAULT_AUDIO_FILE
    voice_input.DAEMON_PID_FILE = _DEFAULT_DAEMON_PID_FILE
    voice_input.SOCKET_PATH = _DEFAULT_SOCKET_PATH
    voice_input.MODEL_STATE_FILE = _DEFAULT_MODEL_STATE_FILE

    # Restore original function references (prevent mock leaks)
    voice_input.is_recording = _ORIGINAL_IS_RECORDING
    voice_input.is_daemon_running = _ORIGINAL_IS_DAEMON_RUNNING
    voice_input.send_to_daemon = _ORIGINAL_SEND_TO_DAEMON
    voice_input.type_text = _ORIGINAL_TYPE_TEXT


# ============ Basic Fixtures ============

@pytest.fixture
def temp_config_dir(tmp_path):
    """
    Use a temporary config directory to avoid affecting the real environment.
    Automatically patches path constants in the voice_input module.
    """
    config_dir = tmp_path / "config" / "voice-input"
    config_dir.mkdir(parents=True)

    pid_file = config_dir / "recording.pid"
    audio_file = config_dir / "recording.wav"
    daemon_pid_file = config_dir / "daemon.pid"
    socket_path = config_dir / "daemon.sock"

    with patch.multiple(
        'voice_input',
        CONFIG_DIR=config_dir,
        PID_FILE=pid_file,
        AUDIO_FILE=audio_file,
        DAEMON_PID_FILE=daemon_pid_file,
        SOCKET_PATH=socket_path,
    ):
        yield {
            'config_dir': config_dir,
            'pid_file': pid_file,
            'audio_file': audio_file,
            'daemon_pid_file': daemon_pid_file,
            'socket_path': socket_path,
        }


@pytest.fixture
def temp_socket(tmp_path):
    """Create a temporary socket path for testing."""
    socket_path = tmp_path / "test.sock"
    yield socket_path
    # Cleanup
    if socket_path.exists():
        socket_path.unlink()


# ============ Mock Fixtures ============

@pytest.fixture
def mock_subprocess():
    """Mock subprocess calls."""
    with patch('voice_input.subprocess') as mock:
        # Mock Popen (used by the new start_recording implementation)
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0
        mock_proc.communicate = MagicMock(return_value=(b"", b""))
        mock.Popen.return_value = mock_proc

        # Mock run (used by legacy code and some tests)
        mock.run.return_value = MagicMock(
            returncode=0,
            stdout="12345\n",  # Compatible with legacy tests
            stderr=""
        )

        # Provide a helper method for tests to easily configure Popen
        def configure_popen(pid=12345, **kwargs):
            """Configure Popen mock return values."""
            proc = MagicMock()
            proc.pid = pid
            proc.returncode = kwargs.get('returncode', 0)
            proc.communicate = MagicMock(return_value=(
                kwargs.get('stdout', b""),
                kwargs.get('stderr', b"")
            ))
            mock.Popen.return_value = proc
            return proc

        mock.configure_popen = configure_popen

        yield mock


@pytest.fixture
def mock_notify():
    """Mock desktop notifications."""
    with patch('voice_input.notify') as mock:
        yield mock


@pytest.fixture
def mock_asr_model():
    """
    Mock FunASR model.
    Used for fast testing without loading real models.
    AutoModel is imported inside functions, so we need to patch funasr.AutoModel.
    """
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"text": "test recognition result"}]

    mock_auto_model_class = MagicMock(return_value=mock_model)

    # Create a mock funasr module
    mock_funasr = MagicMock()
    mock_funasr.AutoModel = mock_auto_model_class

    # AutoModel is imported from funasr inside functions, so we patch the funasr module
    with patch.dict('sys.modules', {'funasr': mock_funasr}):
        yield {
            'model_class': mock_auto_model_class,
            'model_instance': mock_model,
        }


@pytest.fixture
def mock_gtk(monkeypatch):
    """Mock GTK-related modules."""
    import voice_input

    mock_indicator = MagicMock()
    mock_gtk_module = MagicMock()
    mock_glib = MagicMock()

    # GLib.idle_add executes callback immediately
    mock_glib.idle_add = lambda func: func()

    mock_appindicator = MagicMock(
        Indicator=MagicMock(new=MagicMock(return_value=mock_indicator)),
        IndicatorCategory=MagicMock(APPLICATION_STATUS=0),
        IndicatorStatus=MagicMock(ACTIVE=1),
    )

    # Use monkeypatch to set attributes even if they don't exist
    monkeypatch.setattr(voice_input, 'HAS_INDICATOR', True)
    monkeypatch.setattr(voice_input, 'AyatanaAppIndicator3', mock_appindicator, raising=False)
    monkeypatch.setattr(voice_input, 'Gtk', mock_gtk_module, raising=False)
    monkeypatch.setattr(voice_input, 'GLib', mock_glib, raising=False)

    yield {
        'indicator': mock_indicator,
        'gtk': mock_gtk_module,
        'glib': mock_glib,
    }


# ============ Process Simulation Fixtures ============

@pytest.fixture
def fake_process(tmp_path):
    """
    Create a fake process (actually the current process) for testing process check logic.
    Returns the current process PID.
    """
    yield os.getpid()


@pytest.fixture
def dead_process_pid():
    """
    Return a PID that does not exist.
    Uses the /proc filesystem to directly check if a PID exists (more reliable).
    """
    # On Linux, directly checking /proc/{pid} is more reliable
    # This avoids permission issues with os.kill
    for pid in range(99999, 10000, -1):
        proc_path = Path(f"/proc/{pid}")
        if not proc_path.exists():
            # Confirm os.kill also raises ProcessLookupError
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                yield pid
                return
            except PermissionError:
                # Process exists but no permission, keep looking
                continue
    # Last resort: use an extremely large PID
    yield 999999999


# ============ Socket Server Fixtures ============

@pytest.fixture
def mock_socket_server(temp_socket):
    """
    Start a mock daemon socket server.
    Used for testing client communication logic.
    """
    responses = {
        "ping": {"status": "ok"},
        "transcribe": {"text": "mock transcription result"},
        "recording_start": {"status": "ok"},
        "recording_stop": {"status": "ok"},
        "set_idle": {"status": "ok"},
        "stop": {"status": "stopping"},
    }

    server_running = threading.Event()
    server_ready = threading.Event()

    def server_thread():
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(str(temp_socket))
        server.listen(5)
        server.settimeout(0.5)

        server_ready.set()

        while server_running.is_set():
            try:
                client, _ = server.accept()
                data = client.recv(4096).decode()
                msg = json.loads(data)
                command = msg.get("command", "")

                response = responses.get(command, {"error": f"Unknown: {command}"})
                client.send(json.dumps(response).encode())
                client.close()
            except socket.timeout:
                continue
            except Exception:
                break

        server.close()

    server_running.set()
    thread = threading.Thread(target=server_thread, daemon=True)
    thread.start()
    server_ready.wait(timeout=2)

    yield {
        'socket_path': temp_socket,
        'responses': responses,
        'running': server_running,
    }

    server_running.clear()
    thread.join(timeout=1)


# ============ Integration Test Fixtures ============

@pytest.fixture
def isolated_environment(tmp_path, monkeypatch):
    """
    Fully isolated test environment.
    Redirects all paths to a temporary directory.
    """
    config_dir = tmp_path / "config" / "voice-input"
    share_dir = tmp_path / "share" / "voice-input"
    icons_dir = share_dir / "icons"

    config_dir.mkdir(parents=True)
    share_dir.mkdir(parents=True)
    icons_dir.mkdir(parents=True)

    # Create fake icon files
    for icon_name in ["mic-idle.svg", "mic-recording.svg", "mic-processing.svg"]:
        (icons_dir / icon_name).write_text("<svg></svg>")

    # Patch environment
    monkeypatch.setattr('voice_input.CONFIG_DIR', config_dir)
    monkeypatch.setattr('voice_input.PID_FILE', config_dir / "recording.pid")
    monkeypatch.setattr('voice_input.AUDIO_FILE', config_dir / "recording.wav")
    monkeypatch.setattr('voice_input.DAEMON_PID_FILE', config_dir / "daemon.pid")
    monkeypatch.setattr('voice_input.SOCKET_PATH', config_dir / "daemon.sock")
    monkeypatch.setattr('voice_input.MODEL_STATE_FILE', config_dir / "current_model.txt")

    yield {
        'config_dir': config_dir,
        'share_dir': share_dir,
        'icons_dir': icons_dir,
        'socket_path': config_dir / "daemon.sock",
        'pid_file': config_dir / "recording.pid",
        'daemon_pid_file': config_dir / "daemon.pid",
        'audio_file': config_dir / "recording.wav",
        'model_state_file': config_dir / "current_model.txt",
    }


# ============ E2E Test Fixtures ============

@pytest.fixture(scope="session")
def real_asr_model():
    """
    Load a real ASR model (for E2E tests).
    Session scope ensures it is loaded only once.
    """
    try:
        from funasr import AutoModel
        model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            device="cpu",
            disable_update=True,
        )
        yield model
    except ImportError:
        pytest.skip("FunASR not installed")


@pytest.fixture
def sample_audio_file(tmp_path):
    """
    Create a simple test audio file.
    Actually silent; used for testing workflow rather than recognition quality.
    """
    import wave
    import struct

    audio_path = tmp_path / "test_audio.wav"

    # Create 1 second of silence
    sample_rate = 16000
    duration = 1
    num_samples = sample_rate * duration

    with wave.open(str(audio_path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        # Silence data
        for _ in range(num_samples):
            wav_file.writeframes(struct.pack('<h', 0))

    yield audio_path


# ============ Performance Test Fixtures ============

@pytest.fixture
def timer():
    """Simple timer fixture."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer()


# ============ Test Markers ============

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "race: marks tests as race condition tests"
    )
    config.addinivalue_line(
        "markers", "real_model: marks tests that require real ASR model"
    )
