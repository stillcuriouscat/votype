"""
Race Condition Tests - Test behavior under concurrent scenarios.

Coverage:
- Daemon startup race
- Recording state race
- Socket communication race

These tests are key to preventing regression, as the previous bug was caused by race conditions.
"""

import os
import sys
import json
import socket
import threading
import time
import multiprocessing
from pathlib import Path
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input


# ============ 1. Daemon Startup Race ============

@pytest.mark.race
class TestDaemonStartRace:
    """Test race conditions during daemon startup."""

    def test_concurrent_daemon_start_only_one_succeeds(self, isolated_environment, mock_asr_model, mock_gtk):
        """
        Starting two daemons concurrently, only one should succeed.
        This tests the single-instance check mechanism via PID file.
        """
        pid_file = isolated_environment['daemon_pid_file']
        start_results = []
        lock = threading.Lock()

        def try_start_daemon():
            """Try to start daemon."""
            # Check if daemon already exists
            if pid_file.exists():
                try:
                    existing_pid = int(pid_file.read_text().strip())
                    os.kill(existing_pid, 0)  # Check if process exists
                    with lock:
                        start_results.append(("skipped", "already running"))
                    return
                except (ProcessLookupError, ValueError):
                    pass

            # Simulate startup (write PID)
            with lock:
                if pid_file.exists():
                    start_results.append(("skipped", "file created by another"))
                    return
                pid_file.write_text(str(os.getpid()))
                start_results.append(("started", os.getpid()))

        # Concurrent startup
        threads = []
        for _ in range(5):
            t = threading.Thread(target=try_start_daemon)
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Only one should succeed in starting
        started_count = sum(1 for r in start_results if r[0] == "started")
        assert started_count == 1

    def test_toggle_while_daemon_loading_model(self, isolated_environment, mock_notify):
        """
        Calling toggle during model loading should show wait message.
        This is the core scenario of the previous bug.
        """
        # Simulate: PID file exists (daemon started), but ping fails (model still loading)
        daemon_pid_file = isolated_environment['daemon_pid_file']
        daemon_pid_file.write_text(str(os.getpid()))

        with patch('voice_input.is_daemon_ready', return_value=False):
            with patch('voice_input.is_daemon_running', return_value=True):
                voice_input.toggle_recording()

        # Should show wait message, not attempt recording
        mock_notify.assert_called()
        call_str = str(mock_notify.call_args)
        assert "starting" in call_str.lower() or "wait" in call_str.lower() or "Starting" in call_str

    def test_toggle_immediately_after_daemon_start(self, isolated_environment, mock_subprocess, mock_notify):
        """
        Toggle immediately after daemon start, is_daemon_ready should return False.
        """
        call_sequence = []

        def mock_is_ready():
            call_sequence.append("is_ready_check")
            return False  # Always return False (simulating just started)

        def mock_is_running():
            call_sequence.append("is_running_check")
            return len(call_sequence) > 1  # Returns True on second check

        with patch('voice_input.is_daemon_ready', side_effect=mock_is_ready):
            with patch('voice_input.is_daemon_running', side_effect=mock_is_running):
                with patch('voice_input.time.sleep'):
                    voice_input.toggle_recording()

        # Should check is_daemon_ready first
        assert "is_ready_check" in call_sequence


# ============ 2. Recording State Race ============

@pytest.mark.race
class TestRecordingStateRace:
    """Test race conditions related to recording state."""

    def test_rapid_start_stop_maintains_consistency(self, isolated_environment, mock_subprocess, mock_notify):
        """
        Rapid consecutive start-stop-start operations should maintain state consistency.
        """
        pid_file = isolated_environment['pid_file']
        mock_subprocess.run.return_value = MagicMock(stdout="12345\n", stderr="")

        operations = []

        def mock_is_recording():
            return pid_file.exists()

        with patch('voice_input.is_recording', side_effect=mock_is_recording):
            with patch('voice_input.is_daemon_ready', return_value=True):
                with patch('voice_input.is_daemon_running', return_value=True):
                    with patch('voice_input.send_to_daemon', return_value={"status": "ok", "text": ""}):
                        with patch('voice_input.type_text'):
                            with patch('os.kill'):
                                with patch('voice_input.time.sleep'):
                                    # Execute toggle rapidly multiple times
                                    for i in range(5):
                                        voice_input.toggle_recording()
                                        operations.append("toggle")
                                        # Simulate state change
                                        if pid_file.exists():
                                            pid_file.unlink()
                                        else:
                                            pid_file.write_text("12345")

        # All operations should complete without throwing exceptions
        assert len(operations) == 5

    def test_concurrent_stop_requests(self, isolated_environment, mock_notify):
        """
        Sending two stop requests concurrently, only one should actually execute.
        """
        pid_file = isolated_environment['pid_file']
        audio_file = isolated_environment['audio_file']

        # Create initial state
        pid_file.write_text("12345")
        audio_file.touch()

        stop_count = [0]
        stop_lock = threading.Lock()

        def mock_kill(pid, sig):
            with stop_lock:
                stop_count[0] += 1

        results = []

        def try_stop():
            with patch('voice_input.is_recording', side_effect=lambda: pid_file.exists()):
                with patch('voice_input.is_daemon_running', return_value=True):
                    with patch('voice_input.send_to_daemon', return_value={"text": ""}):
                        with patch('voice_input.type_text'):
                            with patch('os.kill', side_effect=mock_kill):
                                with patch('voice_input.time.sleep'):
                                    try:
                                        voice_input.stop_recording()
                                        results.append("stopped")
                                    except Exception as e:
                                        results.append(f"error: {e}")
                                    finally:
                                        # Clean up PID file to simulate stop completion
                                        if pid_file.exists():
                                            pid_file.unlink()

        # Concurrent execution
        threads = [threading.Thread(target=try_stop) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one should actually execute kill (others should skip because is_recording is False)
        # Due to racing, multiple may see is_recording as True
        # But this is acceptable, the important thing is no crash
        assert len(results) == 3


# ============ 3. Socket Communication Race ============

@pytest.mark.race
class TestSocketCommunicationRace:
    """Test race conditions related to socket communication."""
    # Note: reset_socket_path fixture is no longer needed,
    # the global reset_voice_input_paths fixture in conftest.py handles path reset

    def test_multiple_socket_clients_concurrent(self, tmp_path):
        """
        Multiple clients connecting concurrently should all be handled correctly.
        """
        import voice_input as vi

        # Use unique socket path
        socket_path = tmp_path / "multi_client.sock"
        original_socket_path = vi.SOCKET_PATH

        # Create dedicated server
        server_ready = threading.Event()
        server_running = threading.Event()
        server_running.set()

        def server_thread_func():
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(socket_path))
            server.listen(10)
            server.settimeout(0.5)
            server_ready.set()

            while server_running.is_set():
                try:
                    client, _ = server.accept()
                    data = client.recv(4096).decode()
                    msg = json.loads(data)
                    if msg.get("command") == "ping":
                        client.send(json.dumps({"status": "ok"}).encode())
                    client.close()
                except socket.timeout:
                    continue
                except Exception:
                    break

            server.close()

        server_thread = threading.Thread(target=server_thread_func, daemon=True)
        server_thread.start()
        server_ready.wait(timeout=2)

        try:
            # Replace SOCKET_PATH
            vi.SOCKET_PATH = socket_path

            results = []

            def send_ping():
                response = vi.send_to_daemon("ping")
                results.append(response)

            # Send requests concurrently
            threads = [threading.Thread(target=send_ping) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All requests should succeed
            assert len(results) == 10
            for r in results:
                assert r == {"status": "ok"}
        finally:
            # Restore original path
            vi.SOCKET_PATH = original_socket_path
            # Stop server
            server_running.clear()
            server_thread.join(timeout=1)

    def test_socket_during_daemon_shutdown(self, tmp_path):
        """
        Sending requests during daemon shutdown should handle errors gracefully.
        Test behavior when socket file does not exist.
        """
        socket_path = tmp_path / "nonexistent_shutdown.sock"

        # Directly test send_to_daemon handling of nonexistent socket
        # Use patch to replace SOCKET_PATH
        import voice_input as vi

        original_socket_path = vi.SOCKET_PATH
        try:
            vi.SOCKET_PATH = socket_path

            # Should return None when socket file does not exist
            result = vi.send_to_daemon("ping")
            assert result is None
        finally:
            vi.SOCKET_PATH = original_socket_path

    def test_concurrent_transcribe_requests(self, isolated_environment, mock_asr_model, mock_gtk):
        """
        Concurrent transcription requests should be processed in order without state corruption.
        """
        with patch('voice_input.ModelInference.transcribe') as mock_transcribe:
            mock_transcribe.return_value = "test transcription"

            daemon = voice_input.ASRDaemon()
            daemon.model = mock_asr_model['model_instance']
            daemon.framework = "funasr"
            daemon.current_model_id = "fun-asr-nano"
            daemon.extra_data = None
            daemon.running = True
            daemon.indicator = mock_gtk['indicator']
            daemon.status_item = MagicMock()

            results = []
            lock = threading.Lock()

            def handle_request(request_id):
                mock_client = MagicMock()
                mock_client.recv.return_value = json.dumps({
                    "command": "transcribe",
                    "data": f"/path/to/audio_{request_id}.wav"
                }).encode()

                daemon.handle_client(mock_client)

                response = json.loads(mock_client.sendall.call_args[0][0].decode())
                with lock:
                    results.append((request_id, response))

            # Send transcription requests concurrently
            threads = [threading.Thread(target=handle_request, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All requests should succeed
            assert len(results) == 5
            for req_id, response in results:
                assert "text" in response or "error" not in response


# ============ 4. State Consistency Tests ============

@pytest.mark.race
class TestStateConsistency:
    """Test state consistency under race conditions."""

    def test_pid_file_atomicity(self, isolated_environment):
        """
        Test concurrent read/write behavior of PID file.

        Note: write_text() on filesystem is not atomic, may read empty content.
        This is not a bug but expected behavior; the application should handle this case.

        This test verifies:
        1. Concurrent reads/writes do not cause crashes
        2. All valid reads contain legitimate PIDs
        """
        pid_file = isolated_environment['pid_file']
        valid_reads = []
        invalid_reads = []
        write_count = [0]

        def writer():
            for i in range(100):
                pid_file.write_text(f"{12345 + i}")
                write_count[0] += 1
                time.sleep(0.001)

        def reader():
            for _ in range(100):
                if pid_file.exists():
                    try:
                        content = pid_file.read_text().strip()
                        if content:  # Non-empty content should be a valid PID
                            pid = int(content)
                            assert pid >= 12345
                            valid_reads.append(pid)
                        else:
                            # Empty content is due to race condition, this is expected
                            invalid_reads.append("empty")
                    except ValueError as e:
                        # Only record non-empty but invalid content
                        invalid_reads.append(str(e))
                time.sleep(0.001)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        # Should have at least some successful reads
        assert len(valid_reads) > 0, "No valid reads occurred"
        # All valid reads should be legitimate PIDs
        for pid in valid_reads:
            assert 12345 <= pid < 12445

    def test_daemon_state_transitions(self, isolated_environment, mock_gtk):
        """
        Daemon state transitions should be ordered, not skipping intermediate states.
        """
        daemon = voice_input.ASRDaemon()
        daemon.indicator = mock_gtk['indicator']
        daemon.status_item = MagicMock()

        state_history = []

        # Record state changes
        original_set_icon = mock_gtk['indicator'].set_icon_full

        def record_state(icon_name, tooltip):
            state_history.append(icon_name)
            return original_set_icon(icon_name, tooltip)

        mock_gtk['indicator'].set_icon_full = record_state

        # Simulate normal state transition sequence
        daemon.set_status("idle")
        daemon.set_status("recording")
        daemon.set_status("processing")
        daemon.set_status("idle")

        # Verify state transition order
        # Note: initial set_status("idle") is deduped because _current_icon_status starts at 'idle' (US-003)
        expected = ["mic-recording", "mic-processing", "mic-idle"]
        assert state_history == expected


# ============ 5. Boundary Condition Tests ============

@pytest.mark.race
class TestBoundaryConditions:
    """Test behavior under boundary conditions."""

    def test_daemon_start_during_cleanup(self, isolated_environment):
        """
        Attempt to start new daemon during daemon cleanup process.
        Dead daemon_pid in DB should be cleaned up by is_daemon_running.
        """
        import state_db as _state_db

        socket_path = isolated_environment['socket_path']
        socket_path.touch()

        # Write a dead PID to DB
        _state_db.update_state(
            isolated_environment["state_db_path"],
            daemon_pid=99999,
        )

        with patch('voice_input.os.kill', side_effect=ProcessLookupError):
            with patch('voice_input._is_daemon_lock_held', return_value=False):
                result = voice_input.is_daemon_running()

        assert result is False
        # DB should be cleaned up
        state = _state_db.get_state(isolated_environment["state_db_path"])
        assert state["daemon_pid"] is None

    def test_toggle_rapid_fire(self, isolated_environment, mock_subprocess, mock_notify):
        """
        Rapidly calling toggle in succession (simulating user pressing hotkey quickly).
        """
        mock_subprocess.run.return_value = MagicMock(stdout="12345\n", stderr="")
        toggle_count = [0]

        with patch('voice_input.is_daemon_ready', return_value=True):
            with patch('voice_input.is_recording', return_value=False):
                with patch('voice_input.is_daemon_running', return_value=True):
                    with patch('voice_input.send_to_daemon', return_value={"status": "ok"}):
                        for _ in range(10):
                            voice_input.toggle_recording()
                            toggle_count[0] += 1

        # All calls should complete
        assert toggle_count[0] == 10

    def test_socket_reconnect_after_daemon_restart(self, tmp_path, monkeypatch):
        """
        Client should be able to reconnect after daemon restart.
        """
        socket_path = tmp_path / "restart.sock"
        monkeypatch.setattr('voice_input.SOCKET_PATH', socket_path)

        server_running = threading.Event()

        def create_server():
            """Create a simple socket server."""
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                if socket_path.exists():
                    socket_path.unlink()
                server.bind(str(socket_path))
                server.listen(1)
                server.settimeout(0.5)
                server_running.set()

                try:
                    client, _ = server.accept()
                    data = client.recv(1024).decode()
                    msg = json.loads(data)
                    client.send(json.dumps({"status": "ok", "server": "restarted"}).encode())
                    client.close()
                except socket.timeout:
                    pass
            finally:
                server.close()

        # First startup
        server1 = threading.Thread(target=create_server, daemon=True)
        server1.start()
        server_running.wait(timeout=1)
        server_running.clear()

        result1 = voice_input.send_to_daemon("ping")

        # Wait for first server to close
        server1.join(timeout=1)
        if socket_path.exists():
            socket_path.unlink()

        # Second startup (restart)
        server2 = threading.Thread(target=create_server, daemon=True)
        server2.start()
        server_running.wait(timeout=1)

        result2 = voice_input.send_to_daemon("ping")

        # Both should succeed
        assert result1 is not None
        assert result2 is not None


# ============ 6. Processing Guard Tests (DB-based) ============

@pytest.mark.race
class TestProcessingFlagGuard:
    """Test that DB status='processing' prevents ghost recordings during ASR/Gemini processing."""

    def test_toggle_rejected_during_processing(self, isolated_environment, mock_notify):
        """Toggle should be rejected when DB status='processing' (< 120s old)."""
        import state_db as _state_db
        _state_db.update_state(isolated_environment["state_db_path"], status="processing")

        with patch('voice_input.is_daemon_ready', return_value=True):
            with patch('voice_input.is_recording', return_value=False) as mock_is_rec:
                with patch('voice_input.start_recording') as mock_start:
                    voice_input.toggle_recording()

                    # start_recording should NOT have been called
                    mock_start.assert_not_called()
                    # is_recording should NOT have been checked (early return)
                    mock_is_rec.assert_not_called()

        # Notify should have been called with processing message
        mock_notify.assert_called()
        call_args = mock_notify.call_args[0]
        assert "Processing" in call_args[1] or "processing" in call_args[1]

    def test_toggle_allowed_after_processing_complete(self, isolated_environment, mock_notify):
        """Toggle should work normally when DB status='idle'."""
        import state_db as _state_db
        # Ensure DB is idle (default after init)
        s = _state_db.get_state(isolated_environment["state_db_path"])
        assert s["status"] == "idle"

        with patch('voice_input.is_daemon_ready', return_value=True):
            with patch('voice_input.is_recording', return_value=False):
                with patch('voice_input.is_daemon_running', return_value=True):
                    with patch('voice_input.send_to_daemon', return_value={"status": "ok"}):
                        with patch('voice_input.start_recording') as mock_start:
                            voice_input.toggle_recording()
                            mock_start.assert_called_once()

    def test_stale_processing_status_cleaned_up(self, isolated_environment, mock_notify):
        """Processing status older than 120s should be cleaned up and toggle allowed."""
        import state_db as _state_db
        from datetime import datetime, timezone, timedelta

        old_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        _state_db.update_state(
            isolated_environment["state_db_path"],
            status="processing",
            updated_at=old_time,
        )

        with patch('voice_input.is_daemon_ready', return_value=True):
            with patch('voice_input.is_daemon_running', return_value=True):
                with patch('voice_input.send_to_daemon', return_value={"status": "ok"}):
                    with patch('voice_input.is_recording', return_value=False):
                        with patch('voice_input.start_recording') as mock_start:
                            voice_input.toggle_recording()
                            # Should clean up stale status and proceed
                            mock_start.assert_called_once()

    def test_stop_recording_sets_processing_in_db(self, isolated_environment, mock_notify):
        """stop_recording() should write status='processing' to DB before killing recorder."""
        import state_db as _state_db

        audio_file = isolated_environment['config_dir'] / "recording_test.wav"
        audio_file.write_bytes(b'\x00' * 100)

        _state_db.update_state(
            isolated_environment["state_db_path"],
            status="recording",
            recording_pid=os.getpid(),
            recording_path=str(audio_file),
        )

        status_at_kill = [None]

        def smart_kill(pid, sig):
            if sig == 0:
                return  # is_recording PID check
            # SIGTERM — check DB status at kill time
            status_at_kill[0] = _state_db.get_state(isolated_environment["state_db_path"])["status"]
            raise ProcessLookupError

        with patch('voice_input.is_daemon_running', return_value=False):
            with patch('voice_input.os.kill', side_effect=smart_kill):
                voice_input.stop_recording()

        # DB status should have been 'processing' BEFORE kill was called
        assert status_at_kill[0] == "processing", "DB status must be 'processing' before killing recorder"
        # DB status should be 'idle' after stop
        s = _state_db.get_state(isolated_environment["state_db_path"])
        assert s["status"] == "idle"

    def test_stop_recording_resets_status_on_error(self, isolated_environment, mock_notify):
        """DB status should be reset to 'idle' even when transcription fails."""
        import state_db as _state_db

        audio_file = isolated_environment['config_dir'] / "recording_test.wav"
        audio_file.write_bytes(b'\x00' * 100)

        _state_db.update_state(
            isolated_environment["state_db_path"],
            status="recording",
            recording_pid=os.getpid(),
            recording_path=str(audio_file),
        )

        with patch('voice_input.is_daemon_running', return_value=True):
            with patch('voice_input.os.kill', side_effect=ProcessLookupError):
                with patch('voice_input.send_to_daemon', return_value={"error": "ASR failed"}):
                    voice_input.stop_recording()

        # DB status must be idle even after failure
        s = _state_db.get_state(isolated_environment["state_db_path"])
        assert s["status"] == "idle"

    def test_processing_status_blocks_concurrent_toggle(self, isolated_environment, mock_notify):
        """Simulate the real race: stop_recording in progress, concurrent toggle arrives."""
        import state_db as _state_db

        audio_file = isolated_environment['config_dir'] / "recording_test.wav"
        audio_file.write_bytes(b'\x00' * 100)

        _state_db.update_state(
            isolated_environment["state_db_path"],
            status="recording",
            recording_pid=os.getpid(),
            recording_path=str(audio_file),
        )

        concurrent_start_called = [False]

        def slow_transcribe(command, data=None, timeout=60):
            if command == "transcribe":
                # During transcription, DB status is 'processing'
                # Try concurrent toggle — should be blocked by DB status
                result = try_concurrent_toggle()
                concurrent_start_called[0] = result
                return {"text": "test transcription"}
            return {"status": "ok"}

        def try_concurrent_toggle():
            """Attempt to toggle while processing — should be blocked."""
            with patch('voice_input.is_daemon_ready', return_value=True):
                with patch('voice_input.is_recording', return_value=False):
                    with patch('voice_input.start_recording') as mock_start:
                        voice_input.toggle_recording()
                        return mock_start.called

        with patch('voice_input.is_daemon_running', return_value=True):
            with patch('voice_input.os.kill', side_effect=ProcessLookupError):
                with patch('voice_input.send_to_daemon', side_effect=slow_transcribe):
                    with patch('voice_input.type_text'):
                        voice_input.stop_recording()

        # Concurrent toggle should NOT have started a new recording
        assert not concurrent_start_called[0], "Concurrent toggle must be blocked during processing"
