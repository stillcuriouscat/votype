"""
Test daemon singleton enforcement via fcntl.flock.

Verifies that:
1. Two daemons cannot run simultaneously (flock prevents it)
2. is_daemon_running() detects a held lock
3. Lock is released when process dies (no stale locks)
"""

import fcntl
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input


@pytest.fixture
def isolated_lock(tmp_path, monkeypatch):
    """Redirect daemon lock and PID files to tmp_path."""
    config_dir = tmp_path / "config" / "voice-input"
    config_dir.mkdir(parents=True)

    lock_file = config_dir / "daemon.lock"
    pid_file = config_dir / "daemon.pid"
    socket_path = config_dir / "daemon.sock"

    monkeypatch.setattr('voice_input.CONFIG_DIR', config_dir)
    monkeypatch.setattr('voice_input.DAEMON_LOCK_FILE', lock_file)
    monkeypatch.setattr('voice_input.DAEMON_PID_FILE', pid_file)
    monkeypatch.setattr('voice_input.SOCKET_PATH', socket_path)

    yield {
        'config_dir': config_dir,
        'lock_file': lock_file,
        'pid_file': pid_file,
    }


class TestDaemonSingleton:
    """Test flock-based singleton enforcement."""

    def test_second_daemon_exits_when_lock_held(self, isolated_lock):
        """ASRDaemon.run() should exit immediately if lock is already held."""
        lock_file = isolated_lock['lock_file']

        # Simulate first daemon holding the lock
        holder = open(lock_file, 'w')
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

        try:
            daemon = voice_input.ASRDaemon()
            # run() should detect the held lock and return without starting
            with patch('builtins.print') as mock_print:
                daemon.run()

            # Verify it printed the "already running" message
            printed = ' '.join(str(c) for c in mock_print.call_args_list)
            assert "already running" in printed.lower()

            # Verify PID file was NOT written (daemon didn't start)
            assert not isolated_lock['pid_file'].exists()
        finally:
            holder.close()

    def test_is_daemon_running_detects_held_lock(self, isolated_lock):
        """is_daemon_running() should return True when lock is held."""
        lock_file = isolated_lock['lock_file']

        # No lock held -> False
        assert voice_input.is_daemon_running() is False

        # Hold the lock
        holder = open(lock_file, 'w')
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

        try:
            assert voice_input.is_daemon_running() is True
        finally:
            holder.close()

    def test_lock_released_after_close(self, isolated_lock):
        """After lock holder closes fd, is_daemon_running() returns False."""
        lock_file = isolated_lock['lock_file']

        holder = open(lock_file, 'w')
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert voice_input._is_daemon_lock_held() is True

        holder.close()
        assert voice_input._is_daemon_lock_held() is False
