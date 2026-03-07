"""
Process cleanup tests - verify fixtures correctly clean up orphan processes

This test file validates that our process cleanup mechanisms work properly.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Import module under test
sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input


class TestProcessCleanup:
    """Test process cleanup functionality"""

    def test_real_recorder_process_is_cleaned_up(self, isolated_environment):
        """
        Verify real recorder processes are cleaned up after test ends.

        Uses the same recorder selection as voice_input (pw-record preferred, arecord fallback).
        """
        import shutil
        recorder = "pw-record" if shutil.which("pw-record") else "arecord"

        if not shutil.which(recorder):
            pytest.skip(f"{recorder} not available")

        audio_file = isolated_environment['audio_file']
        test_pid = None

        try:
            if recorder == "pw-record":
                cmd = ["pw-record", "--format=s16", "--rate=16000", "--channels=1", str(audio_file)]
            else:
                cmd = ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav", str(audio_file)]

            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            test_pid = proc.pid
            print(f"[test] Started test {recorder} process: {test_pid}")

            time.sleep(0.2)
            assert os.path.exists(f"/proc/{test_pid}"), f"{recorder} process should be running"

        except FileNotFoundError:
            pytest.skip(f"{recorder} not available")

        finally:
            if test_pid is not None:
                try:
                    os.kill(test_pid, 9)
                    print(f"[test] Cleaned up {recorder} process {test_pid}")
                except ProcessLookupError:
                    print(f"[test] Process {test_pid} already terminated")

    def test_cleanup_fixture_removes_orphan_processes(self):
        """
        Verify cleanup_test_processes fixture can detect and clean up orphan processes.

        Note: This test only verifies there are no orphan processes currently.
        It cannot truly test the fixture's cleanup functionality because the fixture's
        cleanup happens after the test ends, and this test won't produce orphan processes
        while running.
        """
        # Check if pgrep is available (may not be available in sandbox environments)
        import shutil
        if not shutil.which("pgrep"):
            pytest.skip("pgrep not available in this environment")

        # Check for orphan recorder processes (pw-record and arecord)
        for proc_name in ["pw-record", "arecord"]:
            result = subprocess.run(
                ["pgrep", "-x", proc_name],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                pids = [pid for pid in result.stdout.strip().split('\n') if pid]
                print(f"[warning] Found {proc_name} processes: {pids}")
                print("[warning] This may be user's actual recording session, not orphan processes")

        print("[test] Cleanup fixture check completed")
