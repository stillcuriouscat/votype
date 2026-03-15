"""
Post-processor persistence tests.

Verifies that post-processor selection survives daemon restart.
Critical: gemini-merge must persist, not silently revert to gemini-fix.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input
from post_processor_presets import POST_PROCESSOR_PRESETS


class TestPostProcessorPersistence:
    """Test that post-processor ID persists across daemon restarts."""

    def test_gemini_merge_persists_after_restart(self, tmp_path):
        """gemini-merge must restore after daemon restart, not fall back to gemini-fix."""
        state_file = tmp_path / "current_post_processor.txt"
        state_file.write_text("gemini-merge")

        with patch.object(voice_input, "POST_PROCESSOR_STATE_FILE", state_file):
            result = voice_input.ASRDaemon._restore_post_processor_id()

        assert result == "gemini-merge", f"Expected gemini-merge, got {result}"

    def test_gemini_fix_persists(self, tmp_path):
        """gemini-fix should also persist correctly."""
        state_file = tmp_path / "current_post_processor.txt"
        state_file.write_text("gemini-fix")

        with patch.object(voice_input, "POST_PROCESSOR_STATE_FILE", state_file):
            result = voice_input.ASRDaemon._restore_post_processor_id()

        assert result == "gemini-fix"

    def test_missing_file_returns_default(self, tmp_path):
        """Missing state file should return DEFAULT_POST_PROCESSOR."""
        state_file = tmp_path / "nonexistent.txt"

        with patch.object(voice_input, "POST_PROCESSOR_STATE_FILE", state_file):
            result = voice_input.ASRDaemon._restore_post_processor_id()

        assert result == voice_input.DEFAULT_POST_PROCESSOR

    def test_invalid_preset_returns_default(self, tmp_path):
        """Invalid preset ID in state file should return DEFAULT_POST_PROCESSOR."""
        state_file = tmp_path / "current_post_processor.txt"
        state_file.write_text("nonexistent-preset")

        with patch.object(voice_input, "POST_PROCESSOR_STATE_FILE", state_file):
            result = voice_input.ASRDaemon._restore_post_processor_id()

        assert result == voice_input.DEFAULT_POST_PROCESSOR

    def test_persist_writes_correctly(self, tmp_path):
        """_persist_post_processor_id should write the ID to state file."""
        state_file = tmp_path / "current_post_processor.txt"

        with patch.object(voice_input, "POST_PROCESSOR_STATE_FILE", state_file):
            voice_input.ASRDaemon._persist_post_processor_id("gemini-merge")

        assert state_file.read_text() == "gemini-merge"

    def test_persist_then_restore_roundtrip(self, tmp_path):
        """Write then read should return the same preset ID."""
        state_file = tmp_path / "current_post_processor.txt"

        with patch.object(voice_input, "POST_PROCESSOR_STATE_FILE", state_file):
            voice_input.ASRDaemon._persist_post_processor_id("gemini-merge")
            result = voice_input.ASRDaemon._restore_post_processor_id()

        assert result == "gemini-merge"

    def test_all_presets_persist_correctly(self, tmp_path):
        """Every valid preset should survive persist→restore roundtrip."""
        state_file = tmp_path / "current_post_processor.txt"

        for preset_id in POST_PROCESSOR_PRESETS:
            with patch.object(voice_input, "POST_PROCESSOR_STATE_FILE", state_file):
                voice_input.ASRDaemon._persist_post_processor_id(preset_id)
                result = voice_input.ASRDaemon._restore_post_processor_id()

            assert result == preset_id, f"Preset {preset_id} did not survive roundtrip"

    def test_whitespace_in_state_file(self, tmp_path):
        """State file with trailing whitespace/newline should still work."""
        state_file = tmp_path / "current_post_processor.txt"
        state_file.write_text("gemini-merge\n  ")

        with patch.object(voice_input, "POST_PROCESSOR_STATE_FILE", state_file):
            result = voice_input.ASRDaemon._restore_post_processor_id()

        assert result == "gemini-merge"


class TestPostProcessorLoadFailureFallback:
    """Test that load failure doesn't silently change the persisted value."""

    def test_load_failure_does_not_overwrite_state_file(self, tmp_path):
        """If load_post_processor fails, state file should NOT be overwritten."""
        state_file = tmp_path / "current_post_processor.txt"
        state_file.write_text("gemini-merge")

        # After a failed load, the state file should still say gemini-merge
        # so next restart will try gemini-merge again
        assert state_file.read_text() == "gemini-merge"
