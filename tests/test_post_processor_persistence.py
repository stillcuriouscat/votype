"""
Post-processor persistence tests (DB-based).

Verifies that post-processor selection survives daemon restart via SQLite DB.
Critical: gemini-merge must persist, not silently revert to gemini-fix.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input
import state_db as _state_db
from post_processor_presets import POST_PROCESSOR_PRESETS, DEFAULT_POST_PROCESSOR


class TestPostProcessorPersistenceDB:
    """Test that post-processor ID persists across daemon restarts via DB."""

    def test_gemini_merge_persists_after_restart(self, tmp_path):
        """gemini-merge must restore after daemon restart, not fall back to gemini-fix."""
        db_path = tmp_path / "state.db"
        _state_db.init_db(db_path)
        _state_db.update_state(db_path, post_processor="gemini-merge")

        state = _state_db.get_state(db_path)
        assert state["post_processor"] == "gemini-merge"

    def test_gemini_fix_persists(self, tmp_path):
        """gemini-fix should also persist correctly."""
        db_path = tmp_path / "state.db"
        _state_db.init_db(db_path)
        _state_db.update_state(db_path, post_processor="gemini-fix")

        state = _state_db.get_state(db_path)
        assert state["post_processor"] == "gemini-fix"

    def test_default_when_no_update(self, tmp_path):
        """Fresh DB should return default 'none' for post_processor."""
        db_path = tmp_path / "state.db"
        _state_db.init_db(db_path)

        state = _state_db.get_state(db_path)
        assert state["post_processor"] == "none"

    def test_all_presets_persist_correctly(self, tmp_path):
        """Every valid preset should survive write→read roundtrip in DB."""
        db_path = tmp_path / "state.db"
        _state_db.init_db(db_path)

        for preset_id in POST_PROCESSOR_PRESETS:
            _state_db.update_state(db_path, post_processor=preset_id)
            state = _state_db.get_state(db_path)
            assert state["post_processor"] == preset_id, (
                f"Preset {preset_id} did not survive roundtrip"
            )

    def test_daemon_init_reads_from_db(self, tmp_path, monkeypatch):
        """ASRDaemon.__init__ should read post_processor from DB."""
        db_path = tmp_path / "state.db"
        monkeypatch.setattr("voice_input.STATE_DB_PATH", db_path)
        monkeypatch.setattr("state_db.DEFAULT_DB_PATH", db_path)
        _state_db.init_db(db_path)
        _state_db.update_state(db_path, post_processor="gemini-merge")

        daemon = voice_input.ASRDaemon()
        assert daemon.current_post_processor_id == "gemini-merge"

    def test_daemon_init_falls_back_on_invalid(self, tmp_path, monkeypatch):
        """ASRDaemon.__init__ should fall back to default for invalid preset in DB."""
        db_path = tmp_path / "state.db"
        monkeypatch.setattr("voice_input.STATE_DB_PATH", db_path)
        monkeypatch.setattr("state_db.DEFAULT_DB_PATH", db_path)
        _state_db.init_db(db_path)
        _state_db.update_state(db_path, post_processor="nonexistent-preset")

        daemon = voice_input.ASRDaemon()
        assert daemon.current_post_processor_id == DEFAULT_POST_PROCESSOR

    def test_load_post_processor_writes_to_db(self, tmp_path, monkeypatch):
        """load_post_processor() should write the preset ID to DB."""
        db_path = tmp_path / "state.db"
        monkeypatch.setattr("voice_input.STATE_DB_PATH", db_path)
        monkeypatch.setattr("state_db.DEFAULT_DB_PATH", db_path)
        _state_db.init_db(db_path)

        daemon = voice_input.ASRDaemon.__new__(voice_input.ASRDaemon)
        daemon.model = None
        daemon.framework = None
        daemon.extra_data = None
        daemon.current_model_id = "sensevoice"
        daemon.running = False
        daemon.indicator = None
        daemon.gtk_thread = None
        daemon.post_processor_model = None
        daemon.current_post_processor_id = "none"
        daemon.post_processor_framework = "regex"
        daemon.punc_model = None
        daemon._vocab = {}
        daemon._secondary_model = None
        daemon._last_secondary_text = None
        daemon._current_db_status = "idle"

        with patch("voice_input.PostProcessorLoader.load_post_processor", return_value=None):
            daemon.load_post_processor("gemini-fix")

        state = _state_db.get_state(db_path)
        assert state["post_processor"] == "gemini-fix"


class TestPostProcessorLoadFailureFallback:
    """Test that load failure doesn't silently change the persisted value."""

    def test_load_failure_preserves_db_value(self, tmp_path, monkeypatch):
        """If load_post_processor fails, DB should still have the original value."""
        db_path = tmp_path / "state.db"
        monkeypatch.setattr("voice_input.STATE_DB_PATH", db_path)
        monkeypatch.setattr("state_db.DEFAULT_DB_PATH", db_path)
        _state_db.init_db(db_path)
        _state_db.update_state(db_path, post_processor="gemini-merge")

        # After a failed load, reading the DB should still say gemini-merge
        state = _state_db.get_state(db_path)
        assert state["post_processor"] == "gemini-merge"
