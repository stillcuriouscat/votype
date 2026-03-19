"""Tests for US-006: SSH ControlMaster docs and status post-processor display."""

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

README_PATH = Path(__file__).parent.parent / "README.md"


class TestReadmeSSHControlMaster:
    """README includes SSH ControlMaster documentation."""

    @pytest.fixture(autouse=True)
    def load_readme(self):
        self.readme = README_PATH.read_text(encoding="utf-8")

    def test_controlmaster_section_exists(self):
        assert "ControlMaster" in self.readme

    def test_controlmaster_auto(self):
        assert "ControlMaster auto" in self.readme

    def test_controlpath(self):
        assert "ControlPath" in self.readme

    def test_controlpersist_600(self):
        assert "ControlPersist 600" in self.readme

    def test_oracle_cloud_host(self):
        assert "oracle-cloud" in self.readme

    def test_does_not_auto_modify_ssh_config(self):
        """Docs should describe manual configuration, not auto-modification."""
        # The README should show the config as an example to add manually
        assert "~/.ssh/config" in self.readme

    def test_haiku_fix_documented(self):
        assert "haiku-fix" in self.readme


class TestStatusShowsPostProcessor:
    """voice-input status shows current post-processor name from DB."""

    def test_status_prints_post_processor_when_daemon_running(self, tmp_path, monkeypatch):
        import state_db as _state_db
        from voice_input import show_status

        db_path = tmp_path / "state.db"
        monkeypatch.setattr("voice_input.STATE_DB_PATH", db_path)
        monkeypatch.setattr("state_db.DEFAULT_DB_PATH", db_path)
        _state_db.init_db(db_path)
        _state_db.update_state(db_path, post_processor="haiku-fix")

        mock_model_response = {
            "model": "firered-asr",
            "name": "FireRed ASR",
            "description": "Chinese SOTA",
        }

        with patch("voice_input.is_daemon_running", return_value=True), \
             patch("voice_input.send_to_daemon", return_value=mock_model_response):

            captured = StringIO()
            with patch("sys.stdout", captured):
                show_status()

            output = captured.getvalue()

        assert "Post-processor:" in output
        assert "Haiku Fix (SSH)" in output
        assert "haiku-fix" in output
