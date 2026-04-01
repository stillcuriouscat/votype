"""Unit tests for icon status fix — US-001, US-002, US-003.

Verifies:
- _post_process() does NOT revert status to 'processing' after polishing
- _handle_transcribe() does NOT redundantly call set_status('processing')
- set_status() deduplicates redundant icon updates
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============ Fixtures ============


@pytest.fixture
def pp_env(tmp_path, monkeypatch):
    """Isolated environment for _post_process tests."""
    from state_db import init_db

    config_dir = tmp_path / ".config" / "voice-input"
    config_dir.mkdir(parents=True)
    state_db_path = config_dir / "state.db"

    monkeypatch.setattr("state_db.DEFAULT_DB_PATH", state_db_path)
    monkeypatch.setattr("voice_input.STATE_DB_PATH", state_db_path)
    monkeypatch.setattr("voice_input.CONFIG_DIR", config_dir)

    init_db(state_db_path)

    return {"state_db_path": state_db_path, "config_dir": config_dir}


# ============ US-001: No post-polishing status revert ============


class TestPostProcessNoStatusRevert:
    """Verify _post_process() never sets status='processing' after polishing."""

    def test_update_state_called_with_polishing_not_processing(self, pp_env, monkeypatch):
        """update_state is called with status='polishing' but NEVER with status='processing'."""
        from voice_input import ASRDaemon

        daemon = MagicMock(spec=ASRDaemon)
        daemon.current_post_processor_id = "gemini-fix"
        daemon.post_processor_framework = "vertex-ai"
        daemon.post_processor_model = None
        daemon.punc_model = None
        daemon._vocab = {}

        with patch("post_processor_configs.process_with_vertex_ai", return_value="polished text"), \
             patch("post_processor_configs.apply_vocab", return_value="input text"), \
             patch("post_processor_configs.glossary_context", return_value=""), \
             patch("post_processor_configs.load_vocab", return_value={}), \
             patch("post_processor_configs.diff_to_vocab", return_value={}), \
             patch("post_processor_configs.save_vocab"), \
             patch("state_db.update_state") as mock_update_state:

            monkeypatch.setattr("voice_input.POST_PROCESSOR_PRESETS", {
                "gemini-fix": {"framework": "vertex-ai", "config": {}},
            })

            # Call real _post_process on our mock
            ASRDaemon._post_process(daemon, "input text")

        # update_state should have been called with 'polishing'
        mock_update_state.assert_any_call(status="polishing")

        # update_state should NEVER be called with 'processing'
        for c in mock_update_state.call_args_list:
            assert c != call(status="processing"), \
                "update_state(status='processing') must not be called in _post_process after polishing"

    def test_status_stays_polishing_after_llm_call(self, pp_env, monkeypatch):
        """After polishing LLM call, the last status written is 'polishing', not 'processing'."""
        from voice_input import ASRDaemon

        daemon = MagicMock(spec=ASRDaemon)
        daemon.current_post_processor_id = "gemini-fix"
        daemon.post_processor_framework = "vertex-ai"
        daemon.post_processor_model = None
        daemon.punc_model = None
        daemon._vocab = {}

        status_calls = []

        def track_update_state(**kwargs):
            status_calls.append(kwargs.get("status"))

        with patch("post_processor_configs.process_with_vertex_ai", return_value="same text"), \
             patch("post_processor_configs.apply_vocab", return_value="same text"), \
             patch("post_processor_configs.glossary_context", return_value=""), \
             patch("state_db.update_state", side_effect=track_update_state):

            monkeypatch.setattr("voice_input.POST_PROCESSOR_PRESETS", {
                "gemini-fix": {"framework": "vertex-ai", "config": {}},
            })

            ASRDaemon._post_process(daemon, "same text")

        # Only 'polishing' should appear, never 'processing'
        assert "polishing" in status_calls
        assert "processing" not in status_calls


# ============ US-002: No redundant set_status('processing') in _handle_transcribe ============


class TestHandleTranscribeNoRedundantStatus:
    """Verify _handle_transcribe() does NOT call set_status('processing')."""

    def test_set_status_not_called_with_processing(self, pp_env, monkeypatch):
        """_handle_transcribe does not call self.set_status('processing') at entry."""
        from voice_input import ASRDaemon

        daemon = MagicMock(spec=ASRDaemon)
        daemon._secondary_model = None
        daemon._last_secondary_text = None
        daemon.current_post_processor_id = "none"
        daemon.post_processor_framework = "regex"
        daemon.post_processor_model = None
        daemon.punc_model = None
        daemon._vocab = {}

        # transcribe returns a simple result
        daemon.transcribe.return_value = {"text": "hello world"}

        # _post_process returns the text as-is
        daemon._post_process.return_value = "hello world"

        msg = {"data": "/tmp/test_audio.wav"}

        ASRDaemon._handle_transcribe(daemon, msg)

        # set_status should NOT have been called with 'processing'
        for c in daemon.set_status.call_args_list:
            assert c != call("processing"), \
                "set_status('processing') should not be called in _handle_transcribe — CLI handles this via DB"

    def test_handle_transcribe_relies_on_db_status(self, pp_env, monkeypatch):
        """_handle_transcribe does not directly set any status — relies on DB-driven status."""
        from voice_input import ASRDaemon

        daemon = MagicMock(spec=ASRDaemon)
        daemon._secondary_model = None
        daemon._last_secondary_text = None
        daemon.current_post_processor_id = "none"
        daemon.post_processor_framework = "regex"
        daemon.post_processor_model = None
        daemon.punc_model = None
        daemon._vocab = {}
        daemon.transcribe.return_value = {"text": "test"}
        daemon._post_process.return_value = "test"

        msg = {"data": "/tmp/test.wav"}
        ASRDaemon._handle_transcribe(daemon, msg)

        # set_status should not be called at all in _handle_transcribe
        daemon.set_status.assert_not_called()
