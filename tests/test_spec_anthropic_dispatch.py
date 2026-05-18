"""Clean-room unit tests for US-004: anthropic dispatch in voice_input.ASRDaemon.

Derived from FUNCTION_SPEC.md Module D behavior tables.
Tests:
- load_post_processor: vocab loading, secondary model idempotency, merge↔fix transitions.
- _post_process: dispatch routing for anthropic / anthropic-merge.

These tests construct ASRDaemon without triggering __init__ side effects by
using object.__new__ and patching only the attributes the methods read/write.
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daemon():
    """Build an ASRDaemon-like stand-in without invoking __init__."""
    import voice_input as vi
    daemon = object.__new__(vi.ASRDaemon)
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
    daemon._current_icon_status = "idle"
    return daemon


# ===========================================================================
# load_post_processor — secondary-model idempotency & framework routing
# ===========================================================================

class TestLoadPostProcessorAnthropic:
    def test_load_claude_fix_loads_vocab_no_secondary(self):
        daemon = _make_daemon()
        with patch("voice_input.PostProcessorLoader") as mock_ldr, \
             patch("voice_input.update_state"), \
             patch("post_processor_configs.load_vocab", return_value={"term": {"variants": {}}}), \
             patch.object(daemon, "_load_secondary_model") as mock_load_sec, \
             patch.object(daemon, "_unload_secondary_model") as mock_unload_sec:
            mock_ldr.load_post_processor.return_value = None
            daemon.load_post_processor("claude-fix")
        assert daemon.current_post_processor_id == "claude-fix"
        assert daemon.post_processor_framework == "anthropic"
        assert isinstance(daemon._vocab, dict)
        assert "term" in daemon._vocab
        mock_load_sec.assert_not_called()
        mock_unload_sec.assert_called_once()

    def test_load_claude_merge_cold_loads_secondary(self):
        daemon = _make_daemon()
        daemon._secondary_model = None
        with patch("voice_input.PostProcessorLoader") as mock_ldr, \
             patch("voice_input.update_state"), \
             patch("post_processor_configs.load_vocab", return_value={}), \
             patch.object(daemon, "_load_secondary_model") as mock_load_sec, \
             patch.object(daemon, "_unload_secondary_model") as mock_unload_sec:
            mock_ldr.load_post_processor.return_value = None
            daemon.load_post_processor("claude-merge")
        assert daemon.current_post_processor_id == "claude-merge"
        assert daemon.post_processor_framework == "anthropic-merge"
        mock_load_sec.assert_called_once()
        mock_unload_sec.assert_not_called()

    def test_switch_gemini_merge_to_claude_merge_does_not_reload_secondary(self):
        """Critical US-004 contract — no unload+reload between merge frameworks."""
        daemon = _make_daemon()
        with patch("voice_input.PostProcessorLoader") as mock_ldr, \
             patch("voice_input.update_state"), \
             patch("post_processor_configs.load_vocab", return_value={}), \
             patch.object(daemon, "_load_secondary_model") as mock_load_sec, \
             patch.object(daemon, "_unload_secondary_model") as mock_unload_sec:
            mock_ldr.load_post_processor.return_value = None

            def _set_loaded():
                daemon._secondary_model = MagicMock()
            mock_load_sec.side_effect = _set_loaded

            daemon.load_post_processor("gemini-merge")
            daemon.load_post_processor("claude-merge")

        assert daemon.post_processor_framework == "anthropic-merge"
        assert mock_load_sec.call_count == 1
        mock_unload_sec.assert_not_called()

    def test_switch_claude_merge_to_gemini_merge_does_not_reload_secondary(self):
        daemon = _make_daemon()
        with patch("voice_input.PostProcessorLoader") as mock_ldr, \
             patch("voice_input.update_state"), \
             patch("post_processor_configs.load_vocab", return_value={}), \
             patch.object(daemon, "_load_secondary_model") as mock_load_sec, \
             patch.object(daemon, "_unload_secondary_model") as mock_unload_sec:
            mock_ldr.load_post_processor.return_value = None

            def _set_loaded():
                daemon._secondary_model = MagicMock()
            mock_load_sec.side_effect = _set_loaded

            daemon.load_post_processor("claude-merge")
            daemon.load_post_processor("gemini-merge")

        assert daemon.post_processor_framework == "vertex-ai-merge"
        assert mock_load_sec.call_count == 1
        mock_unload_sec.assert_not_called()

    def test_switch_claude_merge_to_claude_fix_unloads_secondary(self):
        daemon = _make_daemon()
        with patch("voice_input.PostProcessorLoader") as mock_ldr, \
             patch("voice_input.update_state"), \
             patch("post_processor_configs.load_vocab", return_value={}), \
             patch.object(daemon, "_load_secondary_model") as mock_load_sec, \
             patch.object(daemon, "_unload_secondary_model") as mock_unload_sec:
            mock_ldr.load_post_processor.return_value = None

            def _set_loaded():
                daemon._secondary_model = MagicMock()
            mock_load_sec.side_effect = _set_loaded

            daemon.load_post_processor("claude-merge")
            daemon.load_post_processor("claude-fix")

        assert daemon.post_processor_framework == "anthropic"
        assert mock_load_sec.call_count == 1
        mock_unload_sec.assert_called()

    def test_switch_vertex_ai_to_anthropic_no_secondary_load(self):
        daemon = _make_daemon()
        with patch("voice_input.PostProcessorLoader") as mock_ldr, \
             patch("voice_input.update_state"), \
             patch("post_processor_configs.load_vocab", return_value={}), \
             patch.object(daemon, "_load_secondary_model") as mock_load_sec, \
             patch.object(daemon, "_unload_secondary_model") as mock_unload_sec:
            mock_ldr.load_post_processor.return_value = None
            daemon.load_post_processor("gemini-fix")
            daemon.load_post_processor("claude-fix")
        assert mock_load_sec.call_count == 0
        assert mock_unload_sec.call_count == 2

    def test_unknown_preset_raises(self):
        daemon = _make_daemon()
        with pytest.raises(RuntimeError, match="Unknown post-processor"):
            daemon.load_post_processor("nonexistent-preset")


# ===========================================================================
# _post_process — dispatch routing
# ===========================================================================

class TestPostProcessDispatch:
    def test_dispatch_anthropic_calls_process_with_anthropic(self):
        daemon = _make_daemon()
        daemon.current_post_processor_id = "claude-fix"
        daemon.post_processor_framework = "anthropic"
        daemon._vocab = {}
        with patch("post_processor_configs.process_with_anthropic",
                   return_value="polished") as mock_fix, \
             patch("post_processor_configs.process_with_anthropic_merge") as mock_merge, \
             patch("post_processor_configs.process_with_vertex_ai") as mock_v, \
             patch("post_processor_configs.process_with_gemini_merge") as mock_gm, \
             patch("post_processor_configs.process_with_ssh_claude") as mock_sc, \
             patch("state_db.update_state"):
            result = daemon._post_process("呃这是一段足够长的测试输入")
        mock_fix.assert_called_once()
        mock_merge.assert_not_called()
        mock_v.assert_not_called()
        mock_gm.assert_not_called()
        mock_sc.assert_not_called()
        assert result == "polished"

    def test_dispatch_anthropic_merge_calls_process_with_anthropic_merge(self):
        daemon = _make_daemon()
        daemon.current_post_processor_id = "claude-merge"
        daemon.post_processor_framework = "anthropic-merge"
        daemon._vocab = {}
        daemon._last_secondary_text = "English text"
        with patch("post_processor_configs.process_with_anthropic_merge",
                   return_value="merged") as mock_merge, \
             patch("post_processor_configs.process_with_anthropic") as mock_fix, \
             patch("post_processor_configs.process_with_gemini_merge") as mock_gm, \
             patch("state_db.update_state"):
            result = daemon._post_process("这是一段足够长的测试输入文本")
        mock_merge.assert_called_once()
        args = mock_merge.call_args[0]
        # third positional arg should be secondary_text
        assert args[1] == "English text"
        mock_fix.assert_not_called()
        mock_gm.assert_not_called()
        assert result == "merged"

    def test_dispatch_anthropic_merge_secondary_none(self):
        daemon = _make_daemon()
        daemon.current_post_processor_id = "claude-merge"
        daemon.post_processor_framework = "anthropic-merge"
        daemon._vocab = {}
        daemon._last_secondary_text = None
        with patch("post_processor_configs.process_with_anthropic_merge",
                   return_value="merged") as mock_merge, \
             patch("state_db.update_state"):
            daemon._post_process("这是一段足够长的测试输入文本")
        args = mock_merge.call_args[0]
        assert args[1] is None

    def test_existing_vertex_ai_merge_unchanged(self):
        daemon = _make_daemon()
        daemon.current_post_processor_id = "gemini-merge"
        daemon.post_processor_framework = "vertex-ai-merge"
        daemon._vocab = {}
        with patch("post_processor_configs.process_with_gemini_merge",
                   return_value="merged-gemini") as mock_gm, \
             patch("post_processor_configs.process_with_anthropic_merge") as mock_am, \
             patch("state_db.update_state"):
            result = daemon._post_process("这是一段足够长的测试输入文本")
        mock_gm.assert_called_once()
        mock_am.assert_not_called()
        assert result == "merged-gemini"

    def test_existing_vertex_ai_fix_unchanged(self):
        daemon = _make_daemon()
        daemon.current_post_processor_id = "gemini-fix"
        daemon.post_processor_framework = "vertex-ai"
        daemon._vocab = {}
        with patch("post_processor_configs.process_with_vertex_ai",
                   return_value="polished") as mock_v, \
             patch("post_processor_configs.process_with_anthropic") as mock_a, \
             patch("state_db.update_state"):
            result = daemon._post_process("这是一段足够长的测试输入文本")
        mock_v.assert_called_once()
        mock_a.assert_not_called()
        assert result == "polished"

    def test_existing_ssh_claude_unchanged(self):
        daemon = _make_daemon()
        daemon.current_post_processor_id = "haiku-fix"
        daemon.post_processor_framework = "ssh-claude"
        daemon._vocab = {}
        with patch("post_processor_configs.process_with_ssh_claude",
                   return_value="polished") as mock_sc, \
             patch("post_processor_configs.process_with_anthropic") as mock_a, \
             patch("state_db.update_state"):
            result = daemon._post_process("这是一段足够长的测试输入文本")
        mock_sc.assert_called_once()
        mock_a.assert_not_called()
        assert result == "polished"

    def test_empty_input_returns_empty(self):
        daemon = _make_daemon()
        daemon.current_post_processor_id = "claude-fix"
        daemon.post_processor_framework = "anthropic"
        with patch("post_processor_configs.process_with_anthropic") as mock_a:
            result = daemon._post_process("")
        mock_a.assert_not_called()
        assert result == ""

    def test_regex_framework_skips_ssh_dispatch(self):
        daemon = _make_daemon()
        daemon.current_post_processor_id = "none"
        daemon.post_processor_framework = "regex"
        with patch("post_processor_configs.process_with_anthropic") as mock_a, \
             patch("state_db.update_state") as mock_us:
            daemon._post_process("呃这是一段测试")
        mock_a.assert_not_called()
        mock_us.assert_not_called()
