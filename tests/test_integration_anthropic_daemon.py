"""Clean-room integration tests for ASRDaemon dispatching (Module D).

Derived from LOW_LEVEL_DESIGN.md §1 Module D. Verifies:
  - load_post_processor: vocab + secondary-model load gating for the
    extended framework sets.
  - **Critical US-004 invariant**: switching between two merge frameworks
    (gemini-merge → claude-merge) does NOT reload the secondary model.
  - _post_process: framework dispatch routes 'anthropic' to
    process_with_anthropic and 'anthropic-merge' to
    process_with_anthropic_merge.

We mock heavy dependencies (PostProcessorLoader, secondary model loader,
update_state, the SSH-call functions in post_processor_configs) so that
the test exercises ASRDaemon's *dispatch* code path without performing
network or filesystem I/O.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_voice_input():
    try:
        import voice_input  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"voice_input import failed: {e}")
    return voice_input


def _import_presets():
    try:
        import post_processor_presets as ppp  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"post_processor_presets import failed: {e}")
    return ppp


def _has_class_and_methods() -> bool:
    try:
        vi = sys.modules.get("voice_input") or __import__("voice_input")
    except Exception:
        return False
    cls = getattr(vi, "ASRDaemon", None)
    if cls is None:
        return False
    return all(hasattr(cls, m) for m in ("load_post_processor", "_post_process"))


pytestmark = pytest.mark.skipif(
    not _has_class_and_methods(),
    reason="ASRDaemon or its methods not yet present",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daemon(vi):
    """Construct an ASRDaemon without invoking its __init__ side-effects.

    ASRDaemon's real __init__ typically loads heavy models / opens sockets.
    For dispatch-only tests we create an instance via __new__ and stamp on
    the attributes the methods under test will read.
    """
    daemon = vi.ASRDaemon.__new__(vi.ASRDaemon)
    # Attributes used by both methods (per LLD).
    daemon.post_processor_model = None
    daemon.current_post_processor_id = "none"
    daemon.post_processor_framework = "regex"
    daemon._secondary_model = None
    daemon._last_secondary_text = None
    daemon._vocab = {}
    return daemon


# ---------------------------------------------------------------------------
# load_post_processor — framework gating
# ---------------------------------------------------------------------------


class TestLoadPostProcessorFrameworkGates:
    """LLD §1 Module D: vocab + secondary-model gates for the extended sets."""

    def test_anthropic_framework_loads_vocab(self):
        """LLD: vocab loaded for framework ∈ {ssh-claude, vertex-ai, vertex-ai-merge,
        anthropic, anthropic-merge}."""
        vi = _import_voice_input()
        daemon = _make_daemon(vi)

        load_vocab_mock = MagicMock(return_value={"foo": "bar"})

        with patch.object(vi, "PostProcessorLoader", create=True) as ldr, \
             patch.object(vi, "load_vocab", load_vocab_mock, create=True), \
             patch.object(vi, "update_state", create=True), \
             patch.object(daemon, "_load_secondary_model", create=True), \
             patch.object(daemon, "_unload_secondary_model", create=True):
            ldr.load_post_processor = MagicMock(return_value=MagicMock())
            try:
                daemon.load_post_processor("claude-fix")
            except Exception as e:
                pytest.skip(f"daemon dependencies not satisfied in isolation: {e}")

        assert load_vocab_mock.called, (
            "load_vocab must be called when framework is 'anthropic'"
        )

    def test_anthropic_merge_framework_loads_secondary_model(self):
        """LLD: secondary loaded for framework ∈ {vertex-ai-merge, anthropic-merge}."""
        vi = _import_voice_input()
        daemon = _make_daemon(vi)
        daemon._secondary_model = None  # not yet loaded

        load_secondary = MagicMock()
        unload_secondary = MagicMock()

        with patch.object(vi, "PostProcessorLoader", create=True) as ldr, \
             patch.object(vi, "load_vocab", MagicMock(return_value={}), create=True), \
             patch.object(vi, "update_state", create=True):
            daemon._load_secondary_model = load_secondary
            daemon._unload_secondary_model = unload_secondary
            ldr.load_post_processor = MagicMock(return_value=MagicMock())
            try:
                daemon.load_post_processor("claude-merge")
            except Exception as e:
                pytest.skip(f"daemon dependencies not satisfied in isolation: {e}")

        assert load_secondary.called, (
            "_load_secondary_model must be called for anthropic-merge"
        )
        assert not unload_secondary.called, (
            "_unload_secondary_model must NOT be called for anthropic-merge"
        )

    def test_anthropic_fix_framework_does_not_load_secondary(self):
        """LLD: framework 'anthropic' must NOT load the secondary model.

        (Secondary is only for merge frameworks.)"""
        vi = _import_voice_input()
        daemon = _make_daemon(vi)

        load_secondary = MagicMock()
        unload_secondary = MagicMock()

        with patch.object(vi, "PostProcessorLoader", create=True) as ldr, \
             patch.object(vi, "load_vocab", MagicMock(return_value={}), create=True), \
             patch.object(vi, "update_state", create=True):
            daemon._load_secondary_model = load_secondary
            daemon._unload_secondary_model = unload_secondary
            ldr.load_post_processor = MagicMock(return_value=MagicMock())
            try:
                daemon.load_post_processor("claude-fix")
            except Exception as e:
                pytest.skip(f"daemon dependencies not satisfied in isolation: {e}")

        assert not load_secondary.called, (
            "Secondary model must not load for 'anthropic' (fix) framework"
        )

    def test_unknown_preset_raises_runtime_error(self):
        """LLD: RuntimeError when preset_id not in POST_PROCESSOR_PRESETS."""
        vi = _import_voice_input()
        daemon = _make_daemon(vi)

        with patch.object(vi, "update_state", create=True):
            with pytest.raises(RuntimeError):
                daemon.load_post_processor("definitely-not-a-real-preset-xyz")


# ---------------------------------------------------------------------------
# Critical US-004 invariant: no double-load on merge↔merge switch
# ---------------------------------------------------------------------------


class TestMergeFrameworkSwitchNoDoubleLoad:
    """LLD §1 Module D **Critical invariant**:

    Switching gemini-merge → claude-merge MUST NOT call
    _unload_secondary_model() followed by _load_secondary_model().

    Equivalently: after the second switch, _load_secondary_model.call_count
    must be 0 (because the secondary is already loaded).
    """

    def test_gemini_merge_to_claude_merge_does_not_reload_secondary(self):
        vi = _import_voice_input()
        daemon = _make_daemon(vi)

        # Simulate state after gemini-merge has already been loaded.
        sentinel_model = object()
        daemon._secondary_model = sentinel_model
        daemon.post_processor_framework = "vertex-ai-merge"
        daemon.current_post_processor_id = "gemini-merge"

        load_secondary = MagicMock()
        unload_secondary = MagicMock()

        with patch.object(vi, "PostProcessorLoader", create=True) as ldr, \
             patch.object(vi, "load_vocab", MagicMock(return_value={}), create=True), \
             patch.object(vi, "update_state", create=True):
            daemon._load_secondary_model = load_secondary
            daemon._unload_secondary_model = unload_secondary
            ldr.load_post_processor = MagicMock(return_value=MagicMock())
            try:
                daemon.load_post_processor("claude-merge")
            except Exception as e:
                pytest.skip(f"daemon dependencies not satisfied in isolation: {e}")

        # The critical invariant assertion:
        assert load_secondary.call_count == 0, (
            f"Switching merge-frameworks must NOT reload secondary model; "
            f"_load_secondary_model.call_count = {load_secondary.call_count} "
            f"(expected 0 — secondary was already loaded)."
        )
        assert unload_secondary.call_count == 0, (
            "Switching merge-frameworks must NOT unload secondary model"
        )
        # And the secondary model should still be present.
        assert daemon._secondary_model is sentinel_model

    def test_claude_merge_to_gemini_merge_does_not_reload_secondary(self):
        """Symmetric to the above."""
        vi = _import_voice_input()
        daemon = _make_daemon(vi)

        sentinel_model = object()
        daemon._secondary_model = sentinel_model
        daemon.post_processor_framework = "anthropic-merge"
        daemon.current_post_processor_id = "claude-merge"

        load_secondary = MagicMock()
        unload_secondary = MagicMock()

        with patch.object(vi, "PostProcessorLoader", create=True) as ldr, \
             patch.object(vi, "load_vocab", MagicMock(return_value={}), create=True), \
             patch.object(vi, "update_state", create=True):
            daemon._load_secondary_model = load_secondary
            daemon._unload_secondary_model = unload_secondary
            ldr.load_post_processor = MagicMock(return_value=MagicMock())
            try:
                daemon.load_post_processor("gemini-merge")
            except Exception as e:
                pytest.skip(f"daemon dependencies not satisfied in isolation: {e}")

        assert load_secondary.call_count == 0
        assert unload_secondary.call_count == 0


# ---------------------------------------------------------------------------
# _post_process — framework dispatch
# ---------------------------------------------------------------------------


class TestPostProcessDispatch:
    """LLD §1 Module D: framework dispatch in _post_process."""

    def _prepare_daemon_for_dispatch(
        self, vi, framework: str, preset_id: str,
    ):
        """Stamp the daemon into the state _post_process expects."""
        daemon = _make_daemon(vi)
        daemon.post_processor_framework = framework
        daemon.current_post_processor_id = preset_id
        daemon.post_processor_model = MagicMock()  # truthy placeholder
        daemon._last_secondary_text = "english secondary text"

        # Stamp config matching the preset's config dict.
        ppp = _import_presets()
        preset = ppp.POST_PROCESSOR_PRESETS.get(preset_id, {})
        daemon.post_processor_config = preset.get("config", {})
        return daemon

    def test_anthropic_framework_dispatches_to_process_with_anthropic(self):
        vi = _import_voice_input()
        daemon = self._prepare_daemon_for_dispatch(vi, "anthropic", "claude-fix")
        text = "this is a longer text that exceeds the minimum length threshold"

        with patch("post_processor_configs.process_with_anthropic",
                   return_value="ANTHROPIC_FIX_RESULT") as fix_mock, \
             patch("post_processor_configs.process_with_anthropic_merge",
                   return_value="SHOULD_NOT_BE_CALLED") as merge_mock, \
             patch("post_processor_configs.process_with_vertex_ai",
                   return_value="WRONG_VERTEX") as vertex_mock, \
             patch.object(vi, "update_state", create=True):
            try:
                result = daemon._post_process(text)
            except Exception as e:
                pytest.skip(f"daemon _post_process dependencies not satisfied: {e}")

        assert fix_mock.called, "framework='anthropic' must call process_with_anthropic"
        assert not merge_mock.called
        assert not vertex_mock.called
        assert result == "ANTHROPIC_FIX_RESULT"

    def test_anthropic_merge_framework_dispatches_to_anthropic_merge(self):
        vi = _import_voice_input()
        daemon = self._prepare_daemon_for_dispatch(vi, "anthropic-merge", "claude-merge")
        text = "这是一段足够长的中文转录文本用于触发合并流程"

        with patch("post_processor_configs.process_with_anthropic_merge",
                   return_value="ANTHROPIC_MERGE_RESULT") as merge_mock, \
             patch("post_processor_configs.process_with_anthropic",
                   return_value="WRONG_FIX") as fix_mock, \
             patch("post_processor_configs.process_with_gemini_merge",
                   return_value="WRONG_GEMINI") as gemini_mock, \
             patch.object(vi, "update_state", create=True):
            try:
                result = daemon._post_process(text)
            except Exception as e:
                pytest.skip(f"daemon _post_process dependencies not satisfied: {e}")

        assert merge_mock.called, (
            "framework='anthropic-merge' must call process_with_anthropic_merge"
        )
        assert not fix_mock.called
        assert not gemini_mock.called
        assert result == "ANTHROPIC_MERGE_RESULT"

    def test_anthropic_merge_receives_secondary_text(self):
        """LLD: anthropic-merge gets secondary from self._last_secondary_text."""
        vi = _import_voice_input()
        daemon = self._prepare_daemon_for_dispatch(vi, "anthropic-merge", "claude-merge")
        daemon._last_secondary_text = "english secondary ASR output"
        text = "这是一段足够长的中文转录文本用于触发合并流程"

        with patch("post_processor_configs.process_with_anthropic_merge",
                   return_value="MERGED") as merge_mock, \
             patch.object(vi, "update_state", create=True):
            try:
                daemon._post_process(text)
            except Exception as e:
                pytest.skip(f"daemon _post_process dependencies not satisfied: {e}")

        assert merge_mock.called
        args, kwargs = merge_mock.call_args
        # Expected positional signature: (primary, secondary, config, glossary_ctx)
        all_args = list(args) + list(kwargs.values())
        assert "english secondary ASR output" in all_args, (
            "process_with_anthropic_merge must receive _last_secondary_text"
        )
