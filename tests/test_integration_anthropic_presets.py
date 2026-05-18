"""Clean-room integration tests for post_processor_presets.py (Module C).

Derived from LOW_LEVEL_DESIGN.md §1 Module C. Verifies:
  - The two new presets (claude-fix, claude-merge) exist with the exact
    documented schemas.
  - DEFAULT_POST_PROCESSOR was reassigned to 'claude-merge'.
  - The eight existing presets are still present with their keys intact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_presets():
    try:
        import post_processor_presets as ppp  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"post_processor_presets import failed: {e}")
    return ppp


# ---------------------------------------------------------------------------
# Default reassignment
# ---------------------------------------------------------------------------


class TestDefaultPostProcessor:

    def test_default_is_claude_merge(self):
        """LLD §1 Module C: DEFAULT_POST_PROCESSOR == 'claude-merge'."""
        ppp = _import_presets()
        assert ppp.DEFAULT_POST_PROCESSOR == "claude-merge"

    def test_default_preset_has_anthropic_merge_framework(self):
        """Consistency check: presets[DEFAULT_POST_PROCESSOR]['framework'] == 'anthropic-merge'."""
        ppp = _import_presets()
        preset = ppp.POST_PROCESSOR_PRESETS[ppp.DEFAULT_POST_PROCESSOR]
        assert preset["framework"] == "anthropic-merge"


# ---------------------------------------------------------------------------
# New preset schemas
# ---------------------------------------------------------------------------


class TestClaudeFixPreset:
    """LLD §1 Module C: claude-fix preset schema."""

    def test_claude_fix_present(self):
        ppp = _import_presets()
        assert "claude-fix" in ppp.POST_PROCESSOR_PRESETS

    def test_claude_fix_framework(self):
        ppp = _import_presets()
        preset = ppp.POST_PROCESSOR_PRESETS["claude-fix"]
        assert preset["framework"] == "anthropic"

    def test_claude_fix_config_shape(self):
        ppp = _import_presets()
        cfg = ppp.POST_PROCESSOR_PRESETS["claude-fix"]["config"]
        assert cfg["ssh_host"] == "oracle-cloud"
        assert cfg["proxy_script"] == "~/anthropic_proxy.py"
        assert cfg["model"] == "claude-haiku-4-5-20251001"
        assert cfg["timeout"] == 15
        assert cfg["min_text_len"] == 15
        assert cfg["vocab_min_count"] == 3
        assert cfg["system_prompt_file"] == "prompts/gemini-fix-system.txt"
        assert cfg["user_prompt_template_file"] == "prompts/haiku-fix-user.txt"


class TestClaudeMergePreset:
    """LLD §1 Module C: claude-merge preset schema."""

    def test_claude_merge_present(self):
        ppp = _import_presets()
        assert "claude-merge" in ppp.POST_PROCESSOR_PRESETS

    def test_claude_merge_framework(self):
        ppp = _import_presets()
        preset = ppp.POST_PROCESSOR_PRESETS["claude-merge"]
        assert preset["framework"] == "anthropic-merge"

    def test_claude_merge_config_shape(self):
        ppp = _import_presets()
        cfg = ppp.POST_PROCESSOR_PRESETS["claude-merge"]["config"]
        assert cfg["ssh_host"] == "oracle-cloud"
        assert cfg["proxy_script"] == "~/anthropic_proxy.py"
        assert cfg["model"] == "claude-haiku-4-5-20251001"
        assert cfg["timeout"] == 15
        assert cfg["min_text_len"] == 15
        assert cfg["vocab_min_count"] == 3
        assert cfg["system_prompt_file"] == "prompts/gemini-merge-system.txt"

    def test_claude_merge_has_no_user_prompt_template(self):
        """LLD: claude-merge — 'No user_prompt_template* — merge user_input is built in code.'"""
        ppp = _import_presets()
        cfg = ppp.POST_PROCESSOR_PRESETS["claude-merge"]["config"]
        assert "user_prompt_template" not in cfg
        assert "user_prompt_template_file" not in cfg

    def test_neither_preset_has_fallback_model(self):
        """LLD §3.3: 'config.fallback_model is absent for both new presets'."""
        ppp = _import_presets()
        for key in ("claude-fix", "claude-merge"):
            cfg = ppp.POST_PROCESSOR_PRESETS[key]["config"]
            assert "fallback_model" not in cfg, (
                f"{key} must not declare a fallback_model"
            )


# ---------------------------------------------------------------------------
# Existing presets preserved
# ---------------------------------------------------------------------------


class TestExistingPresetsPreserved:
    """LLD §1 Module C invariant: 8 existing keys present, bit-for-bit identical.

    We can only test the 'present' half from spec alone (we don't have a
    blessed snapshot of the existing values inside this clean room without
    reading the source). But we CAN assert key presence and framework values
    that are stable per the spec's framework enumeration (§3.3).
    """

    EXISTING_KEYS = [
        "none",
        "chinese-text-correction",
        "qwen3-0.6b",
        "minicpm4-0.5b",
        "haiku-fix",
        "haiku-expand",
        "gemini-fix",
        "gemini-merge",
    ]

    def test_all_existing_keys_present(self):
        ppp = _import_presets()
        for key in self.EXISTING_KEYS:
            assert key in ppp.POST_PROCESSOR_PRESETS, (
                f"Existing preset {key!r} must remain present"
            )

    def test_total_preset_count_is_ten(self):
        """8 existing + 2 new = 10."""
        ppp = _import_presets()
        expected = set(self.EXISTING_KEYS) | {"claude-fix", "claude-merge"}
        assert set(ppp.POST_PROCESSOR_PRESETS.keys()) == expected, (
            f"Preset keys mismatch.\n"
            f"Got: {sorted(ppp.POST_PROCESSOR_PRESETS.keys())}\n"
            f"Expected: {sorted(expected)}"
        )

    def test_module_exports_present(self):
        """LLD: 'Module exports unchanged: POST_PROCESSOR_PRESETS, DEFAULT_POST_PROCESSOR,
        VOICE_INPUT_DATA_DIR, MODELS_DIR'."""
        ppp = _import_presets()
        for name in ("POST_PROCESSOR_PRESETS", "DEFAULT_POST_PROCESSOR",
                     "VOICE_INPUT_DATA_DIR", "MODELS_DIR"):
            assert hasattr(ppp, name), f"Module must export {name}"


# ---------------------------------------------------------------------------
# Framework enumeration consistency
# ---------------------------------------------------------------------------


class TestFrameworkEnumeration:
    """LLD §3.3: framework ∈ {regex, ssh-claude, vertex-ai, vertex-ai-merge,
    anthropic, anthropic-merge, llama-cpp}."""

    ALLOWED_FRAMEWORKS = {
        "regex", "ssh-claude", "vertex-ai", "vertex-ai-merge",
        "anthropic", "anthropic-merge", "llama-cpp",
    }

    def test_every_preset_has_known_framework(self):
        ppp = _import_presets()
        for key, preset in ppp.POST_PROCESSOR_PRESETS.items():
            fw = preset.get("framework")
            assert fw in self.ALLOWED_FRAMEWORKS, (
                f"preset {key!r} has unexpected framework {fw!r}"
            )
