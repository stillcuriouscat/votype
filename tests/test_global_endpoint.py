#!/usr/bin/env python3
"""Unit tests for US-003: Switch Vertex AI from regional to global endpoint.

Verifies that gemini-fix and gemini-merge presets use vertex_region='global'
and that the region value flows through to the vertex_proxy stdin JSON.
"""

import json
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")


class TestGeminiFIXPresetGlobalRegion(unittest.TestCase):
    """Verify gemini-fix preset uses global endpoint."""

    def test_gemini_fix_vertex_region_is_global(self):
        """AC: gemini-fix preset vertex_region changed to 'global'."""
        from post_processor_presets import POST_PROCESSOR_PRESETS

        config = POST_PROCESSOR_PRESETS["gemini-fix"]["config"]
        assert config["vertex_region"] == "global"

    def test_gemini_fix_region_not_us_central1(self):
        """Ensure old us-central1 is no longer used."""
        from post_processor_presets import POST_PROCESSOR_PRESETS

        config = POST_PROCESSOR_PRESETS["gemini-fix"]["config"]
        assert config["vertex_region"] != "us-central1"


class TestGeminiMergePresetGlobalRegion(unittest.TestCase):
    """Verify gemini-merge preset uses global endpoint."""

    def test_gemini_merge_vertex_region_is_global(self):
        """AC: gemini-merge preset vertex_region changed to 'global'."""
        from post_processor_presets import POST_PROCESSOR_PRESETS

        config = POST_PROCESSOR_PRESETS["gemini-merge"]["config"]
        assert config["vertex_region"] == "global"

    def test_gemini_merge_region_not_us_central1(self):
        """Ensure old us-central1 is no longer used."""
        from post_processor_presets import POST_PROCESSOR_PRESETS

        config = POST_PROCESSOR_PRESETS["gemini-merge"]["config"]
        assert config["vertex_region"] != "us-central1"


class TestRegionFlowsToStdinJSON(unittest.TestCase):
    """Verify vertex_region flows through to subprocess stdin as 'region' field."""

    @patch("post_processor_configs.subprocess.run")
    def test_gemini_fix_passes_global_region_to_proxy(self, mock_run):
        """process_with_vertex_ai sends region='global' in stdin JSON."""
        from post_processor_configs import process_with_vertex_ai
        from post_processor_presets import POST_PROCESSOR_PRESETS

        mock_run.return_value = MagicMock(
            returncode=0, stdout="polished text output here", stderr=""
        )

        config = POST_PROCESSOR_PRESETS["gemini-fix"]["config"]
        # Use inline system_prompt to avoid file I/O
        test_config = dict(config, system_prompt="Test prompt")
        del test_config["system_prompt_file"]

        process_with_vertex_ai("a" * 50, test_config)

        call_args = mock_run.call_args
        stdin_json = json.loads(call_args.kwargs.get("input") or call_args[1].get("input", ""))
        assert stdin_json["region"] == "global"

    @patch("post_processor_configs.subprocess.run")
    def test_gemini_merge_passes_global_region_to_proxy(self, mock_run):
        """process_with_gemini_merge sends region='global' in stdin JSON."""
        from post_processor_configs import process_with_gemini_merge
        from post_processor_presets import POST_PROCESSOR_PRESETS

        mock_run.return_value = MagicMock(
            returncode=0, stdout="merged text output here", stderr=""
        )

        config = POST_PROCESSOR_PRESETS["gemini-merge"]["config"]
        test_config = dict(config, system_prompt="Test merge prompt")
        del test_config["system_prompt_file"]

        process_with_gemini_merge("a" * 50, "b" * 50, test_config)

        call_args = mock_run.call_args
        stdin_json = json.loads(call_args.kwargs.get("input") or call_args[1].get("input", ""))
        assert stdin_json["region"] == "global"

    @patch("post_processor_configs.subprocess.run")
    def test_fallback_default_is_global(self, mock_run):
        """Config without vertex_region defaults to 'global'."""
        from post_processor_configs import process_with_vertex_ai

        mock_run.return_value = MagicMock(
            returncode=0, stdout="polished text", stderr=""
        )

        config_no_region = {
            "ssh_host": "oracle-cloud",
            "proxy_script": "~/vertex_proxy.py",
            "model": "gemini-2.5-flash",
            "timeout": 15,
            "min_text_len": 15,
            "system_prompt": "Test prompt",
        }

        process_with_vertex_ai("a" * 50, config_no_region)

        call_args = mock_run.call_args
        stdin_json = json.loads(call_args.kwargs.get("input") or call_args[1].get("input", ""))
        assert stdin_json["region"] == "global"


if __name__ == "__main__":
    unittest.main()
