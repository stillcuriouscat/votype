"""
Real-world scenario integration tests

Test actual workflows for model switching, terminal input, and GUI input
"""

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input


class TestTerminalInputDetection:
    """Test terminal input detection functionality"""

    @patch('voice_input.subprocess.run')
    def test_detects_gnome_terminal(self, mock_run):
        """
        Test detecting gnome-terminal

        Scenario: Current window is gnome-terminal
        Expected: is_terminal_window() returns True
        """
        mock_run.return_value = MagicMock(stdout="user@host: ~/project - gnome-terminal")

        result = voice_input.is_terminal_window()

        assert result is True
        mock_run.assert_called_once()

    @patch('voice_input.subprocess.run')
    def test_detects_konsole(self, mock_run):
        """
        Test detecting konsole

        Scenario: Current window is konsole
        Expected: is_terminal_window() returns True
        """
        mock_run.return_value = MagicMock(stdout="Shell - Konsole")

        result = voice_input.is_terminal_window()

        assert result is True

    @patch('voice_input.subprocess.run')
    def test_does_not_detect_browser(self, mock_run):
        """
        Test browser is not detected as terminal

        Scenario: Current window is a browser
        Expected: is_terminal_window() returns False
        """
        mock_run.return_value = MagicMock(stdout="Mozilla Firefox")

        result = voice_input.is_terminal_window()

        assert result is False

    @patch('voice_input.subprocess.run')
    def test_handles_xdotool_not_found(self, mock_run):
        """
        Test xdotool not available scenario

        Scenario: xdotool is not installed on the system
        Expected: Returns False (defaults to non-terminal)
        """
        mock_run.side_effect = FileNotFoundError()

        result = voice_input.is_terminal_window()

        assert result is False


class TestSmartTextInput:
    """Test smart text input functionality"""

    @patch('voice_input.is_terminal_window')
    @patch('voice_input.subprocess.run')
    def test_uses_xdotool_type_in_terminal(self, mock_run, mock_is_terminal):
        """
        Test using xdotool type in terminal

        Scenario: Currently in a terminal, typing text
        Expected: Calls xdotool type for direct input
        """
        mock_is_terminal.return_value = True

        voice_input.type_text("test text")

        # Verify xdotool type was called
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "xdotool"
        assert args[1] == "type"
        assert "test text" in args

    @patch('voice_input.is_terminal_window')
    @patch('voice_input.clipboard_get')
    @patch('voice_input.clipboard_set')
    @patch('voice_input.subprocess.run')
    def test_uses_clipboard_in_gui(self, mock_run, mock_clip_set, mock_clip_get, mock_is_terminal):
        """
        Test using clipboard in GUI

        Scenario: Currently in a browser, typing text
        Expected: Uses clipboard paste method
        """
        mock_is_terminal.return_value = False
        mock_clip_get.return_value = "old clipboard"

        voice_input.type_text("GUI text")

        # Verify clipboard was set
        assert mock_clip_set.call_count >= 1
        # First call should set new text
        first_call = mock_clip_set.call_args_list[0][0][0]
        assert first_call == "GUI text"

        # Verify xdotool key ctrl+v was called
        assert mock_run.called
        args = mock_run.call_args_list[0][0][0]
        assert "xdotool" in args
        assert "key" in args

    @patch('voice_input.is_terminal_window')
    @patch('voice_input.subprocess.run')
    def test_handles_special_characters_in_terminal(self, mock_run, mock_is_terminal):
        """
        Test handling special characters in terminal mode

        Scenario: Input text containing shell special characters
        Expected: Uses -- argument to protect special characters
        """
        mock_is_terminal.return_value = True

        voice_input.type_text("echo $HOME && ls")

        args = mock_run.call_args[0][0]
        # Verify -- was used to protect special characters
        assert "--" in args


class TestEndToEndWorkflow:
    """End-to-end workflow tests"""

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
