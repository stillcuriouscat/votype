"""
Test tray menu state synchronization

Verify that menu item check state updates correctly when switching models
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import threading
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
import voice_input


class TestMenuState:
    """Test tray menu state management"""

    @pytest.fixture
    def mock_gtk_environment(self):
        """Mock GTK environment"""
        # Mock GTK classes
        mock_gtk = MagicMock()
        mock_glib = MagicMock()
        mock_appindicator = MagicMock()

        # Create RadioMenuItem mock
        class MockRadioMenuItem:
            def __init__(self, group=None, label=""):
                self.label = label
                self.group = group
                self.active = False
                self.handlers = {}
                self.next_handler_id = 1

            def set_active(self, active):
                """Set active (checked) state"""
                old_active = self.active
                self.active = active
                # RadioMenuItem behavior: activating one automatically deactivates others in the group
                if active and self.group:
                    for item in self.group.items:
                        if item != self and item.active:
                            item.active = False

            def get_active(self):
                return self.active

            def connect(self, signal, callback, *args):
                """Connect signal"""
                handler_id = self.next_handler_id
                self.next_handler_id += 1
                self.handlers[handler_id] = (callback, args)
                return handler_id

            def handler_block(self, handler_id):
                """Block signal"""
                if handler_id in self.handlers:
                    self.handlers[handler_id] = (*self.handlers[handler_id], True)  # blocked flag

            def handler_unblock(self, handler_id):
                """Unblock signal"""
                if handler_id in self.handlers:
                    callback, args = self.handlers[handler_id][:2]
                    self.handlers[handler_id] = (callback, args, False)

            def emit_toggled(self):
                """Simulate toggled signal"""
                for handler_id, handler_data in self.handlers.items():
                    callback, args = handler_data[:2]
                    blocked = handler_data[2] if len(handler_data) > 2 else False
                    if not blocked:
                        callback(self, *args)

        # Create RadioMenuItemGroup
        class MockRadioMenuItemGroup:
            def __init__(self):
                self.items = []

            def add(self, item):
                self.items.append(item)

        mock_gtk.RadioMenuItem = MockRadioMenuItem
        mock_gtk.MenuItem = MagicMock
        mock_gtk.Menu = MagicMock
        mock_gtk.SeparatorMenuItem = MagicMock
        mock_gtk.main_quit = MagicMock()

        mock_glib.idle_add = lambda func: func()  # Execute immediately

        return {
            'gtk': mock_gtk,
            'glib': mock_glib,
            'appindicator': mock_appindicator,
            'RadioMenuItemGroup': MockRadioMenuItemGroup
        }

    def test_radio_menu_item_mutual_exclusion(self, mock_gtk_environment):
        """
        Test RadioMenuItem mutual exclusion behavior

        Scenario: Create a RadioMenuItem group, verify only one can be active
        Expected: When activating a new item, the old item is automatically deactivated
        """
        RadioMenuItem = mock_gtk_environment['gtk'].RadioMenuItem
        group = mock_gtk_environment['RadioMenuItemGroup']()

        # Create three RadioMenuItems
        item1 = RadioMenuItem(group=group, label="Model 1")
        item2 = RadioMenuItem(group=group, label="Model 2")
        item3 = RadioMenuItem(group=group, label="Model 3")

        group.add(item1)
        group.add(item2)
        group.add(item3)

        # Initial state: first item is active
        item1.set_active(True)
        assert item1.get_active() is True
        assert item2.get_active() is False
        assert item3.get_active() is False

        # Switch to the second item
        item2.set_active(True)
        assert item1.get_active() is False, "Old item should be automatically unchecked"
        assert item2.get_active() is True
        assert item3.get_active() is False

        # Switch to the third item
        item3.set_active(True)
        assert item1.get_active() is False
        assert item2.get_active() is False, "Old item should be automatically unchecked"
        assert item3.get_active() is True

    def test_handler_blocking_prevents_signal(self, mock_gtk_environment):
        """
        Test handler_block prevents signal emission

        Scenario: Connect signal, block handler, modify state, verify callback not triggered
        Expected: Callback is not triggered while handler is blocked
        """
        RadioMenuItem = mock_gtk_environment['gtk'].RadioMenuItem

        item = RadioMenuItem(label="Test")

        callback_called = []
        def callback(widget, model_id):
            callback_called.append(model_id)

        # Connect signal
        handler_id = item.connect("toggled", callback, "test_model")

        # Without blocking: should trigger callback
        item.set_active(True)
        item.emit_toggled()
        assert len(callback_called) == 1, "Callback should be triggered when not blocked"

        # With blocking: should not trigger callback
        item.handler_block(handler_id)
        item.set_active(False)
        item.emit_toggled()
        assert len(callback_called) == 1, "Callback should not be triggered when blocked"

        # After unblocking: should trigger callback
        item.handler_unblock(handler_id)
        item.set_active(True)
        item.emit_toggled()
        assert len(callback_called) == 2, "Callback should be triggered after unblocking"

    @pytest.mark.skipif(not voice_input.HAS_INDICATOR, reason="GTK/AppIndicator not available")
    def test_update_model_menu_with_handler_blocking(self, mock_gtk_environment, isolated_environment):
        """
        Test _update_model_menu correctly blocks signals using handler_id

        Scenario: Call _update_model_menu after switching models
        Expected: Menu state updates without triggering callbacks (signals are blocked)
        """
        with patch('voice_input.AyatanaAppIndicator3'):
            daemon = voice_input.ASRDaemon()

            # Simulate menu already created
            daemon.model_menu_item = MagicMock()
            daemon.model_menu_items = {}
            daemon.model_menu_handlers = {}

            RadioMenuItem = mock_gtk_environment['gtk'].RadioMenuItem

            # Create mock menu items
            for model_id in voice_input.MODEL_PRESETS.keys():
                item = RadioMenuItem(label=f"Model {model_id}")
                handler_id = 100 + len(daemon.model_menu_items)  # Simulated handler_id
                daemon.model_menu_items[model_id] = item
                daemon.model_menu_handlers[model_id] = handler_id

            # Set current model
            daemon.current_model_id = "paraformer"
            daemon.model_menu_items["paraformer"].set_active(True)

            # Call _update_model_menu
            daemon._update_model_menu()

            # Verify only the current model is checked
            for model_id, item in daemon.model_menu_items.items():
                if model_id == daemon.current_model_id:
                    assert item.get_active(), f"{model_id} should be checked"
                else:
                    assert not item.get_active(), f"{model_id} should not be checked"

    @pytest.mark.skipif(not voice_input.HAS_INDICATOR, reason="GTK/AppIndicator not available")
    def test_menu_state_after_model_switch(self, mock_gtk_environment, mock_asr_model, isolated_environment):
        """
        Test menu state correctness after model switch

        Scenario: Switch from model A to model B
        Expected: Only model B is checked in the menu, model A is unchecked
        """
        with patch('voice_input.AyatanaAppIndicator3'):
            # Create daemon without starting the daemon process (single model architecture)
            daemon = voice_input.ASRDaemon()
            daemon.model = mock_asr_model['model_instance']
            daemon.current_model_id = "paraformer"

            # Mock menu items
            daemon.model_menu_item = MagicMock()
            daemon.model_menu_items = {}
            daemon.model_menu_handlers = {}

            RadioMenuItem = mock_gtk_environment['gtk'].RadioMenuItem
            group = mock_gtk_environment['RadioMenuItemGroup']()

            for model_id in ["paraformer", "sensevoice"]:
                item = RadioMenuItem(group=group, label=f"Model {model_id}")
                group.add(item)
                item.set_active(model_id == "paraformer")
                handler_id = item.connect("toggled", daemon._on_model_selected, model_id)
                daemon.model_menu_items[model_id] = item
                daemon.model_menu_handlers[model_id] = handler_id

            # Verify initial state
            assert daemon.model_menu_items["paraformer"].get_active()
            assert not daemon.model_menu_items["sensevoice"].get_active()

            # Switch model
            result = daemon.switch_model("sensevoice")

            assert result["status"] == "ok"
            assert daemon.current_model_id == "sensevoice"

            # Verify menu state
            assert not daemon.model_menu_items["paraformer"].get_active(), "Old model should not be checked"
            assert daemon.model_menu_items["sensevoice"].get_active(), "New model should be checked"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
