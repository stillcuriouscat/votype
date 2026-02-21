#!/usr/bin/env python3
"""
GTK Settings Dialog - Voice Input Tool Settings Interface

Features:
1. Model selection: dropdown menu to select model, restart daemon on confirm
2. Hotword configuration: text box to edit specialized terms
3. Log viewer: display recent recognition logs
"""

import os
import subprocess
from pathlib import Path

# GTK imports
try:
    import gi
    gi.require_version('Gtk', '3.0')
    from gi.repository import Gtk, GLib, Pango
    HAS_GTK = True
except (ImportError, ValueError):
    HAS_GTK = False


# Config file paths
CONFIG_DIR = Path.home() / ".config" / "voice-input"
HOTWORDS_FILE = CONFIG_DIR / "hotwords.txt"
LOG_FILE = Path("/tmp/voice-input-daemon.log")


def ensure_config_dir():
    """Ensure the config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_hotwords():
    """Load hotword configuration."""
    ensure_config_dir()
    if HOTWORDS_FILE.exists():
        return HOTWORDS_FILE.read_text().strip()
    # Default hotwords
    return "software engineer machine learning artificial intelligence Python Claude API React TypeScript GitHub Docker Kubernetes AWS Azure"


def save_hotwords(hotwords):
    """Save hotword configuration."""
    ensure_config_dir()
    HOTWORDS_FILE.write_text(hotwords)


def get_recent_logs(lines=100):
    """Get recent logs."""
    if not LOG_FILE.exists():
        return "Log file not found"
    try:
        result = subprocess.run(
            ["tail", f"-{lines}", str(LOG_FILE)],
            capture_output=True, text=True
        )
        return result.stdout or "Log is empty"
    except Exception as e:
        return f"Failed to read logs: {e}"


class SettingsDialog(Gtk.Dialog):
    """Voice input settings dialog."""

    def __init__(self, parent=None, model_presets=None, current_model_id=None):
        """
        Initialize the settings dialog.

        Args:
            parent: Parent window
            model_presets: Available model presets {model_id: {"name": ..., "description": ...}}
            current_model_id: Currently selected model ID
        """
        super().__init__(
            title="Voice Input Settings",
            parent=parent,
            flags=0
        )

        self.set_default_size(500, 400)
        self.set_border_width(10)

        self.model_presets = model_presets or {}
        self.current_model_id = current_model_id
        self.selected_model_id = current_model_id

        # Add buttons
        self.add_button("Cancel", Gtk.ResponseType.CANCEL)
        self.add_button("Apply", Gtk.ResponseType.APPLY)

        # Create content area
        content_area = self.get_content_area()

        # Create tabs
        notebook = Gtk.Notebook()
        notebook.set_tab_pos(Gtk.PositionType.TOP)

        # Tab 1: Model settings
        model_page = self._create_model_page()
        notebook.append_page(model_page, Gtk.Label(label="Model"))

        # Tab 2: Hotword configuration
        hotwords_page = self._create_hotwords_page()
        notebook.append_page(hotwords_page, Gtk.Label(label="Hotwords"))

        # Tab 3: Log viewer
        log_page = self._create_log_page()
        notebook.append_page(log_page, Gtk.Label(label="Logs"))

        content_area.pack_start(notebook, True, True, 0)

        self.show_all()

    @staticmethod
    def _make_page_box():
        """Create a page container with standard margins."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        for side in ('start', 'end', 'top', 'bottom'):
            getattr(box, f'set_margin_{side}')(10)
        return box

    def _create_model_page(self):
        """Create model settings page."""
        box = self._make_page_box()

        # Title
        title_label = Gtk.Label()
        title_label.set_markup("<b>Select ASR Model</b>")
        title_label.set_halign(Gtk.Align.START)
        box.pack_start(title_label, False, False, 0)

        # Description
        desc_label = Gtk.Label(label="Changing the model requires restarting the daemon. Model loading may take 20-30 seconds.")
        desc_label.set_halign(Gtk.Align.START)
        desc_label.set_line_wrap(True)
        box.pack_start(desc_label, False, False, 0)

        # Model selection radio buttons
        model_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        model_box.set_margin_top(10)

        radio_group = None
        for model_id, preset in self.model_presets.items():
            label_text = f"{preset['name']} - {preset.get('description', '')}"
            radio = Gtk.RadioButton.new_with_label_from_widget(radio_group, label_text)
            if radio_group is None:
                radio_group = radio

            if model_id == self.current_model_id:
                radio.set_active(True)

            radio.connect("toggled", self._on_model_toggled, model_id)
            model_box.pack_start(radio, False, False, 0)

        box.pack_start(model_box, False, False, 0)

        # Current model info
        info_frame = Gtk.Frame(label="Current Model Info")
        info_frame.set_margin_top(10)
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        info_box.set_margin_start(10)
        info_box.set_margin_end(10)
        info_box.set_margin_top(5)
        info_box.set_margin_bottom(5)

        if self.current_model_id and self.current_model_id in self.model_presets:
            preset = self.model_presets[self.current_model_id]
            self.info_label = Gtk.Label(label=f"Current: {preset['name']}")
        else:
            self.info_label = Gtk.Label(label="Current: None selected")
        self.info_label.set_halign(Gtk.Align.START)
        info_box.pack_start(self.info_label, False, False, 0)
        info_frame.add(info_box)
        box.pack_start(info_frame, False, False, 0)

        return box

    def _on_model_toggled(self, radio, model_id):
        """Callback when model selection changes."""
        if radio.get_active():
            self.selected_model_id = model_id

    def _create_hotwords_page(self):
        """Create hotword configuration page."""
        box = self._make_page_box()

        # Title
        title_label = Gtk.Label()
        title_label.set_markup("<b>Hotword Configuration</b>")
        title_label.set_halign(Gtk.Align.START)
        box.pack_start(title_label, False, False, 0)

        # Description
        desc_label = Gtk.Label(label="Add specialized terms to improve recognition accuracy, separated by spaces")
        desc_label.set_halign(Gtk.Align.START)
        desc_label.set_line_wrap(True)
        box.pack_start(desc_label, False, False, 0)

        # Text input area
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_min_content_height(150)

        self.hotwords_textview = Gtk.TextView()
        self.hotwords_textview.set_wrap_mode(Gtk.WrapMode.WORD)
        self.hotwords_textview.get_buffer().set_text(load_hotwords())
        scrolled.add(self.hotwords_textview)
        box.pack_start(scrolled, True, True, 0)

        # Button area
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        button_box.set_halign(Gtk.Align.END)

        reset_button = Gtk.Button(label="Reset Default")
        reset_button.connect("clicked", self._on_reset_hotwords)
        button_box.pack_start(reset_button, False, False, 0)

        box.pack_start(button_box, False, False, 0)

        return box

    def _on_reset_hotwords(self, button):
        """Reset hotwords to default values."""
        default_hotwords = "software engineer machine learning artificial intelligence Python Claude API React TypeScript GitHub Docker Kubernetes AWS Azure"
        self.hotwords_textview.get_buffer().set_text(default_hotwords)

    def _create_log_page(self):
        """Create log viewer page."""
        box = self._make_page_box()

        # Title
        title_label = Gtk.Label()
        title_label.set_markup("<b>Recognition Logs</b>")
        title_label.set_halign(Gtk.Align.START)
        box.pack_start(title_label, False, False, 0)

        # Log display area
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        self.log_textview = Gtk.TextView()
        self.log_textview.set_editable(False)
        self.log_textview.set_cursor_visible(False)
        self.log_textview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)

        # Use monospace font
        self.log_textview.override_font(Pango.FontDescription("monospace 9"))

        self.log_textview.get_buffer().set_text(get_recent_logs())
        scrolled.add(self.log_textview)
        box.pack_start(scrolled, True, True, 0)

        # Button area
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        button_box.set_halign(Gtk.Align.END)

        refresh_button = Gtk.Button(label="Refresh")
        refresh_button.connect("clicked", self._on_refresh_logs)
        button_box.pack_start(refresh_button, False, False, 0)

        clear_button = Gtk.Button(label="Clear Logs")
        clear_button.connect("clicked", self._on_clear_logs)
        button_box.pack_start(clear_button, False, False, 0)

        box.pack_start(button_box, False, False, 0)

        return box

    def _on_refresh_logs(self, button):
        """Refresh logs."""
        self.log_textview.get_buffer().set_text(get_recent_logs())

    def _on_clear_logs(self, button):
        """Clear the log file."""
        try:
            if LOG_FILE.exists():
                LOG_FILE.write_text("")
            self.log_textview.get_buffer().set_text("Logs cleared")
        except Exception as e:
            self.log_textview.get_buffer().set_text(f"Failed to clear logs: {e}")

    def get_selected_model(self):
        """Get the selected model ID."""
        return self.selected_model_id

    def get_hotwords(self):
        """Get hotword configuration."""
        buffer = self.hotwords_textview.get_buffer()
        start, end = buffer.get_bounds()
        return buffer.get_text(start, end, False)

    def apply_settings(self):
        """Apply settings."""
        # Save hotwords
        save_hotwords(self.get_hotwords())

        # If model changed, return the model ID to switch to
        if self.selected_model_id != self.current_model_id:
            return {"model_changed": True, "new_model": self.selected_model_id}

        return {"model_changed": False}


def show_settings_dialog(parent=None, model_presets=None, current_model_id=None):
    """
    Show the settings dialog.

    Args:
        parent: Parent window
        model_presets: Available model presets
        current_model_id: Current model ID

    Returns:
        dict: {"model_changed": bool, "new_model": str or None}
    """
    if not HAS_GTK:
        print("GTK not available, cannot show settings dialog")
        return None

    dialog = SettingsDialog(parent, model_presets, current_model_id)
    response = dialog.run()

    result = None
    if response == Gtk.ResponseType.APPLY:
        result = dialog.apply_settings()

    dialog.destroy()
    return result


if __name__ == "__main__":
    # Test dialog - directly use config from model_presets.py
    if HAS_GTK:
        from model_presets import MODEL_PRESETS, DEFAULT_MODEL

        result = show_settings_dialog(None, MODEL_PRESETS, DEFAULT_MODEL)
        print(f"Settings result: {result}")
    else:
        print("GTK not available")
