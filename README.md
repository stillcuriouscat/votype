# Votype

Global voice typing for Linux -- speak and text appears at your cursor in any application. Powered by offline ASR models, no cloud needed.

## Features

- **Offline speech recognition** with multiple ASR model options (FunASR, FireRedASR)
- **Chinese-English mixed language** support (code-switching)
- **System tray icon** with real-time recording status indicators
- **Hotkey-triggered** push-to-talk workflow
- **Smart paste detection** -- auto-detects terminal vs GUI for correct paste method (`Ctrl+Shift+V` vs `Ctrl+V`)
- **GTK settings dialog** for model switching and hotword configuration
- **Daemon mode** keeps model loaded in memory for instant recognition (~800MB-1GB RAM)

## Supported Models

| Model | Description | Memory |
|-------|-------------|--------|
| `fun-asr-nano` (default) | Latest end-to-end model, 31 languages, high accuracy | ~800MB |
| `paraformer` | Fast and lightweight, Chinese-optimized | ~800MB |
| `sensevoice` | Chinese-English mixed, multilingual | ~900MB |
| `firered-asr` | Xiaohongshu AED model, Chinese SOTA | ~1GB |

## Prerequisites

- Ubuntu 22.04+ with X11 session
- Python 3.8+
- NVIDIA GPU recommended (CUDA support for faster inference)
- System packages: `arecord`, `xdotool`, `xclip`, `libnotify-bin`

## Installation

```bash
cd voice_input
chmod +x install.sh
./install.sh
```

The install script will:

1. Install system dependencies (`ffmpeg`, `xdotool`, `xclip`, `portaudio`, etc.)
2. Create a Python virtual environment at `~/.local/share/voice-input/venv`
3. Install Python dependencies (FunASR, ModelScope, etc.)
4. Create a `voice-input` command in `~/.local/bin/`

Models are downloaded automatically on first use (~1GB per model).

### Installing Additional Model Frameworks

To install dependencies for all supported model frameworks (FunASR, Transformers, FireRedASR):

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

## Usage

### CLI Commands

```bash
voice-input daemon                        # Start the daemon (default model: fun-asr-nano)
voice-input daemon --model sensevoice     # Start with a specific model
voice-input toggle                        # Toggle recording (auto-starts daemon if needed)
voice-input status                        # Show current status and model info
voice-input models                        # List available models
voice-input kill                          # Stop the daemon
```

### Setting Up a Global Hotkey

1. Open **System Settings > Keyboard > Keyboard Shortcuts > Custom Shortcuts**
2. Add a new shortcut:
   - **Name:** `Voice Input`
   - **Command:** `/home/YOUR_USERNAME/.local/bin/voice-input toggle`
   - **Shortcut:** `Super+Space` (or any key combination you prefer)

### Workflow

1. **Press hotkey** -- if the daemon is not running, it starts automatically (first launch takes ~20-30s to load the model)
2. **Press hotkey again** -- recording starts (tray icon turns red)
3. **Speak** (supports Chinese-English mixed speech)
4. **Press hotkey once more** -- recording stops, speech is transcribed, and the result is typed at your cursor position

### System Tray Icon

| Icon Color | State |
|------------|-------|
| Gray | Idle, ready for input |
| Red | Recording in progress |
| Orange | Processing / transcribing |

Right-click the tray icon to:

- Switch ASR model
- Open settings (configure hotwords, view logs)
- Quit the application

### Terminal Support

The tool auto-detects the active window type:

- **Terminal windows** (gnome-terminal, konsole, xfce4-terminal, etc.): pastes via `Ctrl+Shift+V`
- **GUI applications** (browsers, editors, etc.): pastes via `Ctrl+V`

The original clipboard content is preserved after pasting.

## Troubleshooting

### No audio recorded

```bash
# Check available microphones
arecord -l
# Test recording and playback
arecord -d 3 test.wav && aplay test.wav
```

### Slow first run

This is expected. The first launch downloads the ASR model (~1GB) and loads it into memory (~20-30 seconds). Subsequent starts are faster since the model is cached locally.

### Transcribed text not appearing

Make sure `xdotool` and `xclip` are installed:

```bash
sudo apt install xdotool xclip
```

### GPU acceleration

The default configuration uses CUDA (`cuda:0`). If you do not have an NVIDIA GPU or encounter CUDA errors, you can modify `model_presets.py` and change `DEVICE` to `"cpu"`.

## Development

### Deploying Changes

After modifying source code, run the deploy script to apply changes:

```bash
./deploy.sh
```

This will stop the existing daemon, update icons, verify symlinks, and restart the daemon.

### Logs

```bash
# Daemon log (startup, model loading, errors)
cat /tmp/voice-input-daemon.log

# Notification log (recording events, transcription results)
cat /tmp/voice-input-notify.log
```

### Project Structure

```
voice_input.py        # Main entry point (CLI, daemon, recording logic)
model_presets.py      # Model configuration (presets, device, hotwords)
model_configs.py      # Model loading and inference logic
settings_dialog.py    # GTK settings dialog
install.sh            # System installation script
deploy.sh             # Development deployment script
icons/                # Tray icon assets (idle, recording, processing)
```

## Uninstall

```bash
# Stop the daemon
voice-input kill

# Remove installed files
rm -rf ~/.local/share/voice-input
rm ~/.local/bin/voice-input
rm -rf ~/.config/voice-input
```

## License

MIT
