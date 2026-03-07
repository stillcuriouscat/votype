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

| Model | Description | VRAM |
|-------|-------------|------|
| `firered-asr` (default) | Xiaohongshu AED model, Chinese SOTA, auto-punctuation (FireRedPunc) | ~7.5GB |
| `fun-asr-nano` | Latest end-to-end model, 31 languages, high accuracy | ~800MB |
| `paraformer` | Fast and lightweight, Chinese-optimized (CPU mode) | ~800MB |
| `sensevoice` | Chinese-English mixed, multilingual | ~900MB |

## Prerequisites

- Ubuntu 22.04+ with X11 session
- Python 3.8+
- NVIDIA GPU recommended (CUDA support for faster inference)
- System packages: `pw-record` (PipeWire), `xdotool`, `xclip`, `libnotify-bin`
- **Note:** `arecord` (ALSA) is supported as a fallback for systems without PipeWire

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
voice-input daemon                        # Start the daemon (default model: firered-asr)
voice-input daemon --model sensevoice     # Start with a specific model
voice-input toggle                        # Toggle recording (auto-starts daemon if needed)
voice-input status                        # Show current status and model info
voice-input models                        # List available models
voice-input post-processors               # List available LLM post-processors
voice-input post-processor <id>           # Switch LLM post-processor (none/chinese-text-correction/qwen3-0.6b/minicpm4-0.5b)
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

- **Kitty terminal**: uses native `kitty @ send-text` API for direct text injection -- bypasses the clipboard entirely, so your clipboard is never touched
- **Other terminals** (gnome-terminal, konsole, xfce4-terminal, etc.): pastes via `Ctrl+Shift+V`
- **GUI applications** (browsers, editors, etc.): pastes via `Ctrl+V`

The original clipboard content is preserved after pasting (except for Kitty, which does not use the clipboard at all).

#### Kitty Configuration

To enable native text injection in Kitty, add the following to your `~/.config/kitty/kitty.conf`:

```
allow_remote_control socket-only
listen_on unix:/tmp/kitty-socket
```

Restart Kitty after making this change. Votype will automatically detect the Kitty socket and use the native API.

## Troubleshooting

### No audio recorded

```bash
# Test recording and playback with PipeWire (preferred)
pw-record --target 0 test.wav & sleep 3 && kill %1 && pw-play test.wav

# Fallback: test with ALSA (non-PipeWire systems)
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

### System tray icon not showing

The tray icon requires `PyGObject` and `AyatanaAppIndicator3`, which are system packages:

```bash
sudo apt install python3-gi gir1.2-ayatanaappindicator3-0.1
```

The virtual environment **must** be created with `--system-site-packages` so it can access these system libraries. If you created the venv without this flag, either recreate it:

```bash
/usr/bin/python3 -m venv --system-site-packages ~/.local/share/voice-input/venv
```

Or edit `~/.local/share/voice-input/venv/pyvenv.cfg` and change:

```
include-system-site-packages = true
```

### FunASR import errors (Fun-ASR-Nano)

FunASR 1.3.x has a known bug where `fun_asr_nano/model.py` uses implicit relative imports that fail in standard virtual environments. Symptoms:

- `cannot access local variable 'AutoTokenizer'` -- missing `transformers` package
- `cannot access local variable 'get_tokenizer'` -- missing `tiktoken` package
- `FunASRNano is not registered` -- broken imports in FunASR source

**Fix missing packages:**

```bash
~/.local/share/voice-input/venv/bin/pip install transformers tiktoken
```

**Fix FunASR relative imports** (required for Fun-ASR-Nano model):

Edit `~/.local/share/voice-input/venv/lib/python3.*/site-packages/funasr/models/fun_asr_nano/model.py`, change:

```python
# Before (broken)
from ctc import CTC
from tools.utils import forced_align

# After (fixed)
from .ctc import CTC
from .tools.utils import forced_align
```

### Transcription fails with error "0"

This is a FunASR bug where the VAD post-processing code assumes timestamps are `[start, end]` lists, but Fun-ASR-Nano returns dict format (`{"start_time": ..., "end_time": ...}`).

**Fix:** Edit `~/.local/share/voice-input/venv/lib/python3.*/site-packages/funasr/auto/auto_model.py`, find the `inference_with_vad` method around line 557, and replace:

```python
for t in restored_data[j][k]:
    t[0] += vadsegments[j][0]
    t[1] += vadsegments[j][0]
```

With:

```python
for t in restored_data[j][k]:
    if isinstance(t, dict):
        if "start_time" in t:
            t["start_time"] += vadsegments[j][0]
        if "end_time" in t:
            t["end_time"] += vadsegments[j][0]
    else:
        t[0] += vadsegments[j][0]
        t[1] += vadsegments[j][0]
```

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
voice_input.py              # Main entry point (CLI, daemon, recording logic)
model_presets.py            # Model configuration (presets, device, hotwords)
model_configs.py            # Model loading and inference logic
post_processor_presets.py   # Post-processor definitions (LLM text refinement)
post_processor_configs.py   # Post-processor loading and inference logic
settings_dialog.py          # GTK settings dialog
install.sh                  # System installation script
deploy.sh                   # Development deployment script
icons/                      # Tray icon assets (idle, recording, processing)
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
