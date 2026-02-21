#!/bin/bash
# Global voice input tool installation script
# Uses FunASR (Paraformer) - optimized for Chinese-English mixed speech

set -e

echo "=== Installing global voice input tool ==="

# 1. Install system dependencies
echo "[1/4] Installing system dependencies..."
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg libsndfile1 xdotool xclip portaudio19-dev libnotify-bin

# 2. Create virtual environment (using system Python, not anaconda)
echo "[2/4] Creating Python virtual environment..."
INSTALL_DIR="$HOME/.local/share/voice-input"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$INSTALL_DIR"

# Use system Python (to ensure access to gi module)
SYS_PYTHON="/usr/bin/python3"
if [ ! -f "$SYS_PYTHON" ]; then
    SYS_PYTHON="python3"
fi

$SYS_PYTHON -m venv --system-site-packages "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"

# 3. Install Python dependencies
echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

# 4. Create symlinks and copy icons
echo "[4/4] Installing scripts and icons..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ln -sf "$SCRIPT_DIR/voice_input.py" "$INSTALL_DIR/voice_input.py"
chmod +x "$SCRIPT_DIR/voice_input.py"

# Copy custom icons
if [ -d "$SCRIPT_DIR/icons" ]; then
    cp -r "$SCRIPT_DIR/icons" "$INSTALL_DIR/"
    echo "  - Icons copied to $INSTALL_DIR/icons/"
fi

# Create launch script
cat > "$INSTALL_DIR/start.sh" << 'EOF'
#!/bin/bash
source "$HOME/.local/share/voice-input/venv/bin/activate"
python "$HOME/.local/share/voice-input/voice_input.py" "$@"
EOF
chmod +x "$INSTALL_DIR/start.sh"

# Create desktop shortcut
mkdir -p "$HOME/.local/bin"
ln -sf "$INSTALL_DIR/start.sh" "$HOME/.local/bin/voice-input"

echo ""
echo "=== Installation complete! ==="
echo ""
echo "Usage:"
echo "  voice-input start    # Start recording"
echo "  voice-input stop     # Stop recording and input text"
echo "  voice-input toggle   # Toggle recording state"
echo ""
echo "Recommended: set up a global hotkey:"
echo "  System Settings > Keyboard > Custom Shortcuts"
echo "  Command: $HOME/.local/bin/voice-input toggle"
echo "  Shortcut: Super+Space or any combination you prefer"
echo ""
echo "The ASR model (~1GB) will be downloaded automatically on first run"
