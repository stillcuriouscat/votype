#!/bin/bash
# Voice recognition model dependency installation script
# Automatically install dependencies for all supported frameworks

set -e  # Exit immediately on error

echo "========================================================================"
echo "Voice Recognition Model Dependency Installation Script"
echo "========================================================================"
echo ""

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    echo "Please install Python 3.8+ first"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Detected Python version: $PYTHON_VERSION"
echo ""

# 1. Install FunASR dependencies
echo "========================================================================"
echo "[1/3] Installing FunASR framework dependencies"
echo "========================================================================"
echo "Includes: funasr, modelscope"
echo ""
pip install funasr modelscope || {
    echo "Warning: FunASR installation failed"
    echo "You can ignore this error if you don't need Fun-ASR-Nano/Paraformer/SenseVoice"
}
echo ""

# 2. Install Transformers dependencies
echo "========================================================================"
echo "[2/3] Installing Transformers framework dependencies"
echo "========================================================================"
echo "Includes: transformers>=4.37.0, librosa, torch"
echo ""
echo "Note: PyTorch installation may take a while (~1-2GB)"
echo ""
pip install 'transformers>=4.37.0' librosa torch || {
    echo "Warning: Transformers dependency installation failed"
    echo "You can ignore this error if you don't need Qwen2-Audio"
}
echo ""

# 3. Install FireRedASR dependencies
echo "========================================================================"
echo "[3/3] Installing FireRedASR framework dependencies"
echo "========================================================================"
echo "Includes: fireredasr"
echo ""
echo "Attempting to install fireredasr from PyPI..."
pip install fireredasr || {
    echo ""
    echo "Installation from PyPI failed. FireRedASR may need to be installed from source"
    echo ""
    echo "If you need to use FireRedASR, please install manually:"
    echo "  git clone https://github.com/FireRedTeam/FireRedASR.git"
    echo "  cd FireRedASR"
    echo "  pip install -e ."
    echo ""
    echo "Or visit the official repository for the latest installation instructions:"
    echo "  https://github.com/FireRedTeam/FireRedASR"
}
echo ""

# Installation complete
echo "========================================================================"
echo "Dependency installation complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Download models:"
echo "   python download_models.py"
echo ""
echo "2. Or use the installed dependencies and skip dependency checks:"
echo "   python download_models.py qwen2-audio --skip-deps"
echo ""
echo "3. Start the daemon:"
echo "   voice-input daemon --model <model-name>"
echo ""
