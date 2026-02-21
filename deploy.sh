#!/bin/bash
# Deploy script - deploy changes to system
# Usage: ./deploy.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/.local/share/voice-input"

echo "=== Deploying voice-input ==="

# 1. Stop existing daemon
echo "[1/4] Stopping daemon..."
voice-input kill 2>/dev/null || true
sleep 1

# 2. Copy icons (if changed)
echo "[2/4] Updating icons..."
if [ -d "$SCRIPT_DIR/icons" ]; then
    mkdir -p "$INSTALL_DIR/icons"
    cp -r "$SCRIPT_DIR/icons/"* "$INSTALL_DIR/icons/"
    echo "  - Icons updated"
fi

# 3. Verify symlink exists
echo "[3/4] Checking symlink..."
if [ ! -L "$INSTALL_DIR/voice_input.py" ]; then
    ln -sf "$SCRIPT_DIR/voice_input.py" "$INSTALL_DIR/voice_input.py"
    echo "  - Recreated symlink"
else
    echo "  - Symlink exists"
fi

# 4. Start daemon
echo "[4/4] Starting daemon..."
voice-input daemon &

echo ""
echo "=== Deployment complete ==="
echo "Daemon is starting in the background, please wait ~20-30 seconds"
echo "Use 'voice-input status' to check status"
