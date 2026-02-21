#!/bin/bash
# Run pytest in sandbox (least privilege principle)

cd "$(dirname "$0")/.."

docker run --rm \
  --user sandbox \
  -v "$(pwd)":/workspace:ro \
  --network none \
  --memory=2g \
  --cpus=2 \
  --pids-limit=100 \
  --cap-drop=ALL \
  --security-opt=no-new-privileges:true \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  voice-input-sandbox \
  pytest tests/ -v -m "not real_model and not gui" "$@"
