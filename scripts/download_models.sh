#!/usr/bin/env bash
# Simple download helper for pretrained models used by this repo.
# Edit the URLs below to point to your hosted checkpoint files (GitHub Release, S3, Google Drive direct links, etc.)

set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT_DIR/pretrained_models"

# Example: Raft-stereo
mkdir -p "$ROOT_DIR/pretrained_models/raft_stereo"
RAFT_URL="REPLACE_WITH_RAFT_CHECKPOINT_URL"
RAFT_DEST="$ROOT_DIR/pretrained_models/raft_stereo/raft-stereo_20000.pth"

if [ -f "$RAFT_DEST" ]; then
  echo "RAFT checkpoint already exists: $RAFT_DEST"
else
  if [ "$RAFT_URL" = "REPLACE_WITH_RAFT_CHECKPOINT_URL" ]; then
    echo "Please edit scripts/download_models.sh and set RAFT_URL to a real URL or use GitHub Releases."
  else
    echo "Downloading RAFT checkpoint..."
    curl -L -o "$RAFT_DEST" "$RAFT_URL"
    echo "Saved to $RAFT_DEST"
  fi
fi

# Add other models similarly (fast_acvnet, bgnet, gwcnet, etc.)

cat <<EOF
Done. After downloading, verify the paths in config.py point to the files under pretrained_models/.
EOF
