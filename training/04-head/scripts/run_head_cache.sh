#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source ~/vjepa-venv/bin/activate
cd "$REPO/training/04-head"
python train_head.py cache
echo CACHE_EXIT $?
