#!/bin/bash
# usage: run_extract.sh [extra args...]   -> 2 detached ranks (one per GPU)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORK="${SIGHTLINE_WORK_DIR:-$REPO/work}"
LOG="${SIGHTLINE_LOG_DIR:-$WORK/logs}"
mkdir -p "$LOG"
source ~/vjepa-venv/bin/activate
cd "$REPO/training/03-features"
for r in 0 1; do
  nohup python extract_features.py --rank $r --world 2 "$@" > "$LOG/extract_r$r.log" 2>&1 &
  echo "rank $r pid $!"
done
