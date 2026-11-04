#!/usr/bin/env bash
# Copy Muse Glimmer INT4 weights into the work dir when MODEL_PATH points elsewhere, then serve.
# Weights default to $SIGHTLINE_WORK_DIR/models/muse-glimmer-30b-int4-w4a16
# (SIGHTLINE_WORK_DIR defaults to <repo>/work).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORK="${SIGHTLINE_WORK_DIR:-$REPO/work}"
DST="$WORK/models/muse-glimmer-30b-int4-w4a16"
SRC="${MODEL_PATH:-$DST}"
mkdir -p "$DST"
if [ "$(cd "$SRC" && pwd)" != "$(cd "$DST" && pwd)" ]; then
  t0=$(date +%s)
  echo "[$(date '+%F %T')] copy $SRC -> $DST"
  rsync -a --exclude .cache "$SRC/" "$DST/"
  src_sz=$(du -sb --exclude=.cache "$SRC" | cut -f1); dst_sz=$(du -sb "$DST" | cut -f1)
  echo "[$(date '+%F %T')] copy done in $(( $(date +%s)-t0 )) s src=$src_sz dst=$dst_sz"
  [ "$src_sz" = "$dst_sz" ] || { echo "FAIL: size mismatch"; exit 1; }
fi
MODEL_PATH="$DST" exec bash "$SCRIPT_DIR/serve_sglang.sh"
