#!/usr/bin/env bash
# Sightline: copy the INT4 weights from /mnt/f (slow 9p) to WSL ext4 once, then serve from there.
set -euo pipefail
SRC=/mnt/f/models/muse-glimmer-30b-int4-w4a16
DST=$HOME/models/muse-glimmer-30b-int4-w4a16
mkdir -p "$DST"
t0=$(date +%s)
echo "[$(date '+%F %T')] copy $SRC -> $DST"
rsync -a --exclude .cache "$SRC/" "$DST/"
src_sz=$(du -sb --exclude=.cache "$SRC" | cut -f1); dst_sz=$(du -sb "$DST" | cut -f1)
echo "[$(date '+%F %T')] copy done in $(( $(date +%s)-t0 )) s src=$src_sz dst=$dst_sz"
[ "$src_sz" = "$dst_sz" ] || { echo "FAIL: size mismatch"; exit 1; }
MODEL_PATH="$DST" exec bash /mnt/f/models/wsl/serve_sglang.sh