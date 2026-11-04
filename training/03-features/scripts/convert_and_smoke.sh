#!/bin/bash
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORK="${SIGHTLINE_WORK_DIR:-$REPO/work}"
LOG="${SIGHTLINE_LOG_DIR:-$WORK/logs}"
source ~/vjepa-venv/bin/activate
W="$WORK/models/vjepa/vjepa2_1_vitl_dist_vitG_384.pt"
E="$WORK/models/vjepa/vjepa2_1_vitl_384_encoder_fp16.pt"
while [ ! -f "$W" ]; do sleep 10; done
grep -q PDL_OK "$LOG/dl_vjepa.log" || { echo download failed; exit 1; }
sha256sum "$W"
[ -f "$E" ] || python "$REPO/training/03-features/convert_checkpoint.py" "$W" "$E"
ls -la "$WORK/models/vjepa"

cd "$REPO/training/03-features"
python extract_features.py --limit 10 --world 1 --rank 0 --out "$WORK/features/smoke" --workers 3
wait
echo SMOKE_DONE
