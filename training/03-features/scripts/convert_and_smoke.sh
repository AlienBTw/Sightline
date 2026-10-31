#!/bin/bash
set -x
source ~/vjepa-venv/bin/activate
W=~/models/vjepa/vjepa2_1_vitl_dist_vitG_384.pt
E=~/models/vjepa/vjepa2_1_vitl_384_encoder_fp16.pt
while [ ! -f $W ]; do sleep 10; done
grep -q PDL_OK /mnt/f/models/logs/dl_vjepa.log || { echo download failed; exit 1; }
sha256sum $W
[ -f $E ] || python "/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/03-features/convert_checkpoint.py" $W $E
ls -la ~/models/vjepa

cd "/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/03-features"
python extract_features.py --limit 10 --world 1 --rank 0 --out /mnt/f/sightline_features/smoke --workers 3
wait
echo SMOKE_DONE