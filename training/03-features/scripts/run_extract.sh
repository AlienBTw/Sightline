#!/bin/bash
# usage: run_extract.sh [extra args...]   -> 2 detached ranks (one per GPU)
source ~/vjepa-venv/bin/activate
cd "/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/03-features"
for r in 0 1; do
  nohup python extract_features.py --rank $r --world 2 "$@" > /mnt/f/models/logs/extract_r$r.log 2>&1 &
  echo "rank $r pid $!"
done