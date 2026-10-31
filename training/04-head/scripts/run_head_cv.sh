#!/bin/bash
# 5-fold CV of the base config over all 658 unique Voxel51 clips (GPU 0), then summary
source ~/vjepa-venv/bin/activate
cd "/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/04-head"
for f in 0 1 2 3 4; do
  python train_head.py train --gpu 0 --cv 5 --fold $f --name h1_base_cv5_f$f > /mnt/f/models/logs/head_h1_base_cv5_f$f.log 2>&1
done
echo CV_DONE