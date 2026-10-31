#!/bin/bash
# 5-fold CV of the h3_wide config (best on the official test) across both GPUs
source ~/vjepa-venv/bin/activate
cd "/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/04-head"
run() { python train_head.py train --gpu $1 --cv 5 --fold $2 --name h3_wide_cv5_f$2 --hidden 512 --dropout 0.2 --lr 5e-4 > /mnt/f/models/logs/head_h3_wide_cv5_f$2.log 2>&1; }
( run 0 0; run 0 1; run 0 2 ) & ( run 1 3; run 1 4 ) & wait
python summarize_head.py > /dev/null
echo CV_WIDE_DONE