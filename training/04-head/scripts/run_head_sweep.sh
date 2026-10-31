#!/bin/bash
# Head config sweep: two queues, one per GPU. Logs: /mnt/f/models/logs/head_<name>.log
source ~/vjepa-venv/bin/activate
cd "/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/04-head"
L=/mnt/f/models/logs
q0() {
  python train_head.py train --gpu 0 --name h1_base_s0 --seed 0 > $L/head_h1_base_s0.log 2>&1
  python train_head.py train --gpu 0 --name h2_voxel_only --use-zenodo 0 > $L/head_h2_voxel_only.log 2>&1
  python train_head.py train --gpu 0 --name h1_base_s1 --seed 1 > $L/head_h1_base_s1.log 2>&1
}
q1() {
  python train_head.py train --gpu 1 --name h3_wide --hidden 512 --dropout 0.2 --lr 5e-4 > $L/head_h3_wide.log 2>&1
  python train_head.py train --gpu 1 --name h4_no_patch_loss --patch-loss 0 > $L/head_h4_no_patch_loss.log 2>&1
  python train_head.py train --gpu 1 --name h1_base_s2 --seed 2 > $L/head_h1_base_s2.log 2>&1
}
q0 & q1 & wait
echo SWEEP_DONE