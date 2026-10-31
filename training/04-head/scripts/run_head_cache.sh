#!/bin/bash
source ~/vjepa-venv/bin/activate
cd "/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/04-head"
python train_head.py cache
echo CACHE_EXIT $?