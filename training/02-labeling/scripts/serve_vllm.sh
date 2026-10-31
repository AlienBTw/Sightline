#!/usr/bin/env bash
# Sightline: serve Muse Glimmer 30B INT4 (GPTQ W4A16 compressed-tensors) with vLLM in WSL, TP=2 on 2x RTX 3090.
# Log: F:\models\logs\vllm_server.log ; endpoint http://localhost:30001/v1 ; model name muse-glimmer-30b-int4-vllm
# No --reasoning-parser: label_glimmer.py (backend vllm, reasoning none) pre-fills "{" in the to=user channel,
# so the output is the JSON answer only. --disable-custom-all-reduce: no P2P between the 3090s under WSL.
set -euo pipefail
source ~/vllm-venv/bin/activate
export HF_HOME=/mnt/f/hf-cache
# FlashInfer top-k/top-p sampler needs nvcc JIT (no CUDA toolkit in WSL) -> use the PyTorch sampler
export VLLM_USE_FLASHINFER_SAMPLER=0
MODEL_PATH=${MODEL_PATH:-$HOME/models/muse-glimmer-30b-int4-w4a16}
EXTRA=${EXTRA:-}
echo "[$(date '+%F %T')] vllm launch model=$MODEL_PATH extra=$EXTRA"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
exec vllm serve "$MODEL_PATH" \
  --served-model-name muse-glimmer-30b-int4-vllm \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 --port 30001 \
  --generation-config auto \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.88 \
  --limit-mm-per-prompt '{"image": 1}' \
  --disable-custom-all-reduce \
  $EXTRA