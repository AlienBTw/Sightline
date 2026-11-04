#!/usr/bin/env bash
# Sightline: serve Muse Glimmer 30B INT4 (W4A16, compressed-tensors) with SGLang in WSL, image input on.
# Started detached from Windows; log: <repo>/work/logs/sglang_server.log ; endpoint http://localhost:30000/v1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORK="${SIGHTLINE_WORK_DIR:-$REPO/work}"
source ~/sglang-venv/bin/activate
export HF_HOME="${HF_HOME:-$WORK/hf-cache}"
# triton attention + pytorch sampling: flashinfer 0.7.2's JIT (bundled cccl) rejects pip nvcc 13.4 headers.
# CUDA graphs off: decode-graph capture killed rank 0 with SIGFPE (exit -8) on 2026-10-31.
# custom all-reduce off: the two 3090s have no peer access under WSL.
# no system CUDA toolkit in WSL: use the pip one (nvidia/cu13, nvcc 13.4) for SGLang's JIT kernels (gptq_marlin_repack)
export CUDA_HOME=$HOME/sglang-venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
# needs patch_symm.sh applied to the venv: multimem/symmetric-memory logits all-gather SIGFPEs under WSL
export SGLANG_DISABLE_MULTIMEM_AG=1
TP=${TP:-2}
EXTRA=${EXTRA:-}
MODEL_PATH=${MODEL_PATH:-$WORK/models/muse-glimmer-30b-int4-w4a16}
echo "[$(date '+%F %T')] launch tp=$TP extra=$EXTRA model=$MODEL_PATH"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name muse-glimmer-30b-int4 \
  --tp-size "$TP" \
  --host 0.0.0.0 --port 30000 \
  --reasoning-parser muse \
  --context-length 8192 \
  --mem-fraction-static 0.85 \
  --max-running-requests 64 \
  --enable-multimodal \
  --attention-backend triton --sampling-backend pytorch \
  --disable-custom-all-reduce \
  --disable-cuda-graph \
  $EXTRA