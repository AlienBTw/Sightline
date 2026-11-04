#!/usr/bin/env bash
# Sightline: vLLM in its own venv (keeps the patched ~/sglang-venv untouched). Log: <repo>/work/logs/vllm_install.log (SIGHTLINE_WORK_DIR, default <repo>/work).
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd ~
echo "[$(date '+%F %T')] start"
uv venv --python 3.12 ~/vllm-venv
source ~/vllm-venv/bin/activate
uv pip install "vllm>=0.28.0" --torch-backend=auto
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"
python - <<'EOF'
from vllm.model_executor.models.registry import ModelRegistry
a = ModelRegistry.get_supported_archs()
print("glimmer archs:", [x for x in a if "Glimmer" in x])
EOF
echo "[$(date '+%F %T')] done"