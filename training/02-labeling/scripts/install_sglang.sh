#!/usr/bin/env bash
# Sightline: install SGLang into a user venv in WSL Ubuntu (no sudo). Log: F:\models\logs\sglang_install.log
set -euo pipefail
echo "[$(date '+%F %T')] start"
if ! command -v uv >/dev/null && [ ! -x ~/.local/bin/uv ]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
uv --version
[ -d ~/sglang-venv ] || uv venv ~/sglang-venv --python 3.12
source ~/sglang-venv/bin/activate
export UV_HTTP_TIMEOUT=300
uv pip install "sglang==0.5.23"
python -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0))"
python -c "import sglang.srt.models as m, pkgutil; print('glimmer modules:', [x.name for x in pkgutil.iter_modules(m.__path__) if 'glimmer' in x.name or 'muse' in x.name])"
echo "[$(date '+%F %T')] done"