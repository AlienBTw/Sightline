# 00 · Setup

**Goal:** a reproducible environment for phases 01–07 on the training PC.

**Inputs:** none.
**Outputs:** none. The check script is read-only.

## How to run (repo root, PowerShell)

```powershell
pip install -r training/requirements.txt
python training/00-setup/check_env.py
```

The script prints the Python version, key packages, GPUs (through torch), Ollama models and free disk space. It exits with 1 and lists the problems if anything needed is missing.

## Requirements

- Python ≥ 3.11
- `numpy`, `opencv-python-headless`, `PyYAML`, `pytest` (from `training/requirements.txt`)
- PyTorch with CUDA for phases 03–04
- Stage 1 labelling: vLLM serving `RedHatAI/Muse-Glimmer-30B-INT4` as `muse-glimmer-30b-int4-vllm` (tensor parallel on both GPUs). `check_env.py` also prints Ollama models. Ollama `muse-glimmer:30b` was an earlier attempt, not the full run
- Disk space: Voxel51 needs about 9.3 GiB (10.0 GB). Zenodo needs about 38.6 GiB of parts plus about as much again once extracted.

## Results

Oct 30, 22:30: on the training PC, `check_env.py` → **OK**. It found Python 3.11.8, torch 2.6.0+cu124, 2× RTX 3090 (24 GiB), Ollama serving `muse-glimmer:30b`, and 146 GiB free. That Ollama tag was an earlier attempt. The full labelling run later used vLLM and `RedHatAI/Muse-Glimmer-30B-INT4`.

## Known issues

- Git for Windows isn't on the system PATH. Cursor's built-in git is used for commits; a portable MinGit 2.47.1 is also present at `%LOCALAPPDATA%\Programs\MinGit\cmd\git.exe` (used for the Oct 31 push).
- On Windows, `zoneinfo` has no tz database without the `tzdata` package. The scripts therefore use a fixed UTC+1 offset for Tunis, which is correct because Tunisia has no DST.
