# 02-labeling/scripts: inference servers and the full-run launcher

Copies of the helper scripts that ran the Glimmer labeling on the training PC (Oct 31, 2026). The live copies sit outside the repo in `F:\models\wsl\` (logs in `F:\models\logs\`). They are kept here so the labeling run can be reproduced. No secrets are used or stored: the model weights are public (`RedHatAI/Muse-Glimmer-30B-INT4`, Apache-2.0) and no token is needed.

Paths are the training PC's (`F:\models\...`, `/mnt/f/...`, `~/vllm-venv`, `~/sglang-venv` inside WSL2 Ubuntu-24.04). Change them if you run this somewhere else.

| Script | Runs in | What it does |
|---|---|---|
| `dl_glimmer_int4.py` | Windows Python | Downloads `RedHatAI/Muse-Glimmer-30B-INT4` at revision `397b1ee1` (22.2 GB) to `F:\models\muse-glimmer-30b-int4-w4a16`. Xet is turned off because it stalled; plain HTTP resumes. |
| `install_vllm.sh` | WSL | Creates `~/vllm-venv` with uv and installs `vllm>=0.28` (0.32.0 was installed), then checks that `MuseGlimmerForConditionalGeneration` is registered. |
| `serve_vllm.sh` | WSL | **The backend used for the full run.** `vllm serve`, TP=2 on both RTX 3090s, port 30001, model name `muse-glimmer-30b-int4-vllm`, 8k context, 32 sequences, CUDA graphs on, `--disable-custom-all-reduce` (no P2P under WSL), `VLLM_USE_FLASHINFER_SAMPLER=0` (the FlashInfer sampler needs nvcc). It listens on `0.0.0.0`, so keep it off untrusted networks. |
| `stop_vllm.sh` | WSL | SIGINT to `vllm serve`, waits, then prints GPU memory. |
| `label_full_vllm.ps1` | Windows | The detached full-run launcher (v2 prompt, reasoning none, concurrency 32). It writes start/finish/exit code to `label_full_vllm.status.log`. |
| `check_label_full.ps1` | Windows | Progress check during the run: clips done, new frames, rate, ETA, errors. The runner PID (7400) is hard-coded for that run. |
| `install_sglang.sh` | WSL | Creates `~/sglang-venv` and installs `sglang==0.5.23` (the first attempt; superseded by vLLM). |
| `cudafix.sh` | WSL | Makes the pip CUDA toolkit (`nvidia/cu13`, nvcc 13.4) usable as `CUDA_HOME` for SGLang's JIT kernels (symlinks `lib64`, `libcudart.so`, the `libcuda.so` stub). |
| `patch_symm.sh` | WSL | Patches SGLang's multimem logits all-gather (`triton_symm_mem_ag.py`) to fall back to NCCL when `SGLANG_DISABLE_MULTIMEM_AG=1`, because it crashed with SIGFPE on 2x3090 under WSL. It keeps a `.orig` copy, and a reinstall drops the patch. |
| `serve_sglang.sh` | WSL | SGLang server, TP=2, port 30000, triton attention, PyTorch sampler, CUDA graphs off, custom all-reduce off. |
| `copy_and_serve.sh` | WSL | Copies the weights from `/mnt/f` to WSL ext4 once (loading went from ~200 s to 21 s), then starts SGLang. |
| `stop_sglang.sh` | WSL | Stops SGLang. This copy matches by process name; the original used a hard-coded PID. |

## Reproduce the full run

```powershell
# 1. Weights (once)
python training/02-labeling/scripts/dl_glimmer_int4.py
wsl -d Ubuntu-24.04 -- bash -lc "rsync -a --exclude .cache /mnt/f/models/muse-glimmer-30b-int4-w4a16/ ~/models/muse-glimmer-30b-int4-w4a16/"
# 2. vLLM (once), then start the server (ready in about 1.5-2.5 min)
wsl -d Ubuntu-24.04 -- bash -lc "bash /mnt/f/models/wsl/install_vllm.sh"
wsl.exe -d Ubuntu-24.04 -- bash -lc "bash /mnt/f/models/wsl/serve_vllm.sh > /mnt/f/models/logs/vllm_server.log 2>&1"
# 3. Label (from the repo root); about 2 h 36 min for 658 clips at concurrency 32
python -X utf8 training/02-labeling/label_glimmer.py --backend vllm --sources voxel51_safe_unsafe --prompt training/02-labeling/prompts/factors_v2.md --reasoning-effort none --concurrency 32 --out training/02-labeling/outputs/weak_labels_v2_vllm.jsonl
python training/02-labeling/summarize_labels.py
# 4. Free the GPUs
wsl.exe -d Ubuntu-24.04 bash /mnt/f/models/wsl/stop_vllm.sh
```
