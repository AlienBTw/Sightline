# Sightline training pipeline

Everything to do with training the Sightline models lives here, with one folder per phase. Each phase folder has a `README.md` with the goal, inputs, outputs, how to run it, results and known issues, plus that phase's scripts.

All training code in this folder was written during the event. The pre-event code elsewhere in the repo (Oct 13–14) is reference only.

> Results are only written down after something has actually been run. "Not run yet" means exactly that.

## Pipeline

```
01-data  ──►  02-labeling  ──►  03-features  ──►  04-head  ──►  05-eval
(download,     (Glimmer weak      (frozen V-JEPA     (concept-     (per-factor
 verify,        labels F1–F10,     2.1 patch          bottleneck    metrics,
 manifest)      clip level)        features 24×24)    F1–F11→risk)  calibration)
                                                          │
                                                          ▼
                                            06-localization ──► 07-export
                                            (patch activations    (artifacts for the
                                             → circle overlay)     API / mobile app)
```

| Phase | Folder | What it does | Status |
|---|---|---|---|
| 00 | [`00-setup`](00-setup/) | Environment check (Python, CUDA, Ollama, disk) | Done Oct 30: PC passes `check_env.py` |
| 01 | [`01-data`](01-data/) | Download + verify datasets, write `data/manifest.jsonl` | Oct 30: Voxel51 complete (691/691 clips verified, 9.3 GiB in 27.6 min). Oct 31: Zenodo complete (39/39 parts md5 OK, 18,953 images, 18,690 with labels; Train/Val/Test 6,507 / 2,085 / 3,591; only 36 hand-free) |
| 02 | [`02-labeling`](02-labeling/) | Muse Glimmer 30B weak labels for F1-F10 | Oct 31: v1/v2 x reasoning none/low eval (368 calls), v2/none chosen. Backends: Ollama 5.4, SGLang 8.6, vLLM INT4 21.6 calls/min. **Full run done 13:38-16:14 on vLLM: 658/658 clips, 2,632 frames, 0 errors.** F7/F8/F9 from dataset hints |
| 03 | [`03-features`](03-features/) | V-JEPA 2.1 ViT-L feature extraction (384x672, 16-frame windows) | Done Oct 31: 658 clips + 12,183 images, 0 errors, 13.7 min |
| 04 | [`04-head`](04-head/) | Concept-bottleneck head (MIL for clip-level factors, anomaly for F11) | First version done Oct 31 (h3_wide); F11 not yet |
| 05 | [`05-eval`](05-eval/) | Evaluation and the Glimmer label spot-check | Not started |
| 06 | [`06-localization`](06-localization/) | Patch activations to a circle around the flagged area | Not started |
| 07 | [`07-export`](07-export/) | Export weights/config for the backend and the mobile app | Not started |

## Quick start (Windows PowerShell, from the repo root)

```powershell
pip install -r training/requirements.txt
python training/00-setup/check_env.py
python training/01-data/download.py --dry-run --sources voxel51 zenodo
python training/01-data/download.py --sources voxel51 --limit 5      # smoke test (--workers 16 by default)
python training/02-labeling/label_glimmer.py --limit 3                # Glimmer dry run (Ollama running)
python -m pytest training/tests -q                                    # offline tests (62), no Ollama needed
# Batched labeling on vLLM in WSL (see 02-labeling/scripts/README.md for the server):
python -X utf8 training/02-labeling/label_glimmer.py --backend vllm --sources voxel51_safe_unsafe --prompt training/02-labeling/prompts/factors_v2.md --reasoning-effort none --concurrency 32 --out training/02-labeling/outputs/weak_labels_v2_vllm.jsonl
python training/02-labeling/summarize_labels.py
```

## Layout

```
training/
  README.md  requirements.txt  .gitignore
  configs/      factors.yaml (F1–F11 definitions, dataset hints), datasets.yaml
  00-setup/ … 07-export/   README.md each, plus that phase's scripts
  02-labeling/scripts/  copies of the vLLM/SGLang server + full-run launcher scripts (live copies in F:\models\wsl)
  tests/        offline unit tests
  data/         (gitignored) raw datasets + manifest.jsonl
  cache/        (gitignored) Glimmer frame answers, keyframe previews
  weights/      (gitignored) checkpoints
```

## Hazard factors

These are defined in [`configs/factors.yaml`](configs/factors.yaml), which is the single source of truth for the prompt, the head output order and evaluation.

| ID | Factor | Labels from |
|---|---|---|
| F1 | Hand in a machine's danger zone | Glimmer, plus the Zenodo table-saw hand boxes |
| F2 | No head protection | Glimmer |
| F3 | No eye or face protection | Glimmer |
| F4 | Wrong glove state near machinery | Glimmer |
| F5 | No hi-vis vest | Glimmer |
| F6 | Falling or slipping | Glimmer (motion; still frames are weak evidence) |
| F7 | Unstable or falling load | Glimmer, plus Voxel51 class hints |
| F8 | Person in a vehicle path or off the walkway | Glimmer, plus Voxel51 class hints |
| F9 | Open guard or panel | Glimmer, plus Voxel51 class hints |
| F10 | Fire, smoke or sparks | Glimmer |
| F11 | Abnormal machine behaviour | V-JEPA 2.1 prediction error (not labeled) |

## Data and licences

Only permissive licences are used (CC BY 4.0 / CC0 / MIT / Apache-2.0). SH17, iSafetyBench and IPAD are **not** used because their licences are non-commercial or missing. Every manifest line carries its source, licence and attribution. The details are in [`configs/datasets.yaml`](configs/datasets.yaml) and [`01-data/README.md`](01-data/README.md).

Disclaimer: Sightline is a decision-support demo, not a certified safety system.
