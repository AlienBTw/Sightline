# Sightline training pipeline

Everything to do with training the Sightline models lives here, with one folder per phase. Each phase folder has a `README.md` with the goal, inputs, outputs, how to run it, results and known issues, plus that phase's scripts.

All training code in this folder was written during the event. The pre-event code elsewhere in the repo (Oct 13–14) is reference only.

> Results are only written down after something has actually been run. "Not run yet" means exactly that.

Labelling is two stages. **Stage 1** ([`02-labeling`](02-labeling/README.md)): local Muse Glimmer 30B 4-bit (`RedHatAI/Muse-Glimmer-30B-INT4`) served on vLLM as `muse-glimmer-30b-int4-vllm` (tensor parallel) gives weak F1–F10 labels for the initial head. **Stage 2** ([`08-video-ppe`](08-video-ppe/README.md)): SAM 3.1 boxes plus Muse Spark 1.3 Contributor (Meta Model API) relabel the PPE video clips (helmet/vest) to fix the source shortcut. Ollama and SGLang were earlier attempts, not the full run.

## Pipeline

```
01-data  ──►  02-labeling  ──►  03-features  ──►  04-head  ──►  05-eval
(download,     (stage 1: vLLM    (frozen V-JEPA     (concept-     (per-factor
 verify,        Glimmer F1–F10,    2.1 patch          bottleneck    metrics,
 manifest)      clip level)        features 24×24)    F1–F11→risk)  calibration)
                                                          │
                                                          ▼
                                            06-localization ──► 07-export
                                            (patch activations    (artifacts for the
                                             → circle overlay)     API / mobile app)
```

| Phase | Folder | What it does | Status |
|---|---|---|---|
| 00 | [`00-setup`](00-setup/) | Environment check (Python, CUDA, disk) | Done Oct 30: PC passes `check_env.py`. The script also prints Ollama models; that was an earlier labelling attempt |
| 01 | [`01-data`](01-data/) | Download + verify datasets, write `data/manifest.jsonl` | Oct 30: Voxel51 complete (691/691 clips verified, 9.3 GiB in 27.6 min). Oct 31: Zenodo complete (39/39 parts md5 OK, 18,953 images, 18,690 with labels; Train/Val/Test 6,507 / 2,085 / 3,591; only 36 hand-free) |
| 02 | [`02-labeling`](02-labeling/) | Stage 1: weak F1–F10 labels from Muse Glimmer 30B 4-bit (`RedHatAI/Muse-Glimmer-30B-INT4`) on vLLM, served as `muse-glimmer-30b-int4-vllm` (tensor parallel) | **Full run 13:38-16:14: 658/658 clips, 2,632 frames, 0 errors.** Prompt v2/none. F7/F8/F9 from dataset hints. Ollama (5.4 calls/min) and SGLang (8.6) were earlier attempts; the 23-clip prompt choice used 368 calls |
| 03 | [`03-features`](03-features/) | V-JEPA 2.1 ViT-L feature extraction (384x672, 16-frame windows) | Done Oct 31: 658 clips + 12,183 images, 0 errors, 13.7 min |
| 04 | [`04-head`](04-head/) | Concept-bottleneck head (MIL for clip-level factors, anomaly for F11) | First version done Oct 31 (h3_wide); F11 not yet. Nov 4: SHWD and video-PPE retrains rejected, h3_wide kept (see 08) |
| 05 | [`05-eval`](05-eval/) | Evaluation and the Glimmer label spot-check | Oct 31: circle quality on Zenodo test measured (see 06); label spot-check dropped (no more data work) |
| 06 | [`06-localization`](06-localization/) | Patch activations to a circle around the flagged area | Done Oct 31 in the live pipeline [`sightline_pipeline/`](../sightline_pipeline/): F1 circle centre inside a hand box on 93.7% of 3,590 Zenodo test images |
| 07 | [`07-export`](07-export/) | Export weights/config for the backend and the mobile app | Not started |
| 08 | [`08-video-ppe`](08-video-ppe/) | Stage 2: SAM 3.1 boxes + Muse Spark 1.3 Contributor relabel the PPE video clips (helmet/vest) to fix the source shortcut | Nov 4: 383 clips (386 Muse calls). Retrains still learned the video source, so h3_wide stays; live F2/F5 are decided by the Muse Spark verify gates |

Live inference (ingest, sliding windows, events, circles, SAM 3.1 / Muse Spark escalation, clips) is in the top-level package [`sightline_pipeline/`](../sightline_pipeline/README.md), which reuses `03-features` and `04-head` code.

## Quick start (Windows PowerShell, from the repo root)

```powershell
pip install -r training/requirements.txt
python training/00-setup/check_env.py
python training/01-data/download.py --dry-run --sources voxel51 zenodo
python training/01-data/download.py --sources voxel51 --limit 5      # smoke test (--workers 16 by default)
python training/02-labeling/label_glimmer.py --limit 3                # dry run; the script default talks to Ollama (an earlier attempt)
python -m pytest training/tests -q                                    # offline tests (62)
# Stage 1 full run: vLLM serving muse-glimmer-30b-int4-vllm (see 02-labeling/scripts/README.md):
python -X utf8 training/02-labeling/label_glimmer.py --backend vllm --sources voxel51_safe_unsafe --prompt training/02-labeling/prompts/factors_v2.md --reasoning-effort none --concurrency 32 --out training/02-labeling/outputs/weak_labels_v2_vllm.jsonl
python training/02-labeling/summarize_labels.py
```

## Layout

```
training/
  README.md  requirements.txt  .gitignore
  configs/      factors.yaml (F1–F11 definitions, dataset hints), datasets.yaml
  00-setup/ … 07-export/   README.md each, plus that phase's scripts
  02-labeling/scripts/  vLLM server and full-run launcher (SGLang scripts are an earlier attempt). Edit the local paths inside the scripts before running them.
  tests/        offline unit tests
  data/         (gitignored) raw datasets + manifest.jsonl
  cache/        (gitignored) Glimmer frame answers, keyframe previews
  weights/      (gitignored) checkpoints
```

## Hazard factors

These are defined in [`configs/factors.yaml`](configs/factors.yaml), which is the single source of truth for the prompt, the head output order and evaluation.

| ID | Factor | Labels from |
|---|---|---|
| F1 | Hand in a machine's danger zone | Stage 1 Glimmer, plus the Zenodo table-saw hand boxes |
| F2 | No head protection | Stage 1 Glimmer weak label; stage 2 SAM 3.1 boxes + Muse Spark on PPE video |
| F3 | No eye or face protection | Stage 1 Glimmer |
| F4 | Wrong glove state near machinery | Stage 1 Glimmer |
| F5 | No hi-vis vest | Stage 1 Glimmer weak label; stage 2 SAM 3.1 boxes + Muse Spark on PPE video |
| F6 | Falling or slipping | Stage 1 Glimmer (motion; still frames are weak evidence) |
| F7 | Unstable or falling load | Stage 1 Glimmer, plus Voxel51 class hints |
| F8 | Person in a vehicle path or off the walkway | Stage 1 Glimmer, plus Voxel51 class hints |
| F9 | Open guard or panel | Stage 1 Glimmer, plus Voxel51 class hints |
| F10 | Fire, smoke or sparks | Stage 1 Glimmer |
| F11 | Abnormal machine behaviour | V-JEPA 2.1 prediction error (not labeled) |

## Data and licences

Only permissive licences are used (CC BY 4.0 / CC0 / MIT / Apache-2.0). SH17, iSafetyBench and IPAD are **not** used because their licences are non-commercial or missing. Every manifest line carries its source, licence and attribution. The details are in [`configs/datasets.yaml`](configs/datasets.yaml) and [`01-data/README.md`](01-data/README.md).

Disclaimer: Sightline is a decision-support demo, not a certified safety system.
