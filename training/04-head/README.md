# 04 â€“ Head (concept-bottleneck training)

**Goal:** a small head on frozen V-JEPA 2.1 features that predicts the hazard factors (F1-F10) as named concepts, each with its own spatial map, so every alert is explained by factors and can be circled in the frame. The risk score (noisy-OR prior in `configs/factors.yaml`) and F11 (anomaly from V-JEPA prediction error) come later.

## Inputs
- Features from 03 (`<SIGHTLINE_WORK_DIR>/features/vjepa2_1_vitl_384`), pooled 2x2 into a 12 x 21 grid per window (252 tokens x 1024) by `train_head.py cache` -> one fp16 memmap in `<SIGHTLINE_WORK_DIR>/cache/vjepa2_1_vitl_384_p2` (9,587 items, 11,113 windows, ~5.7 GB; built in 9.5 min). Work dir default is `<repo>/work`.
- Labels:
  | Factor | Voxel51 source | Zenodo |
  |---|---|---|
  | F1 hand in danger zone | stage 1 Glimmer (vLLM INT4) | hand box present (+ patch-level box target) |
  | F2 no head protection, F5 no hi-vis | stage 1 Glimmer (~80% present, possible bias). Stage 2 replaces these on PPE video with SAM 3.1 boxes + Muse Spark (see [`08-video-ppe`](../08-video-ppe/README.md)) | - |
  | F3 eye/face, F4 gloves | **masked** (~99% uncertain) | - |
  | F6 fall/slip, F10 fire/sparks | stage 1 Glimmer (9 positives each) | - |
  | F7 / F8 / F9 | dataset class hint where the class defines it, else masked | - |
  Uncertain labels are always masked (ignored in the loss).
- Splits: Voxel51 = the 658 de-duplicated clips (duplicate files merged in 02, so no clip is in both train and test). Official test split = 117 clips; val = 15% of train by a stable hash. Zenodo = Train (every 2nd frame, consecutive frames are near-identical) / Validation / Test as published. Split counts can be reprinted with `python training/04-head/scripts/split_stats.py`.

## Method
```
tokens (all windows of a clip, N x 1024) -> LayerNorm -> Linear(1024,H) -> GELU -> Dropout
per factor f:  attention a_f(p) = w_f . h_p   (softmax over every patch of every window = MIL pooling)
               evidence  s_f(p) = u_f . h_p
               logit z_f = sum_p softmax(a_f)(p) * s_f(p) + b_f
```
- The per-factor softmax weights and `sigmoid(s_f)` are the spatial map for circling (06).
- Loss: masked BCE averaged per factor (so sparse factors are not drowned), `pos_weight = clamp(neg/pos, 1, 20)`, plus an auxiliary patch-level BCE on `s_F1` against the Zenodo hand boxes (weight 0.5, pos_weight 5).
- Sampling: Zenodo ~30% of each batch; window dropout 0.25 and token dropout 0.1; AdamW + OneCycle, 30 epochs x 50 steps x 32; best epoch by validation mean AP; thresholds = best F1 on validation.
- Checkpoints: `04-head/checkpoints/<run>.pt` (gitignored). Metrics + config + per-clip test predictions: `04-head/results/<run>.json` (committed). Table: `results/SUMMARY.md`.

## How to run (WSL, `~/vjepa-venv`)
```bash
python train_head.py cache                                   # once, ~10 min
python train_head.py train --name h3_wide --hidden 512 --dropout 0.2 --lr 5e-4
python train_head.py train --cv 5 --fold 0 --name h3_wide_cv5_f0 ...   # k-fold over all 658 clips
python summarize_head.py                                     # -> results/SUMMARY.md
```
Detached launchers + status: `scripts/run_head_sweep.sh`, `scripts/run_head_cv*.sh`, `scripts/check_head.ps1`. One run takes 1.5-5 min on one 3090.

## Results (2026-10-31)
Voxel51 official test (117 unique clips), AUROC / AP (positives/labeled):

| run | F1 | F5 | F6 | F7 | F8 | F9 | F10 | Zenodo F1 patch AUROC / AP |
|---|---|---|---|---|---|---|---|---|
| h1_base (H=256), seeds 0/1/2 | 0.93-0.94 / 0.11-0.12 (1/115) | 0.73-0.82 / 0.98 | 0.86-0.92 / 0.06-0.10 (1/117) | 1.00 / 1.00 (8/16) | 0.85-0.88 / 0.89-0.93 (32/54) | 0.76-0.92 / 0.82-0.95 (13/25) | 0.93-0.97 / 0.11-0.20 (1/117) | 0.99 / 0.94 |
| h2_voxel_only (no Zenodo) | 0.98 / 0.33 | 0.95 / 1.00 | 0.97 / 0.25 | 1.00 / 1.00 | 0.90 / 0.94 | 0.94 / 0.94 | 1.00 / 1.00 | - |
| **h3_wide (H=512, drop 0.2, lr 5e-4)** | 0.95 / 0.14 | 0.91 / 0.99 | 0.99 / 0.50 | 1.00 / 1.00 | 0.90 / 0.93 | 0.96 / 0.97 | 0.97 / 0.20 | **0.99 / 0.94** |
| h4_no_patch_loss | 0.97 / 0.25 | 0.87 / 0.99 | 0.63 / 0.02 | 1.00 / 1.00 | 0.89 / 0.94 | 0.84 / 0.88 | 0.97 / 0.20 | 0.56 / 0.09 |

h3_wide at the validation-chosen thresholds on the test split: F8 F1 0.86 (P 0.85 / R 0.88), F9 0.72 (P 0.57 / R 1.00), F7 1.00, F5 0.94.

**5-fold out-of-fold over all 658 unique clips (h1_base)** â€“ the more reliable numbers, since the official test has only 1 positive for F1/F6/F10:

| F1 | F2 | F5 | F6 | F7 | F8 | F9 | F10 |
|---|---|---|---|---|---|---|---|
| 0.96 / 0.49 (29/649) | 0.65 / 1.00 (528/531) | 0.75 / 0.98 (513/538) | 0.97 / 0.57 (9/658) | 1.00 / 1.00 (56/86) | 0.87 / 0.96 (210/282) | 0.86 / 0.96 (142/174) | 0.79 / 0.06 (9/658) |

5-fold out-of-fold for **h3_wide**: F1 0.96 / 0.59, F2 0.53 / 0.99, F5 0.74 / 0.97, F6 0.89 / 0.25, F7 1.00 / 1.00, F8 0.83 / 0.93, F9 0.89 / 0.95, F10 0.86 / 0.07. Under CV the two configs are within noise (h3 better on F1/F9/F10, h1 on F6/F8), so h3_wide's lead on the small official test is partly luck.

### What this means
- **F7/F8/F9 (dataset labels) are learned well**: F7 is perfectly separable (overloaded vs safe forklift clips), F8 ~0.87-0.90 AUROC, F9 ~0.86-0.96.
- **F1 localization works on the table-saw domain**: the F1 patch map matches hand boxes at 0.99 patch AUROC / 0.94 AP on 3,591 held-out images. Without the patch loss (h4) the map is near chance (0.56), so the box supervision is what makes circling possible.
- Adding Zenodo costs a little on Voxel51 F1-F10 test numbers (h2 vs h1), but h2 has no localization for F1. h3_wide recovers most of the gap on the official test -> **h3_wide is the current pick**, with h1_base as an equal alternative per CV.
- **F2 cannot be evaluated**: every labeled test clip is "present" (528/531 overall). Combined with the ~80% present rate this looks like stage 1 Glimmer bias or a dataset without helmets; needs the spot-check / a PPE dataset. Nov 4 follow-ups: SHWD bare-head boxes (patch AUROC 0.984-0.990 on the held-out SHWD test split, no transfer to video) and heads retrained on Muse-labelled video (they learned the video source). Stage 2 then relabelled those PPE clips with SAM 3.1 boxes and Muse Spark 1.3 Contributor (see [`08-video-ppe`](../08-video-ppe/README.md) and [`results/PPE_SHORTCUT.md`](results/PPE_SHORTCUT.md)). h3_wide stays live; F2/F5 are decided by the Muse Spark verify gates in `sightline_pipeline`.
- F6/F10 (9 positives each) and Voxel51 F1 (29) are too sparse for trustworthy numbers; F3/F4 are not trained.
- Zenodo F1 image-level metrics are meaningless (3,590 of 3,591 test images contain hands), so F1 is reported per source; mixed-source F1 AUROC would just measure domain.

## Known issues
- Weak labels are clip-level; single-positive factors in the test split make test AP noise-dominated. Use the CV table.
- Zenodo is still images vs Voxel51 video; domain shift for F1 on real workshop video is untested (needs own recorded clips).
