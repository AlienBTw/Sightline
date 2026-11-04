# 08 - Video PPE labels and shortcut check (F2, F5)

**Goal:** give F2 (no head protection) and F5 (no hi-vis vest) labels on real video, retrain those two outputs of the concept-bottleneck head, and check on held-out clips whether the head learned the helmet and the vest or only the video source.

## Inputs and licences

| Footage | Licence | Clips |
|---|---|---|
| Pexels construction / workshop clips, labelled per clip | Pexels License | 87 (train 50, val 19, test 5, demo_eval 13) |
| Voxel51 Safe_and_Unsafe_Behaviours (the phase 03/04 clips) | CC BY 4.0 | 296 labelled (train 179, test 117) |

The Pexels split is grouped by `pexels_id // 10000`, and the demo clips are kept out of training (`demo_eval`); see [`tools/labeling/README.md`](../../tools/labeling/README.md).

## Labels

- **Muse Spark 1.3 Contributor** (`muse-spark-1.3-contributor`, Meta Model API) looks at 3 frames per clip and answers `present` / `absent` / `uncertain` for F2 and F5. That took 386 calls for 383 clips (4 answers could not be parsed; 3 were run again).
- The planned SAM 3.1 boxes are **not** in these labels. The open `facebook/sam3.1` weights are gated on Hugging Face and the download was refused, so the labeller ran Muse-only and the window labels carry the clip verdict. The live pipeline uses SAM 3.1 through the Meta Model API instead (circles only).
- Pexels F2: 49 present, 32 absent, 6 uncertain. Pexels F5: 57 present, 29 absent, 1 uncertain. Voxel51 F2: 279 present, 17 uncertain. Voxel51 F5: 285 present, 11 uncertain. Voxel51 has no `absent` clip for either factor, so the two sources differ in label balance as well as in look.
- Merged labels: `training/data/ppe_video_labels.jsonl` (745 window rows; not committed with the data).

## How to run

From the repo root in WSL, with `PY=/home/user/vjepa-venv/bin/python`:

```bash
# clips, manifest, keyframes, labels (see tools/labeling/README.md for the budgets)
$PY tools/labeling/prep_clips.py
$PY tools/labeling/build_ppe_manifest.py
$PY tools/labeling/sample_frames.py --source ppe_video
$PY tools/labeling/muse_verify.py --source ppe_video --model muse-spark-1.3-contributor --max-calls 84
$PY tools/labeling/merge_labels.py

# features, cache, retrain (example: only F2/F5 trainable)
$PY training/03-features/extract_features.py --sources ppe_video --gpu 1
$PY training/04-head/train_head.py cache --sources ppe_video --out ~/sightline_cache/ppe_video_p2
$PY training/04-head/train_head.py train --name ppe_ft_frz --gpu 1 --hidden 512 --dropout 0.2 \
  --init-checkpoint training/04-head/checkpoints/h3_wide.pt \
  --ppe-labels training/data/ppe_video_labels.jsonl --ppe-cache ~/sightline_cache/ppe_video_p2 \
  --ppe-override-voxel --mask-glimmer-f2f5 --ppe-patch-loss 0 \
  --lr 5e-4 --freeze-except F2,F5 --ppe-weight 0.25

# shortcut check of h3_wide and every checkpoints/ppe_*.pt
$PY training/04-head/ppe_shortcut_eval.py --gpu 1
```

`muse_verify.py` calls a paid API; its `--max-calls` default is 0. Test the setup with `--dry-run` first.

## Results

Held-out set: Pexels test + demo_eval (F2 12 positive / 6 negative clips, F5 16 / 2). Domain AUROC: how well the factor score alone separates Voxel51 clips from Pexels clips (0.5 = no source signal). Full table: [`../04-head/results/PPE_SHORTCUT.md`](../04-head/results/PPE_SHORTCUT.md).

| Checkpoint | Training | F2 held-out AUROC | F5 held-out AUROC | F2 domain AUROC | F5 domain AUROC |
|---|---|---|---|---|---|
| **h3_wide** (live) | Glimmer weak labels | 0.403 [0.117, 0.736] | 0.875 | 0.972 | 0.945 |
| ppe_ft_frz | all Muse labels, only F2/F5 trainable | 0.389 | 0.750 | 1.000 | 1.000 |
| ppe_ft_all | all Muse labels, whole head | 0.556 | 0.750 | 1.000 | 1.000 |
| ppe_ft_frz_w05 | as ppe_ft_frz, PPE loss weight 0.5 | 0.194 [0.000, 0.444] | 0.875 | 1.000 | 1.000 |
| ppe_pex_frz | Pexels labels only, only F2/F5 trainable | 0.194 | 0.844 | 0.538 | 1.000 |
| ppe_pex_all | Pexels labels only, whole head | 0.583 | 0.812 | 0.813 | 0.946 |

- F1 is unchanged: test AUROC 0.99989 for h3_wide and the frozen variants, 0.99991 for ppe_ft_all.
- A Pexels-only variant with SHWD bare-head boxes (`ppe_pex_shwd_frz`) did not run: the cache it was pointed at has no SHWD rows.
- With 6 F2 negatives and 2 F5 negatives the confidence intervals are wide; no run is clearly better than h3_wide.

## Conclusion

Every retrain separates the two video sources (domain AUROC up to 1.00) without telling a helmet from a bare head on unseen clips (F2 held-out AUROC 0.19-0.58). **h3_wide stays live.** F2 and F5 alert only after the Muse Spark 2-of-3 verify gate in [`sightline_pipeline`](../../sightline_pipeline/README.md) confirms them. Fixing this needs same-camera, paired helmet / no-helmet and vest / no-vest footage, then a partial V-JEPA 2.1 fine-tune.

## Phase 2 - SAM 3.1 (unsloth) boxes (2026-11-04)

Phase 2 puts local SAM 3.1 boxes (unsloth/sam3.1, SAM License, 19 Nov 2025) on the same 383 clips as window supervision for F2 (no head protection) and F5 (no hi-vis vest), retrains three heads from h3_wide, and checks whether any of them separate helmet from bare head on held-out Pexels clips. Detail and the licence notes are in [`LOG.md`](LOG.md).

Held-out is Pexels test + demo_eval (F2 12 positive / 6 negative, F5 16 / 2), from the Summary rows in [`../04-head/results/PPE_SHORTCUT.md`](../04-head/results/PPE_SHORTCUT.md). Domain AUROC is how well the factor score alone separates Voxel51 from Pexels. F1 patch is Zenodo `F1_patch_vs_boxes` on the training `TEST` line. Voxel51 test F2 clip AUROC is the `PPE_TEST_CLIP` line (117 clips, 116 positive / 1 negative). Demo replay is `data/demo/sightline_demo.mp4` (F2 fraction of steps at or above the run's own threshold). The grinder column is F2 max versus that threshold on `data/demo_extra/pexels_8987012_bare_head_grinder.mp4`.

| Run | held-out F2 AUROC [95% CI] | F5 AUROC [95% CI] | domain AUROC F2/F5 | F1 patch AUROC/AP | Voxel51 test F2 clip AUROC | demo replay F2 frac_above | bare-head grinder F2 max vs threshold |
|---|---|---|---|---|---|---|---|
| h3_wide (live) | 0.403 [0.117, 0.736] | 0.875 [0.688, 1.000] | 0.972 / 0.945 | 0.992 / 0.939 | n/a | 0.807 | 0.755 vs 0.914 |
| A ppe_sam_frz | 0.625 [0.333, 0.911] | 0.250 [0.000, 0.600] | 0.914 / 0.862 | 0.992 / 0.939 | 0.940 | 1.000 | 0.811 vs 0.477 |
| B ppe_sam_shwd_frz | 0.486 [0.219, 0.750] | 0.375 [0.062, 0.706] | 0.815 / 0.701 | 0.992 / 0.939 | 0.966 | 0.789 | 0.608 vs 0.425 |
| C ppe_sam_shwd_all | 0.792 [0.549, 1.000] | 0.656 [0.353, 0.941] | 0.671 / 0.843 | 0.991 / 0.940 | 0.991 | 0.877 | 0.983 vs 0.675 |

h3_wide has no `PPE_TEST_CLIP` line. Its own test F2 is 95/95 present, so that AUROC is n/a. Thresholds behind `frac_above` and the grinder comparison: h3_wide 0.914, A 0.477, B 0.425, C 0.675.

**Gate:** F2 95% CI lower bound > 0.5, and F2 domain AUROC drops versus h3_wide, and F1 patch does not regress.

**Verdict:** only C passes the F2 gate (0.792 [0.549, 1.000], domain 0.671 vs 0.972, F1 patch 0.991/0.940 vs 0.992/0.939). F5 regresses from 0.875 to 0.656 (CI [0.353, 0.941]). C is not promoted. h3_wide stays live by Alien's decision. A and B miss the F2 interval (lower bounds 0.333 and 0.219).
