# 04 · Head (concept-bottleneck training)

**Goal:** a small head on frozen V-JEPA 2.1 features that predicts the 11 hazard factors first and the risk score from those factors (a concept bottleneck), so every alert is explained by named factors.

## Inputs
- Cached features (03), weak labels `02-labeling/outputs/weak_labels_v2_vllm.jsonl` (full run; use them for F1-F6/F10), dataset class hints (F7/F8/F9 on Voxel51), and the Zenodo hand boxes (F1)

## Outputs (planned)
- `training/weights/head_<run>.pt` (gitignored), plus the run config and metrics JSON (committed)

## Method (planned)
- Per-patch factor logits, with multiple-instance pooling (e.g. top-k / LSE over patches and windows) for factors that only have clip-level labels
- F11 from the V-JEPA prediction error, calibrated on normal footage
- Risk = a learned combination of factor probabilities. The prior is noisy-OR (`configs/factors.yaml`).
- Uncertain labels are masked out of the loss.

## How to run
Not written yet.

## Results
Not run yet.

## Known issues
- The duplicate Voxel51 clips must be grouped into the same split (see 01-data).
