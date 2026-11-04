# 05 · Eval

**Goal:** honest numbers for each factor and for the risk score, on a de-duplicated test split, plus a check of how good the Glimmer weak labels are.

## Planned
- A 10% human spot-check of the Glimmer labels, measuring per-factor agreement. Factors that are too noisy get dropped rather than faked.
- Glimmer vs. dataset-class hints (F7/F8/F9) agreement
- Per-factor AUROC/F1 and risk calibration (reliability plot) on Voxel51 test, after removing train/test duplicates
- F1 localization against the Zenodo hand boxes (from 06)

## Results
- F1 localization against the Zenodo hand boxes: done, see [`06-localization`](../06-localization/README.md).
- Per-factor test and cross-validation metrics are in [`04-head`](../04-head/README.md).
- The human spot-check of Glimmer labels and an out-of-distribution test set are future work, since no new data work happens during the event.
