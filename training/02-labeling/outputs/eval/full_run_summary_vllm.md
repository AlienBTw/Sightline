# Stage 1 full labelling run: Muse Glimmer 30B 4-bit (`RedHatAI/Muse-Glimmer-30B-INT4`) on vLLM, served as `muse-glimmer-30b-int4-vllm`, tensor parallel, v2/none, c=32 (2026-10-31)

## Run

- Rows: 658, unique clips: 658, valid rows: 658, malformed: 0, duplicate clip ids: 0
- Frames (keyframes labeled): 2632
- Splits: {'test': 117, 'train': 541}; sources: {'voxel51_safe_unsafe': 658}
- Model: {'muse-glimmer-30b-int4-vllm': 658}; prompt: {'v2-7238ea9f': 658}; reasoning: {'none': 658}
- Clips standing for a byte-identical duplicate group (hints merged): 32
- All F1-F10 clip labels: present 0.28, uncertain 0.24

## Per-factor labels

Clip level = MIL aggregation (present if any frame says present with conf >= 0.5; absent only if every frame says absent; otherwise uncertain). Frame level = the raw per-keyframe answers.

| Factor | Clip present | Clip absent | Clip uncertain | Frame present | Frame absent | Frame uncertain |
|---|---|---|---|---|---|---|
| F1 | 29 (4%) | 620 (94%) | 9 (1%) | 44 (2%) | 2574 (98%) | 14 (1%) |
| F2 | 528 (80%) | 3 (0%) | 127 (19%) | 1772 (67%) | 155 (6%) | 705 (27%) |
| F3 | 7 (1%) | 2 (0%) | 649 (99%) | 8 (0%) | 102 (4%) | 2522 (96%) |
| F4 | 0 (0%) | 4 (1%) | 654 (99%) | 0 (0%) | 184 (7%) | 2448 (93%) |
| F5 | 513 (78%) | 25 (4%) | 120 (18%) | 1648 (63%) | 311 (12%) | 673 (26%) |
| F6 | 9 (1%) | 649 (99%) | 0 (0%) | 10 (0%) | 2622 (100%) | 0 (0%) |
| F7 | 26 (4%) | 632 (96%) | 0 (0%) | 43 (2%) | 2589 (98%) | 0 (0%) |
| F8 | 629 (96%) | 28 (4%) | 1 (0%) | 2154 (82%) | 448 (17%) | 30 (1%) |
| F9 | 120 (18%) | 520 (79%) | 18 (3%) | 252 (10%) | 2348 (89%) | 32 (1%) |
| F10 | 9 (1%) | 649 (99%) | 0 (0%) | 10 (0%) | 2622 (100%) | 0 (0%) |

## Glimmer vs Voxel51 class hints (F7 / F8 / F9)

Positive = Glimmer clip label present. Uncertain counts as a miss for recall and is not a false positive. Agree-when-decided ignores uncertain clips.

| Factor | Hint present (n) | -> present / absent / unc | Hint absent (n) | -> present / absent / unc | Precision | Recall | Specificity | Uncertain | Agree when decided |
|---|---|---|---|---|---|---|---|---|---|
| F7 | 56 | 23 / 33 / 0 | 30 | 3 / 27 / 0 | 0.89 | 0.41 | 0.90 | 0.00 | 0.58 |
| F8 | 210 | 194 / 15 / 1 | 72 | 71 / 1 / 0 | 0.73 | 0.92 | 0.01 | 0.00 | 0.69 |
| F9 | 142 | 49 / 82 / 11 | 32 | 0 / 31 / 1 | 1.00 | 0.34 | 0.97 | 0.07 | 0.49 |

## Clip-level present count by dataset class

| Class | Clips | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Safe Walkway Violation | 210 | 2 | 165 | 0 | 0 | 166 | 3 | 0 | 194 | 9 | 6 |
| Opened Panel Cover | 141 | 21 | 138 | 4 | 0 | 126 | 4 | 0 | 137 | 49 | 0 |
| Unauthorized Intervention | 81 | 3 | 72 | 1 | 0 | 72 | 0 | 0 | 78 | 28 | 1 |
| Safe Walkway | 72 | 1 | 66 | 2 | 0 | 67 | 0 | 0 | 71 | 13 | 2 |
| Carrying Overload with Forklift | 56 | 0 | 20 | 0 | 0 | 23 | 0 | 23 | 53 | 0 | 0 |
| Authorized Intervention | 37 | 2 | 34 | 0 | 0 | 27 | 1 | 0 | 36 | 21 | 0 |
| Closed Panel Cover | 31 | 0 | 29 | 0 | 0 | 28 | 1 | 0 | 31 | 0 | 0 |
| Safe Carrying | 30 | 0 | 4 | 0 | 0 | 4 | 0 | 3 | 29 | 0 | 0 |

## Malformed rows

None: every row has F1-F10 with a valid label, a confidence in [0, 1] and votes that add up to its keyframes.

## Notes (hand-written, 2026-10-31 16:30 Tunis; rerunning summarize_labels.py overwrites this file)

- Run: 13:38:21-16:14:17 Tunis (2 h 36 min), EXIT 0. 2,536 new model calls + 96 frames from cache (23 eval clips + smoke clip) = 2,632 frames. 16.3 calls/min at concurrency 32 (about 20/min in the first hour). Seconds per frame incl. queueing: mean 114.6, median 116.2, max 176.6. 0 errors, no retries logged, `weak_labels_v2_vllm.errors.jsonl` not created, stderr empty.
- **F8 from Glimmer is useless on Voxel51:** present on 629/658 clips (96%), including 71/72 Safe Walkway clips (specificity 0.01). Use the class hints for F8.
- **F7 and F9 are precise but miss most positives:** F7 P 0.89 / R 0.41 (23/56 forklift-overload clips), F9 P 1.00 / R 0.34 (49/142 opened-panel clips). Keep the class hints for F7/F9 on Voxel51; Glimmer F7/F9 is only a weak extra signal.
- **F2 / F5 (no helmet / no hi-vis) say present on ~80% of clips** across every class (except Safe Carrying, 4/30). Either the scenes really lack PPE or it is a prompt bias; check in the 05-eval spot check before training on them.
- **F3 / F4 carry no signal:** 99% uncertain at clip level (96% / 93% of frames).
- **Very few positives:** F6 9, F10 9, F1 29 (21 of them on Opened Panel Cover), F3 7, F4 0. These factors need other data (PPE / fire sets, Zenodo hands, recorded workshop clips).
