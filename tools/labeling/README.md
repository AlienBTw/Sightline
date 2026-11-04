# PPE video labelling (F2 no head protection, F5 no hi-vis vest)

Stage 2 of labelling. Stage 1 weak F1–F10 labels came from local Muse Glimmer 30B 4-bit (`RedHatAI/Muse-Glimmer-30B-INT4`) on vLLM (`training/02-labeling`). Here, SAM 3.1 boxes and Muse Spark 1.3 Contributor (Meta Model API) relabel the PPE video clips (helmet and vest) to fix the source shortcut.

Real-video window labels for the concept-bottleneck head. V-JEPA features stay
frozen. This folder only builds clips, manifests, keyframes, and labels. It
does not train the head and it does not call a paid API unless you raise the
cap yourself.

Hazard present is `1` (no helmet, or no hi-vis vest). Absent is `0`. Unknown
is JSON `null`.

## Order

From the repo root in WSL:

```bash
# from the repo root, in WSL
PY=/home/user/vjepa-venv/bin/python

$PY tools/labeling/prep_clips.py
$PY tools/labeling/build_ppe_manifest.py
$PY tools/labeling/sample_frames.py --source ppe_video
$PY tools/labeling/sam_label.py --backend api --dry-run --max-api-calls 0
/home/user/sam3-venv/bin/python tools/labeling/sam_label.py --backend local --gpu 1
$PY tools/labeling/muse_verify.py --dry-run --max-calls 2
$PY tools/labeling/merge_labels.py
```

Feature extraction (not part of labelling; uses GPU):

```bash
$PY training/03-features/extract_features.py --sources ppe_video --gpu 1
```

`prep_clips.py` is safe to re-run while `raw_<id>.mp4` files are still
downloading. A finished clip is skipped until the source size or mtime changes.
`--jobs` sets the process count (default min(4, cpu/2)).

Voxel51 keyframes, same windowing as phase 03:

```bash
$PY tools/labeling/sample_frames.py --source voxel51_safe_unsafe
```

## Inputs and licences

| Footage | Licence | What we record |
|---|---|---|
| Pexels clips (raw downloads outside the repo, plus any extra vest-eval sources you pass in) | Pexels License (free use, no attribution required) | `https://www.pexels.com/video/<id>/` in `attribution`, download URL in `url` when `pexels_urls.txt` has it |
| Voxel51 Safe_and_Unsafe_Behaviours, the 658 sha256-deduped clips phase 03/04 already use | CC BY 4.0 | existing manifest row; official train/test split kept |
| `data/demo_extra/pexels_8987012_bare_head_grinder.mp4` | Pexels License | replay only, in `ppe_demo_eval.jsonl` |
| `data/demo/sightline_demo.mp4` | CC BY 4.0 (Voxel51 test clips and Zenodo test frames; see `data/demo/ATTRIBUTION.json`) | replay only |

SAM 3.1 open weights stay outside the repo under the SAM License. They are not copied into the repo.

## Where files go

Repo (small manifests):

- `training/data/ppe_video_manifest.jsonl` — one row per transcoded Pexels clip
- `training/data/ppe_demo_eval.jsonl` — demo_eval Pexels clips plus the two replay files
- `training/data/ppe_video_labels.jsonl` — merged F2/F5 labels, Pexels and Voxel51

Outside the repo (the scripts' default output directory):

- `clips/pexels_<id>.mp4` — longest side 1280, about 25 fps, at most 30 s, H.264
- `frames/<safe id>/w<k>_<j>.jpg` and `frames/index.jsonl`
- `sam_cache/` — one JSON per keyframe, safe to resume
- `sam_windows.jsonl`
- `sam_api_calls.jsonl` — append-only; this script never touches `data/live/escalation_budget.json`
- `muse_budget.json` — persistent Muse counter
- `muse_calls.jsonl` and `muse_verdicts.jsonl`

## Split rule

`group = pexels_id // 10000`. The same shoot or creator series shares a group, and a group is never broken across splits.

These ids, and every other id in their group, are `demo_eval` and are never train, val, or test: 8987012, 8987022, 8987023, 4292900, 5100044, 5100048, 7457419, 10817415. Their groups are 898, 429, 510, 745, and 1081.

Every other group is hashed with sha256 of `ppe-video-group:0:<group>` into train (`< 0.70`), val (`< 0.85`), or test. If test would have fewer than 3 clips, whole groups with the highest hash are moved from train/val into test until it has 3. Seed is 0.

`demo_eval` rows stay in the feature manifest so they can be extracted, and they are listed again in `ppe_demo_eval.jsonl` with the two replay files. Do not train on `demo_eval`.

## Label rules

Keyframes are the first, middle, and last frame of each V-JEPA window (`window_starts` / `window_indices` in `training/03-features/extract_features.py`: 16 frames, step 4, at most 8 windows).

SAM prompts: `head`, `hard hat`, `helmet`, `safety vest`, `high-visibility vest`, `person`. Hard hat and helmet are one hat set. The two vest phrases are one vest set. A person box shorter than 40 px or shorter than 6% of the frame is ignored. If that leaves no person, the keyframe is uncertain.

- Head: a detected `head` box on that person is bare unless a hard-hat or helmet box covers at least 0.3 of the head (intersection over the head's area). If that person has no head box, the fallback is the top 22% of the person box, width shrunk to the central 60%, and a hat counts when at least 0.3 of the hat's own area lies in that region.
- Torso region: 20% to 60% of the person height, full width. A vest counts when it covers at least 0.25 of that region.

`F2_pos` is bare heads (or the fallback head region). `F2_neg` is helmets and helmeted heads. `F5_pos` is torsos without a vest. `F5_neg` is vests. Keyframe cache boxes are normalised to [0, 1] and keep the score. Window boxes stay xyxy with four numbers so the head trainer can read them.

Window vote, separately for F2 and F5: `1` if a confident person is missing that PPE on at least 2 of the 3 keyframes, `0` if every confident person has it on at least 2 keyframes and none of the keyframes show it missing, otherwise `null`. No confident person in the window is `null`.

Window boxes are xyxy normalised to the keyframe (uniform resize, so they match the original frame): `F2_pos` bare heads, `F2_neg` helmets and helmeted heads, `F5_pos` torsos without a vest, `F5_neg` vests.

Muse Spark sees 3 frames (the keyframes of the most informative SAM window, otherwise frames spread over the clip) and returns `present` / `absent` / `uncertain` plus a short reason. `present` means at least one person is missing that protection.

Merge, per factor:

- Muse verdict exists: the clip label is Muse (`present` 1, `absent` 0, `uncertain` null). If Muse says absent, every window is 0. If Muse says present, a window stays 1 only when SAM said 1; SAM 0 or null becomes null (the clip label carries the positive). If Muse is uncertain, windows are null.
- Muse and no SAM at all: each window (from the keyframe index, or one window if there is no index) gets the clip label and empty boxes.
- SAM only: window labels stay, and the clip is 1 if any window is 1, 0 if every window is 0, otherwise null.
- Neither: clip and windows are null.

Voxel51 rows use ids `voxel51/<name>` and their existing train/test split, so the head trainer can replace the old Glimmer F2/F5 labels.

## Budgets

- SAM API: `--max-api-calls` defaults to 0. A real run with 0 refuses before any HTTP. `--dry-run` prints the plan and does not log a call. The call log is `sam_api_calls.jsonl` in the output directory outside the repo.
- Muse: `--max-calls` defaults to 0 (this run) and `--budget-total` defaults to 400 (all runs, file `muse_budget.json` in that directory). A call is refused when either cap is already reached. The counter increments before the request, so a crash still counts. Parse failures and HTTP errors count as a call and are stored as uncertain. `--dry-run` does not read `.env`, does not send HTTP, and does not touch the counter.
- Local SAM does not spend either budget. Default GPU is 1, and the process hides every other GPU before torch starts. `--dtype auto` loads bf16 when the GPU supports it, else fp16. The unsloth mirror is a transformers directory passed with `--checkpoint`. `sam3.1_multiplex.pt` is the facebookresearch/sam3 image checkpoint and is not in that mirror.

The API key is `MODEL_API_KEY` (or `META_API_KEY`), loaded the same way as `sightline_pipeline.run` (environment, then `.env` with `override=False`). The key is never printed.
