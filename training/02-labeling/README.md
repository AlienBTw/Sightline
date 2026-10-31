# 02 · Labeling (Muse Glimmer weak labels)

**Goal:** clip-level weak labels for F1–F10 on every manifest clip, from Muse Glimmer 30B running locally. These labels are the supervision for the concept-bottleneck head (04), trained with multiple-instance learning. F11 isn't labeled here, because it comes from V-JEPA prediction error.

## Inputs

- `training/data/manifest.jsonl` (from 01-data)
- Ollama serving `muse-glimmer:30b` at `http://localhost:11434/v1` (OpenAI-compatible; image in, text out). It can be overridden with `--base-url` / `--model` or the `GLIMMER_BASE_URL` / `GLIMMER_MODEL` environment variables.
- Or a batched server: `--backend vllm` (used for the full run; `http://localhost:30001/v1`, model `muse-glimmer-30b-int4-vllm`) or `--backend sglang` (port 30000). Server scripts: [`scripts/`](scripts/README.md).
- The prompts [`prompts/factors_v1.md`](prompts/factors_v1.md) and [`prompts/factors_v2.md`](prompts/factors_v2.md) (`v2-7238ea9f`, the one used for the full run), versioned. The factor definitions are filled in from `configs/factors.yaml`.

## Outputs

- `outputs/weak_labels_v1.jsonl` / `outputs/weak_labels_v2_vllm.jsonl` (the full run; small, OK to commit). One line per clip: `clip_id, source, split, source_label, factor_hints, hint_agreement, model, prompt_id, keyframes, factors{F1..F10: label present|absent|uncertain, confidence, votes, frames_present}`.
- `outputs/weak_labels_v1.errors.jsonl`: frames or clips that failed
- `training/cache/glimmer/<prompt_id>/<model>/s<max_side>/<clip>/fNNNNNN.json` (gitignored): one validated answer per frame, plus the raw reply. Reruns skip cached frames.

## How it works

1. **Keyframes:** K frames per clip (default 4) at the centres of K equal segments, resized so the long side is at most 768 px, sent as JPEG. An image counts as one frame.
2. **One request per frame:** temperature 0, `response_format: json_object`, and `reasoning_effort: none` by default. Glimmer is a reasoning model, and its hidden reasoning counts against `max_tokens` (default 2048). The model must return `{"factors": {"F1": {"state", "confidence", "evidence"}, …, "F10": …}}`.
3. **Validation** (`glimmer_schema.py`, documented in `schema/frame_labels.schema.json`):
   - It requires all of F1–F10 and rejects F11 or any unknown keys.
   - `state` must be one of present, absent or uncertain.
   - `confidence` must be in [0, 1], so 85 meaning "85%" is rejected.
   - It tolerates code fences, `<think>` blocks and chatter around the object.
   - On an invalid answer it retries (2 extra attempts by default) and feeds the error back to the model.
4. **Clip aggregation (MIL view):**
   - *present* if any frame says present with confidence ≥ 0.5 (max confidence)
   - *absent* if every frame says absent (min confidence)
   - *uncertain* otherwise
5. **Hint agreement:** where the dataset class implies a factor (e.g. Safe Walkway Violation → F8 present), the output records whether Glimmer agrees. This is a cheap sanity check on label quality.
6. Byte-identical duplicate clips are labeled only once. Use `--keep-duplicates` to label every copy.

## How to run (repo root, Ollama running)

```powershell
python training/02-labeling/label_glimmer.py --plan-only --limit 3     # no model calls; saves keyframes to cache/glimmer/_preview
python training/02-labeling/label_glimmer.py --limit 3                 # dry run: 3 clips x 4 frames = 12 calls
python training/02-labeling/label_glimmer.py --sources voxel51_safe_unsafe
python -m pytest training/tests -q                                     # offline tests (no Ollama)
```

Useful flags: `--reasoning-effort none|low|medium|high|default`, `--max-tokens 2048`, `--split train`, `--ids voxel51/0_tr1`, `--keyframes 6`, `--max-side 1024`, `--retries 3`, `--no-json-mode` (for when the server rejects `response_format`), and `--present-threshold 0.6`.

## Results

Training PC, Oct 30 (Tunis), `muse-glimmer:30b` through Ollama 0.40.4, prompt `v1-143ada25`:

| Run | Result |
|---|---|
| `--plan-only --limit 3` | 12 keyframes (4 per clip) decoded and saved in 2.2 s |
| `--limit 3` with the old default `max_tokens=900`, thinking on | **failed**: 3 of 4 frames on `0_te1` gave up after 3 attempts (cut-off JSON, empty reply, missing `factors`). About 27 s per attempt |
| `--limit 3`, JSON mode, `--reasoning-effort none` (new default) | **3/3 clips, 12/12 frames valid on the first attempt, 143 s total, 11.5 s per frame** (median 11.5, max 12.3), about 315 completion tokens and about 1,117 prompt tokens per frame |
| `0_te1` at `--max-side 1280`, effort none | 12.0 s per frame, the same labels as at 768 |
| `0_te1` at `--max-side 1280`, effort low | 21.9 s per frame. More decisive: F2/F5 present, F1/F9 uncertain |

Probe on one frame, 900 tokens: JSON mode with default thinking gave cut-off JSON (`finish_reason=length`, about 3.5k characters of reasoning). Without JSON mode the content was empty, for the same reason. `think: false` was ignored by the OpenAI-compatible endpoint. `reasoning_effort: none` worked in 12.3 s, and `reasoning_effort: low` worked in 22.0 s. With `max_tokens` 4096 and default thinking it also worked, in 36.5 s. **JSON mode with images works**, so `--no-json-mode` isn't needed.

**Labels from the 3-clip run** (all three are "Safe Walkway Violation", so the hint is F8 present):

| Clip | Present | Absent | Uncertain | F8 vs hint |
|---|---|---|---|---|
| `0_te1` | – | F1, F6, F10 | F2–F5, F7, F8, F9 | uncertain (2 frames absent, 2 uncertain) |
| `0_te10` | – | F1, F6, F7, F9, F10 | F2–F5, F8 | uncertain (4 uncertain) |
| `0_te11` | – | F1, F6, F7, F9, F10 | F2–F5, F8 | uncertain (2 absent, 2 uncertain) |

Evidence on `0_te1` F8 reads like "people on marked pedestrian walkway". Hint agreement is 0/3 confirmed (3 uncertain, 0 contradicted).

**Estimate for a full run (not started):** 658 unique Voxel51 clips × 4 frames = 2,632 calls. At 11.5 s that's about **8.4 h** with reasoning off; at about 22 s it's about 16 h with `low`. Zenodo images (649 now, roughly 25k if fully downloaded) would be 11.5 s each, so they should be sampled, not labeled in full.

### Prompt eval: v1 vs v2, reasoning none vs low (Oct 31, 01:20 Tunis)

| Config | F7 P / R / unc | F8 P / R / unc | F9 P / R / unc | Uncertain, all F1-F10 | Present, all | s/call (mean / median) |
|---|---|---|---|---|---|---|
| v1 / none | 1.00 / 1.00 / 0.00 | - / 0.00 / 1.00 | 1.00 / 0.25 / 0.86 | 0.59 | 0.05 | 11.80 / 11.52 |
| **v2 / none** | 1.00 / 1.00 / 0.00 | 0.56 / 1.00 / 0.00 | 1.00 / 0.25 / 0.00 | 0.30 | 0.22 | **11.13 / 11.13** |
| v1 / low | 1.00 / 0.25 / 0.43 | - / 0.00 / 1.00 | 1.00 / 0.25 / 0.86 | 0.62 | 0.13 | 22.68 / 22.64 |
| v2 / low | 1.00 / 0.25 / 0.29 | 0.33 / 0.20 / 0.44 | 1.00 / 0.25 / 0.00 | 0.30 | 0.23 | 21.00 / 20.78 |

Eval set: 23 unique Voxel51 clips (`outputs/eval/eval_set.json`), 4 keyframes at 768 px, 92 calls per config. GT clips per factor: F7 7 (4 present / 3 absent), F8 9 (5 / 4), F9 7 (4 / 3). P = precision, R = recall (uncertain counts as a miss), unc = share of hinted clips labeled uncertain. F8 precision of 0.56 equals the base rate: v2/none says F8 present on all 4 Safe Walkway clips too.

**Chosen setting: `v2` prompt (`v2-7238ea9f`), `--reasoning-effort none`.** Full run with it: about 8.1 h for 2,632 calls (about 7.9 h after the cached eval clips). Not started.

### Full run: vLLM INT4, v2/none, concurrency 32 (Oct 31, 13:38-16:14 Tunis)

Output: `outputs/weak_labels_v2_vllm.jsonl` (658 rows, one per unique Voxel51 clip; train 541 / test 117). Summary: [`outputs/eval/full_run_summary_vllm.md`](outputs/eval/full_run_summary_vllm.md), regenerate with `python training/02-labeling/summarize_labels.py`.

- 658/658 clips, 2,536 new calls + 96 cached frames, **0 errors, 0 malformed rows**, 2 h 36 min at 16.3 calls/min (vs ~7.9 h estimated on Ollama).
- Clip-level present / absent / uncertain: F1 29/620/9, F2 528/3/127, F3 7/2/649, F4 0/4/654, F5 513/25/120, F6 9/649/0, F7 26/632/0, F8 629/28/1, F9 120/520/18, F10 9/649/0.
- Against the class hints: **F7 P 0.89 / R 0.41**, **F8 P 0.73 / R 0.92 but specificity 0.01** (present on 71/72 Safe Walkway clips), **F9 P 1.00 / R 0.34**.
- Use for training: Glimmer for F1, F2, F5, F6, F10 (after the 05-eval spot check); class hints for F7/F8/F9. F3/F4 (99% uncertain) and F6/F10 (9 positives each) need other data.

## Known issues

- **Weak on the dataset's own hazard:** v1 never decides F8 (uncertain on 9/9 hinted clips). v2 decides, but says F8 present on safe-walkway clips too (precision 0.56 = base rate), and no frame-vote threshold separates them. F9 recall is 0.25 with both prompts. For Voxel51, use the class hints for F7/F8/F9 and treat Glimmer as the source for the other factors.
- PPE (F2–F5) came back *uncertain* on 12/12 frames with reasoning off. Going to 1280 px didn't change that and cost no extra time, which suggests the model resizes internally. With `low` it commits.
- Glimmer sometimes appends `<|eot|>` after the JSON. The parser ignores it (tested).
- Still frames are weak evidence for the motion factors F6 and F7.
- Cost (measured): about 11.1-11.8 s per call with reasoning off and 21-23 s with `low` (4-config eval, 92 calls each).
- Weak labels need the 10% human spot-check planned in 05-eval before they're trusted.

## SGLang backend (faster, batched)

- **Server:** WSL2 Ubuntu-24.04, `sglang==0.5.23` in `~/sglang-venv`, started with `F:\models\wsl\serve_sglang.sh` (TP=2 over both 3090s, port 30000, `--reasoning-parser muse`, `--enable-multimodal`).
- **Weights:** `RedHatAI/Muse-Glimmer-30B-INT4` (GPTQ W4A16 compressed-tensors, Apache-2.0, vision tower unquantized), stored in `F:\models\muse-glimmer-30b-int4-w4a16`. Loads and serves images in SGLang 0.5.23 (verified 2026-10-31) with triton attention, CUDA graphs off and a local multimem patch (see the 00-setup LOG). Weights are copied to WSL ext4 at `~/models/` for fast loading. Measured on the 23-clip eval (v2/none, concurrency 16): **8.6 calls/min vs 5.4 on Ollama (1.6x)**, but F7 recall 0.50 and F8 recall 0.20 vs 1.00/1.00 on Ollama, so the labels do not match Ollama's quality yet. **Update 13:34:** that SGLang run still had reasoning on (the template has no 'none'; 614 vs 308 completion tokens), so the drop was not the quantization. Use `--backend vllm` instead (server `F:\models\wsl\serve_vllm.sh`, port 30001): it pre-fills the answer channel to skip reasoning, and on the same eval it scored F7 1.00/0.75, F8 0.50/0.80, F9 1.00/0.25 at 13.2 calls/min (c=16) / 21.6 (c=32). See the LOG.
- **Run:**

```powershell
python training/02-labeling/label_glimmer.py --backend sglang --concurrency 16 --limit 3
python training/02-labeling/eval_labels.py --backend sglang --configs v2:none --concurrency 16   # writes outputs/eval/*_sglang.*
```

- The cache key includes the model name, so Ollama and SGLang answers are kept separate. The INT4 GPTQ weights aren't the same quantization as Ollama's GGUF Q4, so compare the labels on the eval set before switching over.