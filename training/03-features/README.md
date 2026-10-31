# 03 · Features (V-JEPA 2.1 extraction)

**Goal:** run the frozen V-JEPA 2.1 encoder ([facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2), MIT) once over every clip and cache the dense features, so head training (04) never touches the backbone.

## Inputs
- `training/data/manifest.jsonl` (video clips)
- V-JEPA 2.1 weights: ViT-B (80M) for live use, ViT-L (300M) offline. The g/G sizes are probably too heavy.

## Outputs (planned, gitignored)
- `training/features/` and any `*.npy`/`*.npz`/`*.h5`/`*.parquet` there are ignored by `training/.gitignore`; only the small index/stats summaries should be committed.
- `training/features/<model>/<clip_id>/window_XXX.npy`: per-patch features on the 24×24 grid (384 px input, 64-frame windows), with the stride recorded
- A predictor-error map per window for F11 (anomaly)
- An index file mapping clip → windows → time span

## How to run
Not written yet.

## Results
Not run yet.

## Known issues / open questions
- Check the exact loading API, checkpoint names and the token layout (temporal tubelets vs 24×24 spatial grid) against the vjepa2 repo before writing the extractor.
- Storage: 24×24×D per window adds up. Pick float16 and the window stride after measuring.
