# 03 – Features (V-JEPA 2.1 extraction)

**Goal:** run the frozen V-JEPA 2.1 encoder once over every training clip and image and cache dense patch features, so head training (04) never touches the backbone.

## Model
- **V-JEPA 2.1 ViT-L/16, 384 px** (`vjepa2_1_vitl_dist_vitG_384.pt`, 304.7 M encoder params, distilled from the ViT-G teacher), released 2026-03-16 by Meta FAIR in [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2).
- **License: MIT** (repo `LICENSE`; the hub code that loads these checkpoints is MIT-headered). Any-purpose use is allowed with the copyright notice.
- Source: `https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt` (5.15 GB, includes predictor + optimizer state). Meta has **not** put 2.1 on Hugging Face yet (vjepa2 issue #137 still open; only community re-uploads exist), so we used the official file. Note: on `main`, `src/hub/backbones.py` points `VJEPA_BASE_URL` at `localhost:8300`, so `torch.hub.load(...)` does not download; we fetch the URL directly and load the encoder ourselves.
- Why ViT-L: ViT-B is the live/edge candidate, ViT-g/G (1-2 B) would be 3-7x slower for offline extraction. ViT-L is the best quality per second on a 3090 and needs only ~2.6 GiB of allocated VRAM at our batch sizes.
- `convert_checkpoint.py` keeps only the `ema_encoder` weights (what the official hub loader uses), fp16: `vjepa2_1_vitl_384_encoder_fp16.pt` (609 MB) in `F:\models\vjepa\` and `~/models/vjepa/` (WSL). The raw 5.15 GB file is kept in WSL `~/models/vjepa/`.
- Repo code: `~/vjepa2` (WSL clone of `main`), imported for `app.vjepa_2_1.models.vision_transformer.vit_large` with the hub settings (RoPE, `img_temporal_dim_size=1`, `interpolate_rope=True`).

## Sampling and outputs
| | Videos (Voxel51) | Images (Zenodo) |
|---|---|---|
| Input | 16 frames, every 4th source frame (~2.6 s at 25 fps) per window | single frame, image tokenizer (T=1) |
| Windows | about one per 2.6 s, evenly spaced, max 8 per clip (short clips: smaller step) | 1 |
| Resolution | 384 x 672 (full frame squashed slightly from 16:9, **no center crop**, so edge hazards stay visible) | same |
| Tokens | 8 x 24 x 42 = 8,064 per window | 24 x 42 = 1,008 |
| Saved `grid` | (W, 24, 42, 1024) fp16, mean over the 8 temporal tokens | (1, 24, 42, 1024) fp16 |
| Saved `tmean` | (W, 8, 1024) fp16, spatial mean per temporal token (keeps the motion order) | (1, 1, 1024) |

- Normalization: ImageNet mean/std, fp32 weights under fp16 autocast (pure `.half()` fails: the RoPE path produces fp32 q/k).
- One `.npz` per item, named by a stable hash key (`item_key()`), plus `index_r<rank>.jsonl` rows (id, split, windows, starts) and `errors_r<rank>.jsonl`.
- Output: `F:\sightline_features\vjepa2_1_vitl_384\{voxel51,zenodo}\` (not in git; C: is the WSL disk so bulk data stays on F:).
- Voxel51 items are the **658 de-duplicated clips** from the weak-label file (one row per byte-identical group), so no clip can appear in both train and test. Zenodo items: Train/Validation/Test (12,183). Augmentation (a copy of Train) and Inpaint (no labels) are skipped.

## How to run
WSL env `~/vjepa-venv` (Python 3.12). To avoid a second 3 GB CUDA download it re-uses the PyTorch 2.13 + cu132 stack of `~/vllm-venv` through a `.pth` file and adds `timm`, `scikit-learn`, `scipy`, `narwhals`, `pytest` on top (`--no-deps`).
```bash
source ~/vjepa-venv/bin/activate
python convert_checkpoint.py ~/models/vjepa/vjepa2_1_vitl_dist_vitG_384.pt ~/models/vjepa/vjepa2_1_vitl_384_encoder_fp16.pt
python extract_features.py --limit 10 --out /mnt/f/sightline_features/smoke          # smoke test
bash /mnt/f/models/wsl/run_extract.sh                                               # 2 detached ranks, one per GPU
powershell -File F:\models\wsl\check_extract.ps1                                    # status (Windows)
```
Resume is automatic: finished `.npz` files are skipped. Launcher/status scripts are copied in `scripts/`.

## Results (2026-10-31)
- Smoke test (10 clips + 10 images, 1 GPU): 0.2 min, 0 errors, peak allocated VRAM 2.6 GiB. Sanity: no NaNs; neighbouring patch tokens cosine 0.82 vs 0.57 for patches 21 columns apart (the grid is spatially coherent).
- **Full run 17:04-17:18 Tunis: 12,841 / 12,841 items, 0 errors, 13.7 min on 2x RTX 3090.**
  - Voxel51: 658 clips -> 2,184 windows in ~4.8 min (~2.3 clips/s, ~7.6 windows/s total). Bottleneck is CPU decoding of 1080p H.264 (all 8 WSL cores at ~95%), not the GPUs.
  - Zenodo: 12,183 PNGs in ~8.5 min (~24 images/s total), also decode-bound.
  - Peak allocated VRAM 2.6 GiB per GPU (nvidia-smi 3.8-5.5 GiB incl. context), so there is a lot of headroom for ViT-g or bigger batches.
  - Disk: 27.7 GiB on F: (about 2 MB per window/image).

## Known issues / open questions
- Time-mean grids lose fine motion order inside a 2.6 s window; `tmean` keeps the per-step global vectors for the temporal factors (F6/F7) and F11 later.
- The 2.1 predictor (needed for F11 prediction-error anomaly scores) is in the raw checkpoint but not used yet.
- Zenodo is a still-image domain (image tokenizer); Voxel51 is video. The head sees both, so per-source metrics are reported separately in 04.
