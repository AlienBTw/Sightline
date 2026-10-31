"""Phase 03: extract frozen V-JEPA 2.1 features for Sightline.

One item = one Voxel51 clip (several 16-frame windows) or one Zenodo image.
Per item we save an uncompressed .npz with
  grid   (W, Gh, Gw, D) float16  patch tokens, mean-pooled over the window's time axis
                                 (W = windows; images have W = 1)
  tmean  (W, Tt, D)     float16  spatial mean per temporal token (videos; images Tt = 1)
  starts (W,)           int32    first source frame of each window (videos)
and one JSON line per item in index_r<rank>.jsonl.

Runs in WSL (~/vjepa-venv) with the vjepa2 repo on sys.path. Launch one process
per GPU with --rank/--world; items are assigned round-robin so shards balance.
Resume: items whose .npz already exists are skipped.

Pure helpers (planning, naming, sharding) import without torch so they can be
unit-tested on Windows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]

# ---- model / sampling constants (V-JEPA 2.1 ViT-L/16, 384) ----
PATCH = 16
TUBELET = 2
FRAMES_PER_WINDOW = 16          # 8 temporal tokens
FRAME_STEP = 4                  # ~6 fps at 25 fps source -> a window spans ~2.6 s
MAX_WINDOWS = 8
IN_H, IN_W = 384, 672           # 16:9-ish, keeps the full field of view (24 x 42 patches)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ZENODO_SPLITS = ("train", "validation", "test")   # augmentation = copy of train, inpaint = unlabeled


# ---------------------------------------------------------------- pure helpers
def item_key(item_id: str) -> str:
    """Filesystem-safe, stable key (ids contain '/', spaces and non-ASCII)."""
    src = item_id.split("/", 1)[0]
    return f"{src}_{hashlib.sha1(item_id.encode('utf-8')).hexdigest()[:16]}"


def window_starts(n_frames: int, span: int = FRAMES_PER_WINDOW * FRAME_STEP,
                  max_windows: int = MAX_WINDOWS) -> list[int]:
    """Evenly spaced window starts covering the clip, about one window per span."""
    n_frames = max(1, int(n_frames))
    n_win = max(1, min(max_windows, int(round(n_frames / span))))
    last = max(0, n_frames - span)
    if n_win == 1:
        return [last // 2]
    return [int(round(i * last / (n_win - 1))) for i in range(n_win)]


def window_indices(start: int, n_frames: int, frames: int = FRAMES_PER_WINDOW,
                   step: int = FRAME_STEP) -> list[int]:
    """Frame indices of one window; shrinks the step for short clips, clamps to the last frame."""
    n_frames = max(1, int(n_frames))
    if n_frames < frames * step:
        step = max(1, n_frames // frames)
        start = min(start, max(0, n_frames - frames * step))
    return [min(n_frames - 1, start + i * step) for i in range(frames)]


def shard(items: list, rank: int, world: int) -> list:
    return items[rank::world]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_items(training: Path, sources: list[str], labels_path: Path | None = None) -> list[dict]:
    """Items to extract. Voxel51 = the de-duplicated clips in the weak-label file
    (one row per unique clip); Zenodo = Train/Validation/Test images."""
    items = []
    if "voxel51" in sources:
        labels_path = labels_path or training / "02-labeling/outputs/weak_labels_v2_vllm.jsonl"
        man = {m["id"]: m for m in load_jsonl(training / "data/manifest.jsonl")}
        for r in load_jsonl(labels_path):
            m = man.get(r["clip_id"], {})
            items.append({"id": r["clip_id"], "source": "voxel51", "kind": "video", "path": r["path"],
                          "split": r["split"], "frames": int(m.get("frames") or 0)})
    if "zenodo" in sources:
        for m in load_jsonl(training / "data/manifest.jsonl"):
            if m["source"] == "zenodo_tablesaw" and m.get("split") in ZENODO_SPLITS:
                items.append({"id": m["id"], "source": "zenodo", "kind": "image", "path": m["path"],
                              "split": m["split"], "label_path": m.get("label_path"),
                              "n_hand_boxes": m.get("n_hand_boxes", 0)})
    for it in items:
        it["key"] = item_key(it["id"])
    return items


# ---------------------------------------------------------------- IO (cv2)
def _resize_rgb(bgr):
    import cv2
    return cv2.cvtColor(cv2.resize(bgr, (IN_W, IN_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def read_video_windows(path: str, n_frames_hint: int):
    """Returns uint8 array (W, T, H, W, 3), starts, actual frame count."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_frames_hint
    starts = window_starts(n)
    wins = [window_indices(s, n) for s in starts]
    need = sorted({i for w in wins for i in w})
    got, idx, last = {}, 0, None
    want = set(need)
    while idx <= need[-1]:
        ok = cap.grab()
        if not ok:
            break
        if idx in want:
            ok, fr = cap.retrieve()
            if ok:
                last = _resize_rgb(fr)
                got[idx] = last
        idx += 1
    cap.release()
    if not got:
        raise RuntimeError(f"no frames decoded: {path}")
    keys = sorted(got)
    def nearest(i):
        if i in got:
            return got[i]
        best = min(keys, key=lambda k: abs(k - i))
        return got[best]
    arr = np.stack([np.stack([nearest(i) for i in w]) for w in wins])
    return arr, starts, idx


def read_image(path: str):
    import cv2
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read {path}")
    return _resize_rgb(bgr)


# ---------------------------------------------------------------- model
def load_encoder(vjepa_repo: str, weights: str, device):
    import torch
    sys.path.insert(0, vjepa_repo)
    from app.vjepa_2_1.models import vision_transformer as vit
    enc = vit.vit_large(patch_size=PATCH, img_size=(384, 384), num_frames=64, tubelet_size=TUBELET,
                        use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False,
                        use_rope=True, img_temporal_dim_size=1, interpolate_rope=True)
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    if "ema_encoder" in sd:  # raw Meta checkpoint
        sd = sd["ema_encoder"]
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    enc.load_state_dict(sd, strict=True)
    return enc.to(device).eval().float()   # fp32 weights + fp16 autocast (RoPE mixes dtypes under .half())


def to_input(u8, device):
    """uint8 (..., H, W, 3) -> normalized float (B, 3, T, H, W)."""
    import torch
    x = torch.from_numpy(u8).to(device, non_blocking=True)
    if x.ndim == 4:                    # (B, H, W, 3) images -> T = 1
        x = x.unsqueeze(1)
    x = x.permute(0, 4, 1, 2, 3).float().div_(255)
    m = torch.tensor(MEAN, device=device).view(1, 3, 1, 1, 1)
    s = torch.tensor(STD, device=device).view(1, 3, 1, 1, 1)
    return (x - m) / s


def tokens_to_outputs(tok, t_tokens: int):
    """(B, Tt*Gh*Gw, D) -> grid (B, Gh, Gw, D) time-mean, tmean (B, Tt, D)."""
    gh, gw = IN_H // PATCH, IN_W // PATCH
    b, n, d = tok.shape
    assert n == t_tokens * gh * gw, (n, t_tokens, gh, gw)
    t = tok.view(b, t_tokens, gh, gw, d).float()
    return t.mean(1).half(), t.mean((2, 3)).half()


# ---------------------------------------------------------------- main loop
class _DS:
    def __init__(self, items, root):
        self.items, self.root = items, root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        p = str(self.root / it["path"])
        try:
            if it["kind"] == "video":
                arr, starts, n = read_video_windows(p, it.get("frames", 0))
                return {"i": i, "u8": arr, "starts": starts, "n": n, "err": None}
            return {"i": i, "u8": read_image(p), "starts": [0], "n": 1, "err": None}
        except Exception as e:  # recorded, never fatal
            return {"i": i, "u8": None, "starts": [], "n": 0, "err": repr(e)}


def run(args):
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(f"cuda:{args.gpu}")
    out = Path(args.out)
    items = build_items(TRAINING, args.sources.split(","))
    if args.limit:
        items = [it for it in items if it["source"] == "voxel51"][: args.limit] + \
                [it for it in items if it["source"] == "zenodo"][: args.limit]
    items = shard(items, args.rank, args.world)
    todo = [it for it in items if not (out / it["source"] / f"{it['key']}.npz").exists()]
    print(f"[r{args.rank}] items={len(items)} todo={len(todo)}", flush=True)
    for s in ("voxel51", "zenodo"):
        (out / s).mkdir(parents=True, exist_ok=True)
    enc = load_encoder(args.vjepa_repo, args.weights, dev)
    idx_f = open(out / f"index_r{args.rank}.jsonl", "a", encoding="utf-8")
    err_f = open(out / f"errors_r{args.rank}.jsonl", "a", encoding="utf-8")
    vids = [it for it in todo if it["kind"] == "video"]
    imgs = [it for it in todo if it["kind"] == "image"]
    t0, done = time.time(), 0

    def save(it, grid, tmean, starts, n):
        nonlocal done
        dst = out / it["source"] / f"{it['key']}.npz"
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(tmp, grid=grid, tmean=tmean, starts=np.asarray(starts, np.int32))
        os.replace(tmp, dst)
        row = {k: it[k] for k in ("id", "key", "source", "kind", "split", "path") if k in it}
        row.update({"windows": int(grid.shape[0]), "starts": [int(s) for s in starts], "frames": int(n),
                    "grid": list(grid.shape[1:3]), "dim": int(grid.shape[-1])})
        idx_f.write(json.dumps(row, ensure_ascii=False) + "\n"); idx_f.flush()
        done += 1
        if done % 25 == 0:
            el = time.time() - t0
            print(f"[r{args.rank}] {done}/{len(todo)} {done/el:.2f} it/s "
                  f"vram_peak={torch.cuda.max_memory_allocated(dev)/2**30:.1f}GiB", flush=True)

    def err(it, msg):
        err_f.write(json.dumps({"id": it["id"], "error": msg}, ensure_ascii=False) + "\n"); err_f.flush()

    coll = lambda b: b
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        if vids:
            dl = torch.utils.data.DataLoader(_DS(vids, TRAINING), batch_size=1, num_workers=args.workers,
                                             collate_fn=coll, prefetch_factor=2)
            for (b,) in dl:
                it = vids[b["i"]]
                if b["err"]:
                    err(it, b["err"]); continue
                grids, tms = [], []
                u8 = b["u8"]
                for j in range(0, len(u8), args.video_batch):
                    tok = enc(to_input(u8[j:j + args.video_batch], dev))
                    g, tm = tokens_to_outputs(tok, FRAMES_PER_WINDOW // TUBELET)
                    grids.append(g.cpu()); tms.append(tm.cpu())
                save(it, torch.cat(grids).numpy(), torch.cat(tms).numpy(), b["starts"], b["n"])
        if imgs:
            dl = torch.utils.data.DataLoader(_DS(imgs, TRAINING), batch_size=args.image_batch,
                                             num_workers=args.workers, collate_fn=coll, prefetch_factor=4)
            for batch in dl:
                good = [b for b in batch if not b["err"]]
                for b in batch:
                    if b["err"]:
                        err(imgs[b["i"]], b["err"])
                if not good:
                    continue
                tok = enc(to_input(np.stack([b["u8"] for b in good]), dev))
                g, tm = tokens_to_outputs(tok, 1)
                g, tm = g.cpu().numpy(), tm.cpu().numpy()
                for k, b in enumerate(good):
                    save(imgs[b["i"]], g[k:k + 1], tm[k:k + 1], [0], 1)
    el = time.time() - t0
    print(f"[r{args.rank}] DONE {done} items in {el/60:.1f} min "
          f"vram_peak={torch.cuda.max_memory_allocated(dev)/2**30:.1f}GiB", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/f/sightline_features/vjepa2_1_vitl_384")
    ap.add_argument("--weights", default=os.path.expanduser("~/models/vjepa/vjepa2_1_vitl_384_encoder_fp16.pt"))
    ap.add_argument("--vjepa-repo", default=os.path.expanduser("~/vjepa2"))
    ap.add_argument("--sources", default="voxel51,zenodo")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--video-batch", type=int, default=4)
    ap.add_argument("--image-batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items per source")
    args = ap.parse_args(argv)
    if args.gpu is None:
        args.gpu = args.rank
    run(args)


if __name__ == "__main__":
    main()
