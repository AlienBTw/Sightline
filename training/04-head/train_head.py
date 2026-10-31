"""Phase 04: concept-bottleneck head on frozen V-JEPA 2.1 features.

Head (per factor F1-F10, order from configs/factors.yaml):
  tokens (N, 1024) -> LayerNorm -> Linear(1024, H) -> GELU -> Dropout
  attention logits  a_f(p) = w_f . h_p   (softmax over all patches of all windows = MIL pooling)
  patch evidence    s_f(p) = u_f . h_p
  factor logit      z_f    = sum_p softmax(a_f)(p) * s_f(p) + b_f
The softmax weights and sigmoid(s_f) give a per-factor spatial map for circling (phase 06).

Labels:
  Voxel51 (658 unique clips; duplicate files were merged in 02-labeling, so no clip is in both splits)
    F1, F2, F5, F6, F10 : Glimmer clip label (present=1, absent=0, uncertain=ignored)
    F3, F4              : ignored by default (~99% uncertain)
    F7, F8, F9          : dataset class hint where the class defines it, else ignored
  Zenodo table saw (images): F1 = hand box present; other factors ignored.
    Hand boxes also give a patch-level F1 target (aux loss on s_F1).

Subcommands:
  cache  - pool saved grids 2x2 (24x42 -> 12x21) into one fp16 memmap on WSL ext4
  train  - train one config, write checkpoint (ignored) + results JSON (committed)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
FACTORS = [f"F{i}" for i in range(1, 11)]
GLIMMER_FACTORS = ["F1", "F2", "F3", "F4", "F5", "F6", "F10"]
HINT_FACTORS = ["F7", "F8", "F9"]
DEFAULT_IGNORE = ["F3", "F4"]
POOL = 2
GRID = (24, 42)


# ---------------------------------------------------------------- labels (pure)
def voxel_targets(row: dict, ignore=DEFAULT_IGNORE):
    """Weak-label row -> (y[10], mask[10]) lists."""
    y, m = [0.0] * 10, [0.0] * 10
    for i, f in enumerate(FACTORS):
        if f in ignore:
            continue
        if f in HINT_FACTORS:
            lab = (row.get("factor_hints") or {}).get(f)
        else:
            lab = (row.get("factors") or {}).get(f, {}).get("label")
        if lab == "present":
            y[i], m[i] = 1.0, 1.0
        elif lab == "absent":
            m[i] = 1.0
    return y, m


def zenodo_targets(n_boxes: int):
    y, m = [0.0] * 10, [0.0] * 10
    m[0] = 1.0
    y[0] = 1.0 if n_boxes > 0 else 0.0
    return y, m


def read_yolo(path) -> list[tuple]:
    """YOLO lines 'cls cx cy w h' (normalized) -> list of (cx, cy, w, h)."""
    boxes = []
    if not path or not os.path.exists(path):
        return boxes
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 5:
            boxes.append(tuple(float(v) for v in p[1:5]))
    return boxes


def box_patch_mask(boxes, gh: int, gw: int, min_frac: float = 0.25):
    """Patch grid (gh x gw) target: 1 where a hand box covers >= min_frac of the cell.
    Works in normalized coordinates (the extractor squashes the full frame, no crop)."""
    out = [[0.0] * gw for _ in range(gh)]
    for cx, cy, w, h in boxes:
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        for r in range(gh):
            cy0, cy1 = r / gh, (r + 1) / gh
            oy = max(0.0, min(y1, cy1) - max(y0, cy0))
            if oy <= 0:
                continue
            for c in range(gw):
                cx0, cx1 = c / gw, (c + 1) / gw
                ox = max(0.0, min(x1, cx1) - max(x0, cx0))
                if ox * oy * gh * gw >= min_frac:
                    out[r][c] = 1.0
    return out


def cv_fold(item_id: str, cv: int) -> int:
    import hashlib
    return int(hashlib.md5(f"cv:{item_id}".encode()).hexdigest()[:8], 16) % cv


def assign_split(item: dict, val_frac: float, seed: int, cv: int = 0, fold: int = 0) -> str:
    """train/val/test. Voxel51: keep the dataset test split (or, with cv>0, use hash fold
    `fold` of all 658 unique clips as test); carve val from the rest by a stable hash.
    Zenodo: train/validation/test as published."""
    s = item["split"]
    if item["source"] == "zenodo":
        return {"train": "train", "validation": "val", "test": "test"}.get(s, "skip")
    if cv > 0:
        if cv_fold(item["id"], cv) == fold:
            return "test"
    elif s == "test":
        return "test"
    import hashlib
    h = int(hashlib.md5(f"{seed}:{item['id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


def pos_weights(ys, ms, cap: float = 20.0):
    """Per-factor neg/pos ratio over labeled entries, clamped to [1, cap]."""
    out = []
    for i in range(len(FACTORS)):
        pos = sum(1 for y, m in zip(ys, ms) if m[i] and y[i] > 0.5)
        neg = sum(1 for y, m in zip(ys, ms) if m[i] and y[i] <= 0.5)
        out.append(float(min(cap, max(1.0, neg / pos))) if pos else 1.0)
    return out


def best_f1_threshold(y, p):
    best = (0.0, 0.5)
    for t in sorted(set(round(float(v), 4) for v in p)):
        tp = sum(1 for a, b in zip(y, p) if a > 0.5 and b >= t)
        fp = sum(1 for a, b in zip(y, p) if a <= 0.5 and b >= t)
        fn = sum(1 for a, b in zip(y, p) if a > 0.5 and b < t)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


# ---------------------------------------------------------------- cache
def cmd_cache(a):
    import numpy as np
    feat = Path(a.features)
    rows = {}
    for p in sorted(feat.glob("index_r*.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            rows[r["key"]] = r
    labels = {r["clip_id"]: r for r in map(json.loads, open(TRAINING / a.labels, encoding="utf-8"))}
    man = {m["id"]: m for m in map(json.loads, open(TRAINING / "data/manifest.jsonl", encoding="utf-8"))}
    items = []
    for r in rows.values():
        if r["source"] == "zenodo" and r["split"] == "train" and a.zenodo_train_stride > 1:
            n = int(r["id"].rsplit("_", 1)[-1]) if r["id"].rsplit("_", 1)[-1].isdigit() else 0
            if n % a.zenodo_train_stride:
                continue
        items.append(r)
    items.sort(key=lambda r: (r["source"], r["id"]))
    gh, gw = GRID[0] // POOL, GRID[1] // POOL
    total = sum(r["windows"] for r in items)
    out = Path(os.path.expanduser(a.cache)); out.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out / "tokens.npy", mode="w+", dtype=np.float16,
                                   shape=(total, gh * gw, 1024))
    meta, off, t0 = [], 0, time.time()
    for k, r in enumerate(items):
        z = np.load(feat / r["source"] / f"{r['key']}.npz")
        g = z["grid"].astype(np.float32)                       # (W, 24, 42, D)
        w = g.shape[0]
        g = g.reshape(w, gh, POOL, gw, POOL, -1).mean((2, 4)).reshape(w, gh * gw, -1)
        mm[off:off + w] = g.astype(np.float16)
        if r["source"] == "voxel51":
            y, m = voxel_targets(labels[r["id"]], ignore=[])
            pm = None
        else:
            mr = man[r["id"]]
            y, m = zenodo_targets(int(mr.get("n_hand_boxes") or 0))
            pm = box_patch_mask(read_yolo(str(TRAINING / mr["label_path"])) if mr.get("label_path") else [], gh, gw)
        meta.append({"id": r["id"], "source": r["source"], "split": r["split"], "off": off, "n": w,
                     "y": y, "m": m, "patch": pm})
        off += w
        if k % 500 == 0:
            print(f"cache {k}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    mm.flush()
    with open(out / "meta.jsonl", "w", encoding="utf-8") as f:
        for r in meta:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"cache done items={len(meta)} windows={total} in {time.time()-t0:.0f}s -> {out}")


# ---------------------------------------------------------------- model / train
def build_model(dim=1024, hidden=256, drop=0.1, nf=10):
    import torch.nn as nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.proj = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop))
            self.att = nn.Linear(hidden, nf)
            self.val = nn.Linear(hidden, nf)

        def forward(self, x, valid):
            import torch
            h = self.proj(self.norm(x))
            a = self.att(h).masked_fill(~valid[..., None], float("-inf"))
            w = torch.softmax(a, dim=1)                       # (B, N, F)
            s = self.val(h)                                   # (B, N, F)
            return (w * s).sum(1), s, w

    return Head()


def masked_bce(logits, y, m, pos_weight):
    """Mean BCE per factor over labeled entries, then mean over factors that have any."""
    import torch
    import torch.nn.functional as F
    l = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    num, den = (l * m).sum(0), m.sum(0)
    has = den > 0
    if not has.any():
        return logits.sum() * 0
    return (num[has] / den[has]).mean()


def cmd_train(a):
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    cache = Path(os.path.expanduser(a.cache))
    tok = np.load(cache / "tokens.npy", mmap_mode="r")
    meta = [json.loads(l) for l in open(cache / "meta.jsonl", encoding="utf-8")]
    ignore = [f for f in a.ignore.split(",") if f]
    labels = {r["clip_id"]: r for r in map(json.loads, open(TRAINING / a.labels, encoding="utf-8"))}
    for r in meta:
        if r["source"] == "voxel51":
            r["y"], r["m"] = voxel_targets(labels[r["id"]], ignore=ignore)
        r["set"] = assign_split(r, a.val_frac, a.seed, a.cv, a.fold)
    if not a.use_zenodo:
        meta = [r for r in meta if r["source"] != "zenodo"]
    tr = [r for r in meta if r["set"] == "train"]
    va = [r for r in meta if r["set"] == "val"]
    te = [r for r in meta if r["set"] == "test"]
    pw = torch.tensor(pos_weights([r["y"] for r in tr], [r["m"] for r in tr], a.pw_cap), device=dev)
    print("split sizes", {k: len(v) for k, v in (("train", tr), ("val", va), ("test", te))}, "pos_weight",
          [round(v, 2) for v in pw.tolist()], flush=True)
    # load everything needed into RAM (pooled tokens are small: ~0.5 MB per window)
    def batchify(rs, train):
        xs = []
        for r in rs:
            x = np.asarray(tok[r["off"]:r["off"] + r["n"]])          # (W, P, D)
            if train and r["n"] > 1 and a.window_drop > 0:
                keep = [i for i in range(r["n"]) if random.random() > a.window_drop] or [random.randrange(r["n"])]
                x = x[keep]
            xs.append(x.reshape(-1, x.shape[-1]))
        n = max(len(x) for x in xs)
        X = np.zeros((len(xs), n, xs[0].shape[-1]), np.float16)
        V = np.zeros((len(xs), n), bool)
        for i, x in enumerate(xs):
            X[i, :len(x)] = x; V[i, :len(x)] = True
        X, V = torch.from_numpy(X).to(dev).float(), torch.from_numpy(V).to(dev)
        if train and a.token_drop > 0:
            V = V & (torch.rand_like(V, dtype=torch.float) > a.token_drop)
            V[:, 0] |= ~V.any(1)
        y = torch.tensor([r["y"] for r in rs], device=dev)
        m = torch.tensor([r["m"] for r in rs], device=dev)
        P = [r.get("patch") for r in rs]
        return X, V, y, m, P

    model = build_model(hidden=a.hidden, drop=a.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)
    src_w = {"voxel51": 1.0, "zenodo": 1.0}
    nv = sum(1 for r in tr if r["source"] == "voxel51"); nz = len(tr) - nv
    if nz:
        src_w["zenodo"] = a.zenodo_share * nv / max(1, nz) / max(1e-6, 1 - a.zenodo_share)
    weights = [src_w[r["source"]] for r in tr]

    def predict(rs):
        model.eval(); out, pmaps = [], []
        with torch.no_grad():
            for i in range(0, len(rs), 32):
                X, V, y, m, P = batchify(rs[i:i + 32], False)
                z, s, w = model(X, V)
                out.append(torch.sigmoid(z).cpu())
                pmaps.append(torch.sigmoid(s[..., 0]).cpu())   # F1 patch evidence
        model.train()
        return torch.cat(out).numpy(), pmaps

    def evaluate(rs):
        if not rs:
            return {}, None
        p, pmaps = predict(rs)
        res = {}
        for i, f in enumerate(FACTORS):
            yy = [r["y"][i] for r, mm in zip(rs, [r["m"] for r in rs]) if mm[i]]
            pp = [float(p[k, i]) for k, r in enumerate(rs) if r["m"][i]]
            d = {"n": len(yy), "pos": int(sum(yy))}
            if len(set(yy)) == 2:
                d["auroc"] = float(roc_auc_score(yy, pp)); d["ap"] = float(average_precision_score(yy, pp))
                d["base_rate"] = d["pos"] / d["n"]
            res[f] = d
        # patch-level F1 vs hand boxes (zenodo only)
        py, pp = [], []
        k = 0
        for bi in range(0, len(rs), 32):
            chunk = rs[bi:bi + 32]; pm = pmaps[k]; k += 1
            for j, r in enumerate(chunk):
                if r.get("patch") is not None:
                    t = np.asarray(r["patch"], np.float32).reshape(-1)
                    py.extend(t.tolist()); pp.extend(pm[j, :len(t)].numpy().tolist())
        if py and len(set(py)) == 2:
            res["F1_patch_vs_boxes"] = {"n_patches": len(py), "pos": int(sum(py)),
                                        "auroc": float(roc_auc_score(py, pp)),
                                        "ap": float(average_precision_score(py, pp))}
        return res, p

    def summary(res):
        aps = [v["ap"] for f, v in res.items() if f in FACTORS and "ap" in v]
        return float(np.mean(aps)) if aps else 0.0

    best, best_state, hist, t0 = -1, None, [], time.time()
    for ep in range(a.epochs):
        tot = 0.0
        for st in range(a.steps):
            rs = random.choices(tr, weights=weights, k=a.batch)
            X, V, y, m, P = batchify(rs, True)
            z, s, w = model(X, V)
            loss = masked_bce(z, y, m, pw)
            if a.patch_loss > 0:
                idx = [i for i, p in enumerate(P) if p is not None]
                if idx:
                    t = torch.tensor([sum(P[i], []) for i in idx], device=dev)
                    sp = s[idx, :t.shape[1], 0]
                    vp = V[idx, :t.shape[1]].float()
                    pl = torch.nn.functional.binary_cross_entropy_with_logits(
                        sp, t, reduction="none", pos_weight=torch.tensor(a.patch_pw, device=dev))
                    loss = loss + a.patch_loss * (pl * vp).sum() / vp.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        vres, _ = evaluate(va)
        sc = summary(vres)
        hist.append({"epoch": ep + 1, "loss": tot / a.steps, "val_mean_ap": sc})
        print(f"ep {ep+1} loss {tot/a.steps:.4f} val_mAP {sc:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if sc > best:
            best, best_state = sc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    vres, vp = evaluate(va)
    thr = {}
    for i, f in enumerate(FACTORS):
        yy = [r["y"][i] for r in va if r["m"][i]]
        pp = [float(vp[k, i]) for k, r in enumerate(va) if r["m"][i]]
        thr[f] = best_f1_threshold(yy, pp) if len(set(yy)) == 2 else 0.5
    tres, tp = evaluate(te)
    for i, f in enumerate(FACTORS):
        yy = [r["y"][i] for r in te if r["m"][i]]
        pp = [float(tp[k, i]) for k, r in enumerate(te) if r["m"][i]]
        if yy:
            t = thr[f]
            tpn = sum(1 for q, w in zip(yy, pp) if q > .5 and w >= t)
            fp = sum(1 for q, w in zip(yy, pp) if q <= .5 and w >= t)
            fn = sum(1 for q, w in zip(yy, pp) if q > .5 and w < t)
            tres[f].update({"threshold": t, "f1": 2 * tpn / (2 * tpn + fp + fn) if tpn else 0.0,
                            "precision": tpn / (tpn + fp) if tpn + fp else 0.0,
                            "recall": tpn / (tpn + fn) if tpn + fn else 0.0})
    # per-source test breakdown for F1
    tv = [r for r in te if r["source"] == "voxel51"]
    tz = [r for r in te if r["source"] == "zenodo"]
    sub = {"voxel51": evaluate(tv)[0] if tv else {}, "zenodo": evaluate(tz)[0] if tz else {}}
    run = a.name or time.strftime("head_%Y%m%d_%H%M")
    ck = TRAINING / "04-head/checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    import torch as _t
    _t.save({"state_dict": best_state, "config": vars(a), "factors": FACTORS, "thresholds": thr,
             "pool": POOL, "grid": GRID}, ck / f"{run}.pt")
    resd = TRAINING / "04-head/results"; resd.mkdir(parents=True, exist_ok=True)
    preds = [{"id": r["id"], "p": [round(float(v), 5) for v in tp[k]], "y": r["y"], "m": r["m"]}
             for k, r in enumerate(te) if r["source"] == "voxel51"]
    out = {"run": run, "config": vars(a), "best_val_mean_ap": best, "history": hist, "val": vres,
           "test_predictions_voxel51": preds,
           "test": tres, "test_by_source": sub, "thresholds": thr,
           "sizes": {"train": len(tr), "val": len(va), "test": len(te)},
           "pos_weight": pw.tolist(), "minutes": (time.time() - t0) / 60}
    json.dump(out, open(resd / f"{run}.json", "w", encoding="utf-8"), indent=1)
    print("TEST", json.dumps({f: {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}
                              for f, d in tres.items()}), flush=True)
    print(f"saved {ck / (run + '.pt')} and {resd / (run + '.json')}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("cache")
    c.add_argument("--features", default="/mnt/f/sightline_features/vjepa2_1_vitl_384")
    c.add_argument("--cache", default="~/sightline_cache/vjepa2_1_vitl_384_p2")
    c.add_argument("--labels", default="02-labeling/outputs/weak_labels_v2_vllm.jsonl")
    c.add_argument("--zenodo-train-stride", type=int, default=2)
    t = sp.add_parser("train")
    t.add_argument("--cache", default="~/sightline_cache/vjepa2_1_vitl_384_p2")
    t.add_argument("--labels", default="02-labeling/outputs/weak_labels_v2_vllm.jsonl")
    t.add_argument("--name", default="")
    t.add_argument("--gpu", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--ignore", default="F3,F4")
    t.add_argument("--val-frac", type=float, default=0.15)
    t.add_argument("--cv", type=int, default=0, help="k-fold over all Voxel51 clips (0 = official test split)")
    t.add_argument("--fold", type=int, default=0)
    t.add_argument("--use-zenodo", type=int, default=1)
    t.add_argument("--zenodo-share", type=float, default=0.3)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=0.05)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--steps", type=int, default=50)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--pw-cap", type=float, default=20.0)
    t.add_argument("--window-drop", type=float, default=0.25)
    t.add_argument("--token-drop", type=float, default=0.1)
    t.add_argument("--patch-loss", type=float, default=0.5)
    t.add_argument("--patch-pw", type=float, default=5.0)
    a = ap.parse_args(argv)
    {"cache": cmd_cache, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
