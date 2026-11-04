"""Circle quality on the Zenodo table-saw TEST split (3,591 images, never used for training).

Uses the saved V-JEPA 2.1 features from phase 03 (same encoder, same 384x672 squash), runs the
h3_wide head, turns the F1 map into a circle with the live pipeline's localize code, and compares
with the YOLO hand boxes:
  pointing  = circle centre lies inside a hand box
  box_hit   = share of hand boxes whose centre lies inside the circle
  iou       = circle vs best-matching box IoU (rasterised); also IoU vs the box-enclosing circle
Baseline: a circle of the same radius at the mean training-box centre (shows it is not trivial).

  python -m sightline_pipeline.eval_circles --out sightline_pipeline/results/circles_zenodo_test.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import localize
from .config import REPO, load_config, resolve_path

WORK = Path(os.environ.get("SIGHTLINE_WORK_DIR", REPO / "work"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(WORK / "features" / "vjepa2_1_vitl_384"))
    ap.add_argument("--out", default=str(REPO / "sightline_pipeline/results/circles_zenodo_test.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rel", default="0.4,0.55,0.7")
    a = ap.parse_args(argv)
    import torch
    from .model import training_modules
    cfg = load_config()
    ef, th = training_modules()
    ck = torch.load(resolve_path(cfg["model"]["head_checkpoint"]), map_location="cpu", weights_only=False)
    head = th.build_model(hidden=ck["config"]["hidden"], drop=ck["config"]["dropout"], nf=len(ck["factors"]))
    head.load_state_dict(ck["state_dict"]); head.eval()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); head.to(dev)
    feat = Path(a.features)
    rows = {}
    for p in sorted(feat.glob("index_r*.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r["source"] == "zenodo":
                rows[r["id"]] = r
    man = {}
    for l in open(REPO / "training/data/manifest.jsonl", encoding="utf-8"):
        m = json.loads(l)
        if m["source"] == "zenodo_tablesaw":
            man[m["id"]] = m
    test = sorted(i for i, r in rows.items() if r["split"] == "test")
    train = [i for i, r in rows.items() if r["split"] == "train"]
    # baseline centre: mean hand-box centre over train labels
    cs = []
    for i in train[::10]:
        lp = man[i].get("label_path")
        for b in th.read_yolo(str(REPO / "training" / lp) if lp else None):
            cs.append(b[:2])
    base_c = np.mean(cs, 0) if cs else np.array([0.5, 0.5])
    if a.limit:
        test = test[: a.limit]
    W, H = 1920, 1080
    rels = [float(x) for x in a.rel.split(",")]
    modes = ["evidence", "attention", "attention_x_evidence"]
    lc = cfg["localization"]
    acc = {(m, r): {"point": [], "hit": [], "iou": [], "iou_enc": [], "radius": []} for m in modes for r in rels}
    base = {"point": [], "hit": [], "iou": []}
    t0, n_img, gh, gw, pool = time.time(), 0, 12, 21, 2
    for k, iid in enumerate(test):
        lp = man[iid].get("label_path")
        boxes = [localize.yolo_to_xyxy(b, W, H) for b in th.read_yolo(str(REPO / "training" / lp) if lp else None)]
        if not boxes:
            continue
        z = np.load(feat / "zenodo" / f"{rows[iid]['key']}.npz")
        g = z["grid"][0].astype(np.float32)
        g = g.reshape(gh, pool, gw, pool, -1).mean((1, 3)).reshape(gh * gw, -1)
        with torch.no_grad():
            X = torch.from_numpy(g)[None].to(dev)
            _, s, w = head(X, torch.ones(X.shape[:2], dtype=torch.bool, device=dev))
            ev = torch.sigmoid(s[0, :, 0]).cpu().numpy().reshape(gh, gw)
            at = w[0, :, 0].cpu().numpy().reshape(gh, gw)
        maps = {"evidence": ev, "attention": at, "attention_x_evidence": ev * at}
        n_img += 1
        for m in modes:
            for r in rels:
                cc = localize.circles_from_map(maps[m], W, H, r, lc["min_radius_frac"], lc["max_radius_frac"],
                                               lc["radius_scale"], 1)
                if not cc:
                    continue
                c = cc[0]
                d = acc[(m, r)]
                d["point"].append(any(x1 <= c.cx <= x2 and y1 <= c.cy <= y2 for x1, y1, x2, y2 in boxes))
                d["hit"].append(np.mean([localize.contains(c, (x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]))
                d["iou"].append(max(localize.circle_box_iou(c, b, 128) for b in boxes))
                encs = [localize.box_to_circle(*b, W, H) for b in boxes]
                d["iou_enc"].append(max(localize.circle_iou(c, e) for e in encs))
                d["radius"].append(c.r / math.hypot(W, H))
        # baseline with the evidence-map radius at rel 0.55
        rb = acc[("evidence", rels[min(1, len(rels) - 1)])]["radius"]
        rr = (rb[-1] if rb else 0.1) * math.hypot(W, H)
        bc = localize.Circle(base_c[0] * W, base_c[1] * H, rr, W, H)
        base["point"].append(any(x1 <= bc.cx <= x2 and y1 <= bc.cy <= y2 for x1, y1, x2, y2 in boxes))
        base["hit"].append(np.mean([localize.contains(bc, (x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]))
        base["iou"].append(max(localize.circle_box_iou(bc, b, 128) for b in boxes))
        if k % 500 == 0:
            print(f"{k}/{len(test)} {time.time() - t0:.0f}s", flush=True)
    res = {"split": "zenodo test", "images_with_hands": n_img, "frame": [W, H], "head": Path(str(cfg["model"]["head_checkpoint"])).stem,
           "results": [], "baseline_mean_train_centre": {k: round(float(np.mean(v)), 4) for k, v in base.items()},
           "seconds": round(time.time() - t0, 1)}
    for (m, r), d in acc.items():
        res["results"].append({"map": m, "rel_threshold": r, "n": len(d["point"]),
                               **{k: round(float(np.mean(v)), 4) for k, v in d.items() if v}})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
