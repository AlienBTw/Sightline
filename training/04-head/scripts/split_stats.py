"""Recompute the data-split statistics used by 04-head (read-only; no training).

Uses the same assign_split / cv_fold / voxel_targets as train_head.py, the feature index
from 03 and the manifest/weak labels from 01/02. Prints JSON to stdout.
Usage (from training/):  python 04-head/scripts/split_stats.py [--features F:/sightline_features/vjepa2_1_vitl_384]
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

T = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(T / "04-head"))
from train_head import FACTORS, assign_split, cv_fold, voxel_targets  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="F:/sightline_features/vjepa2_1_vitl_384")
ap.add_argument("--labels", default="02-labeling/outputs/weak_labels_v2_vllm.jsonl")
ap.add_argument("--stride", type=int, default=2)
a = ap.parse_args()

man = [json.loads(l) for l in open(T / "data/manifest.jsonl", encoding="utf-8")]
vm = [m for m in man if m["source"] == "voxel51_safe_unsafe"]
lab = {r["clip_id"]: r for r in map(json.loads, open(T / a.labels, encoding="utf-8"))}
idx = {}
for p in sorted(Path(a.features).glob("index_r*.jsonl")):
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); idx[r["id"]] = r
out = {}
# ---- raw manifest / duplicates
out["raw_files"] = len(vm); out["raw_split"] = Counter(m["split"] for m in vm)
out["raw_class_split"] = {f"{m['split']}|{m['source_label']}": 0 for m in vm}
for m in vm: out["raw_class_split"][f"{m['split']}|{m['source_label']}"] += 1
by = defaultdict(list)
for m in vm: by[m["sha256"]].append(m)
groups = [g for g in by.values() if len(g) > 1]
out["unique"] = len(by); out["dup_groups"] = len(groups); out["dup_files"] = sum(len(g) for g in groups)
out["dup_group_sizes"] = Counter(len(g) for g in groups)
out["dup_groups_conflicting_class"] = sum(1 for g in groups if len({m["source_label"] for m in g}) > 1)
cross = []
for g in groups:
    sp = {m["split"] for m in g}
    kept = [m["id"] for m in g if m["id"] in lab]
    if len(sp) > 1:
        cross.append({"files": [f"{m['id']}({m['split']},{m['source_label']})" for m in g],
                      "kept": kept, "kept_split": [lab[k]["split"] for k in kept]})
out["cross_split_groups"] = cross
out["within_split_groups"] = Counter(next(iter({m["split"] for m in g})) for g in groups if len({m["split"] for m in g}) == 1)
# ---- unique clips (label file rows)
V = [r for r in lab.values() if r["source"] == "voxel51_safe_unsafe"]
out["label_rows"] = len(V); out["unique_split"] = Counter(r["split"] for r in V)
def classes(r):
    return sorted({r["source_label"], *[l for l in (man_by[r["clip_id"]].get("duplicate_labels") or [])]})
man_by = {m["id"]: m for m in vm}
def item(r): return {"id": r["clip_id"], "source": "voxel51", "split": r["split"]}
def win(r): return idx.get(r["clip_id"], {}).get("windows")
out["missing_features"] = [r["clip_id"] for r in V if r["clip_id"] not in idx]
def stats(rows):
    s = {"clips": len(rows), "windows": sum(win(r) or 0 for r in rows), "classes": Counter(), "multi_class_clips": 0,
         "factors": {}}
    for r in rows:
        cs = classes(r); s["multi_class_clips"] += len(cs) > 1
        for c in cs: s["classes"][c] += 1
    for i, f in enumerate(FACTORS):
        if f in ("F3", "F4"):
            continue
        p = n = ig = 0
        for r in rows:
            y, m = voxel_targets(r)
            if not m[i]: ig += 1
            elif y[i] > .5: p += 1
            else: n += 1
        s["factors"][f] = f"{p}/{n}/{ig}"
    return s
# official protocol (seed 0, 1, 2 differ only in val)
for seed in (0, 1, 2):
    sets = defaultdict(list)
    for r in V: sets[assign_split(item(r), 0.15, seed)].append(r)
    out[f"official_seed{seed}"] = {k: stats(v) for k, v in sets.items()}
# CV
folds = defaultdict(list)
for r in V: folds[cv_fold(r["clip_id"], 5)].append(r)
out["cv_folds"] = {k: stats(folds[k]) for k in sorted(folds)}
out["cv_fold_official_split"] = {k: Counter(r["split"] for r in folds[k]) for k in sorted(folds)}
cvv = {}
for f in range(5):
    sets = Counter(assign_split(item(r), 0.15, 0, 5, f) for r in V)
    cvv[f] = dict(sets)
out["cv_train_val_test_voxel"] = cvv
out["all_unique"] = stats(V)
# ---- zenodo
Z = [r for r in idx.values() if r["source"] == "zenodo"]
zm = {m["id"]: m for m in man if m["source"] == "zenodo_tablesaw"}
out["zenodo_extracted"] = Counter(r["split"] for r in Z)
def kept(r):
    if r["split"] != "train" or a.stride <= 1: return True
    t = r["id"].rsplit("_", 1)[-1]; n = int(t) if t.isdigit() else 0
    return n % a.stride == 0
zk = [r for r in Z if kept(r)]
out["zenodo_used"] = Counter(r["split"] for r in zk)
out["zenodo_handfree_used"] = Counter(r["split"] for r in zk if not (zm.get(r["id"], {}).get("n_hand_boxes") or 0))
out["zenodo_handfree_all"] = Counter(r["split"] for r in Z if not (zm.get(r["id"], {}).get("n_hand_boxes") or 0))
out["zenodo_windows"] = Counter(r["split"] for r in zk for _ in range(r["windows"]))
print(json.dumps(out, indent=1, default=str, ensure_ascii=False))
