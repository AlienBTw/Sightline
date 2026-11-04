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
    F7, F8, F9          : dataset class hint where the class defines it;
                            every other clip is an explicit negative (not ignored)
  Zenodo table saw (images): F1 = hand box present; F6-F10 explicit negatives
                            (a saw frame is not a fall, a load, a vehicle path,
                            an open guard, or fire). F2-F5 stay ignored.
                            --zenodo-f9-frac can drop F9 on train rows so easy saw
                            frames do not drown the closed-panel contrast.
    Hand boxes also give a patch-level F1 target (aux loss on s_F1).
  SHWD (opt-in, --f2-box-source shwd): F2 patch target = bare-head boxes after the
    same 384x672 letterbox used at extraction; hat boxes are explicit negatives.
    Clip-level F2 is the max over patches. Every other factor is masked.
    --drop-glimmer-f2 clears Glimmer's F2 label on Voxel51 training rows.
  PPE video (opt-in, --ppe-labels / --ppe-cache): real-video F2 and F5 labels.
    Pexels clips supervise only F2/F5 (other factors masked). demo_eval is held out.
    --init-checkpoint fine-tunes from a saved head. --freeze-except F2,F5 trains
    only those factors' attention, evidence and bias. Omit it and pass a lower
    --lr to fine-tune every parameter. h3_wide needs --hidden 512 --dropout 0.2.
    --ppe-bg-neg (default 0) labels patches farther than --ppe-bg-margin (default
    1 patch, Chebyshev) from every F2 or F5 box in that window as negatives.
    A window is included only when it has at least one box for that factor, on
    Pexels rows and on Voxel51 rows loaded with --ppe-override-voxel.
    --ppe-window-labels adds BCE on the max patch evidence of each window that
    has its own non-null F2/F5 label (--ppe-window-weight, default 0.5), so a
    frame with no bare head inside a present clip is not forced positive.
    SHWD box targets need the cache that contains shwd rows
    (``<SIGHTLINE_WORK_DIR>/cache/vjepa2_1_vitl_384_p2_shwd``); that cache combines with
    --ppe-cache.

Subcommands:
  cache  - pool saved grids 2x2 (24x42 -> 12x21) into one fp16 memmap on WSL ext4
  train  - train one config, write checkpoint (ignored) + results JSON (committed)
  eval   - score an existing checkpoint on the same splits, write results/<name>_eval.json
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
REPO = TRAINING.parent
WORK = Path(os.environ.get("SIGHTLINE_WORK_DIR", REPO / "work"))
DEFAULT_FEATURES = str(WORK / "features" / "vjepa2_1_vitl_384")
DEFAULT_CACHE = str(WORK / "cache" / "vjepa2_1_vitl_384_p2")
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
            if lab == "present":
                y[i], m[i] = 1.0, 1.0
            else:
                # absent, or no hint: this scene is not that hazard
                m[i] = 1.0
            continue
        lab = (row.get("factors") or {}).get(f, {}).get("label")
        if lab == "present":
            y[i], m[i] = 1.0, 1.0
        elif lab == "absent":
            m[i] = 1.0
    return y, m


# A table-saw frame cannot be a fall, an unstable load, a vehicle path, an open guard, or fire.
ZENODO_NEGATIVE = ("F6", "F7", "F8", "F9", "F10")


def zenodo_targets(n_boxes: int):
    y, m = [0.0] * 10, [0.0] * 10
    m[0] = 1.0
    y[0] = 1.0 if n_boxes > 0 else 0.0
    for f in ZENODO_NEGATIVE:
        m[FACTORS.index(f)] = 1.0
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
    """Patch grid (gh x gw) target: 1 where a box covers >= min_frac of the cell.
    Boxes are (cx, cy, w, h) in normalized input coordinates. Zenodo hands are already
    in that space (the extractor squashes the full frame). SHWD boxes are mapped
    through the letterbox first."""
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


def xyxy_to_cxcywh(boxes):
    out = []
    for x0, y0, x1, y1 in boxes:
        out.append(((float(x0) + float(x1)) / 2, (float(y0) + float(y1)) / 2,
                    max(0.0, float(x1) - float(x0)), max(0.0, float(y1) - float(y0))))
    return out


_EF = None


def feature_mod():
    """Load extract_features.py by path so this module stays importable from the pipeline."""
    global _EF
    if _EF is None:
        import importlib.util
        import sys
        path = TRAINING / "03-features" / "extract_features.py"
        name = "sightline_extract_features"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _EF = mod
    return _EF


def shwd_patch_targets(bare_boxes, hat_boxes, width: int, height: int, gh: int, gw: int,
                       min_frac: float = 0.30, hat_weight: float = 2.0):
    """F2 patch target on the pooled grid.

    Bare-head boxes are positives. Patches covered by a hat box are explicit negatives
    (hat wins on overlap) and carry ``hat_weight``. Clip label is the max over patches.
    Returns (target[gh][gw], weight[gh][gw], clip).
    """
    mapped_bare = feature_mod().map_xyxy_letterbox(bare_boxes, int(width), int(height))
    mapped_hat = feature_mod().map_xyxy_letterbox(hat_boxes, int(width), int(height))
    bare_m = box_patch_mask(xyxy_to_cxcywh(mapped_bare), gh, gw, min_frac)
    hat_m = box_patch_mask(xyxy_to_cxcywh(mapped_hat), gh, gw, min_frac)
    target = [[0.0] * gw for _ in range(gh)]
    weight = [[1.0] * gw for _ in range(gh)]
    for r in range(gh):
        for c in range(gw):
            if hat_m[r][c] >= 1.0:
                target[r][c] = 0.0
                weight[r][c] = float(hat_weight)
            elif bare_m[r][c] >= 1.0:
                target[r][c] = 1.0
    clip = 1.0 if any(v >= 1.0 for row in target for v in row) else 0.0
    return target, weight, clip


def shwd_targets(clip: float):
    """SHWD supervises F2 only. Every other factor stays masked."""
    y, m = [0.0] * 10, [0.0] * 10
    i = FACTORS.index("F2")
    y[i] = 1.0 if clip > 0.5 else 0.0
    m[i] = 1.0
    return y, m


def mask_glimmer_f2(rows) -> int:
    """Clear Glimmer's F2 label on Voxel51 training rows. Val and test stay labeled."""
    i = FACTORS.index("F2")
    n = 0
    for r in rows:
        if r.get("source") != "voxel51" or r.get("set") != "train":
            continue
        if r["m"][i]:
            n += 1
        r["m"][i] = 0.0
        r["y"][i] = 0.0
    return n


def ppe_binary(value):
    """1, 0, or None. None (and unknown strings) means the factor is masked."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ("", "null", "none", "unknown", "uncertain"):
            return None
        if key in ("1", "present", "true", "yes"):
            return 1.0
        if key in ("0", "absent", "false", "no"):
            return 0.0
        raise SystemExit(f"ppe label value {value!r} is not 1, 0, or null")
    if isinstance(value, (int, float)):
        if float(value) in (0.0, 1.0):
            return float(value)
        raise SystemExit(f"ppe label value {value!r} is not 1, 0, or null")
    raise SystemExit(f"ppe label value {value!r} is not 1, 0, or null")


def set_f2_f5(y, m, clip):
    """Replace F2 and F5. A null clip value masks that factor. Other factors stay."""
    y, m = list(y), list(m)
    clip = clip or {}
    for f in ("F2", "F5"):
        i = FACTORS.index(f)
        v = ppe_binary(clip.get(f))
        if v is None:
            y[i], m[i] = 0.0, 0.0
        else:
            y[i], m[i] = v, 1.0
    return y, m


def clear_f2_f5(row):
    for f in ("F2", "F5"):
        i = FACTORS.index(f)
        row["y"][i] = 0.0
        row["m"][i] = 0.0


def ppe_metric_source(row, label=None) -> str:
    src = str((label or {}).get("source") or "")
    if src in ("voxel51", "voxel51_safe_unsafe"):
        return "voxel51"
    if src in ("pexels", "ppe_video"):
        return "pexels"
    item = str((row or {}).get("id") or "")
    if item.startswith("voxel51/"):
        return "voxel51"
    return "pexels"


def load_ppe_label_file(path) -> dict:
    mp = resolve_repo_file(str(path))
    if not mp.is_file():
        alt = TRAINING.parent / str(path)
        mp = alt if alt.is_file() else mp
    if not mp.is_file():
        raise SystemExit(f"ppe labels not found: {path}")
    out = {}
    with open(mp, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row:
                raise SystemExit("ppe label row is missing id")
            out[row["id"]] = row
    return out


def align_ppe_windows(n: int, starts, label_windows):
    """Feature windows -> label window records. Match on start frame, else window index."""
    by_start, by_w = {}, {}
    for rec in label_windows or []:
        if not isinstance(rec, dict):
            continue
        if rec.get("start") is not None:
            by_start[int(rec["start"])] = rec
        if rec.get("w") is not None:
            by_w[int(rec["w"])] = rec
    out = []
    for i in range(int(n)):
        rec = None
        if starts and i < len(starts) and int(starts[i]) in by_start:
            rec = by_start[int(starts[i])]
        elif i in by_w:
            rec = by_w[i]
        out.append(rec)
    return out


def _patches_farther_than(boxes_xyxy, gh: int, gw: int, margin: float):
    """Chebyshev distance, in patches, from each cell to the nearest box.

    Distance 0 is any positive overlap. Returns None when no box covers a cell,
    so the caller does not mark the whole frame negative. A cell is background
    when its distance is greater than ``margin`` (1 leaves the 8-neighbour ring
    unlabeled).
    """
    hits = []
    for raw in boxes_xyxy or []:
        if raw is None or len(raw) < 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in raw[:4])
        xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
        ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
        if xb <= xa or yb <= ya:
            continue
        for r in range(gh):
            cy0, cy1 = r / gh, (r + 1) / gh
            if min(yb, cy1) - max(ya, cy0) <= 0:
                continue
            for c in range(gw):
                cx0, cx1 = c / gw, (c + 1) / gw
                if min(xb, cx1) - max(xa, cx0) > 0:
                    hits.append((r, c))
    if not hits:
        return None
    seeds = set(hits)
    far = [[False] * gw for _ in range(gh)]
    for r in range(gh):
        for c in range(gw):
            dist = min(max(abs(r - sr), abs(c - sc)) for sr, sc in seeds)
            far[r][c] = dist > margin
    return far


def ppe_patch_from_boxes(pos_boxes, neg_boxes, width: int, height: int, gh: int, gw: int,
                         min_frac: float = 0.30, neg_weight: float = 2.0,
                         bg_neg: float = 0.0, bg_margin: float = 1.0):
    """Flat patch target and weight for one factor on one window.

    Boxes are xyxy normalised to the original frame. Letterbox matches
    extract_features (384x672). Negative regions win on overlap and use
    ``neg_weight``. Cells outside both regions are ignored (weight 0).
    When ``bg_neg`` > 0 and this window has at least one box, cells farther
    than ``bg_margin`` patches from every box are negatives with weight
    ``bg_neg``. ``bg_neg`` 0 is the previous behaviour. Returns None when no
    cell is supervised.
    """
    pos_boxes = list(pos_boxes or [])
    neg_boxes = list(neg_boxes or [])
    if not pos_boxes and not neg_boxes:
        return None
    if float(bg_neg) < 0 or float(bg_margin) < 0:
        raise SystemExit("--ppe-bg-neg and --ppe-bg-margin must be >= 0")
    mapped_pos = feature_mod().map_xyxy_letterbox(pos_boxes, int(width), int(height))
    mapped_neg = feature_mod().map_xyxy_letterbox(neg_boxes, int(width), int(height))
    pos_m = box_patch_mask(xyxy_to_cxcywh(mapped_pos), gh, gw, min_frac)
    neg_m = box_patch_mask(xyxy_to_cxcywh(mapped_neg), gh, gw, min_frac)
    far = None
    if float(bg_neg) > 0:
        far = _patches_farther_than(list(mapped_pos) + list(mapped_neg), gh, gw, float(bg_margin))
    target, weight = [], []
    supervised = False
    for r in range(gh):
        for c in range(gw):
            if neg_m[r][c] >= 1.0:
                target.append(0.0)
                weight.append(float(neg_weight))
                supervised = True
            elif pos_m[r][c] >= 1.0:
                target.append(1.0)
                weight.append(1.0)
                supervised = True
            elif far is not None and far[r][c]:
                target.append(0.0)
                weight.append(float(bg_neg))
                supervised = True
            else:
                target.append(0.0)
                weight.append(0.0)
    if not supervised:
        return None
    return target, weight


def _media_size(label, row):
    for src in (label or {}, row or {}):
        w = src.get("width") or src.get("frame_width")
        h = src.get("height") or src.get("frame_height")
        if w and h:
            return int(w), int(h)
    return None


def probe_media_size(path: str):
    """Width and height of a video or image. None when the file cannot be read."""
    cands = []
    raw = Path(os.path.expanduser(str(path)))
    cands.append(raw)
    if not raw.is_absolute():
        cands.append(TRAINING / path)
        cands.append(TRAINING.parent / path)
        cands.append(Path.cwd() / path)
    found = next((c for c in cands if c.is_file()), None)
    if found is None:
        return None
    import cv2
    cap = cv2.VideoCapture(str(found))
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if w >= 1 and h >= 1:
        return int(w), int(h)
    img = cv2.imread(str(found))
    if img is None:
        return None
    return int(img.shape[1]), int(img.shape[0])


def attach_ppe_windows(row, label, gh, gw, min_frac, neg_weight, bg_neg=0.0, bg_margin=1.0):
    """Per-window F2/F5 labels and patch targets. Windows with a null label are not supervised."""
    n = int(row.get("n") or 0)
    windows = label.get("windows") if isinstance(label, dict) else None
    if not windows:
        row["ppe_windows"] = [None] * n
        return
    aligned = align_ppe_windows(n, row.get("starts"), windows)
    needs_size = False
    for rec in aligned:
        if not rec:
            continue
        boxes = rec.get("boxes") or {}
        for f, pos_k, neg_k in (("F2", "F2_pos", "F2_neg"), ("F5", "F5_pos", "F5_neg")):
            if ppe_binary(rec.get(f)) is None:
                continue
            if boxes.get(pos_k) or boxes.get(neg_k):
                needs_size = True
    size = _media_size(label, row)
    if needs_size and size is None:
        probed = None
        for src in (label, row):
            if src and src.get("path"):
                probed = probe_media_size(src["path"])
                if probed:
                    break
        size = probed
    if needs_size and size is None:
        raise SystemExit(
            f"ppe row {row.get('id')} has window boxes but no frame size. "
            "Add width and height to the label row, or a readable path."
        )
    built = []
    for rec in aligned:
        if rec is None:
            built.append(None)
            continue
        entry = {"F2": ppe_binary(rec.get("F2")), "F5": ppe_binary(rec.get("F5")), "patch": {}}
        boxes = rec.get("boxes") or {}
        if size is not None:
            for f, pos_k, neg_k in (("F2", "F2_pos", "F2_neg"), ("F5", "F5_pos", "F5_neg")):
                if entry[f] is None:
                    continue
                packed = ppe_patch_from_boxes(
                    boxes.get(pos_k), boxes.get(neg_k), size[0], size[1], gh, gw, min_frac, neg_weight,
                    bg_neg, bg_margin)
                if packed is not None:
                    entry["patch"][f] = {"y": packed[0], "w": packed[1]}
        built.append(entry)
    row["ppe_windows"] = built


def apply_pexels_row(row, label, gh, gw, min_frac, neg_weight, bg_neg=0.0, bg_margin=1.0):
    """Pexels clip: MIL labels from clip.F2/F5 only. Every other factor is masked."""
    y, m = set_f2_f5([0.0] * 10, [0.0] * 10, label.get("clip") or {})
    row["y"], row["m"] = y, m
    if label.get("split"):
        row["split"] = label["split"]
    row["ppe_item"] = True
    row["ppe_labeled"] = True
    row["ppe_source"] = ppe_metric_source(row, label)
    attach_ppe_windows(row, label, gh, gw, min_frac, neg_weight, bg_neg, bg_margin)
    return row


def apply_voxel_ppe_row(row, label, gh, gw, min_frac, neg_weight, bg_neg=0.0, bg_margin=1.0):
    """Replace Glimmer F2/F5 on a Voxel51 row. F1 and F6-F10 stay as they were."""
    y, m = set_f2_f5(row["y"], row["m"], label.get("clip") or {})
    row["y"], row["m"] = y, m
    row["ppe_labeled"] = True
    row["ppe_source"] = "voxel51"
    attach_ppe_windows(row, label, gh, gw, min_frac, neg_weight, bg_neg, bg_margin)
    return row


def apply_voxel_ppe_policy(rows, labels, override: bool, mask: bool, gh, gw, min_frac, neg_weight,
                           bg_neg=0.0, bg_margin=1.0):
    """PPE rows replace Voxel51 F2/F5 when ``override``. Clips with no PPE row are masked when ``mask``."""
    labels = labels or {}
    cleared = overridden = 0
    for r in rows:
        if r.get("source") != "voxel51":
            continue
        lab = labels.get(r["id"])
        if lab and override:
            apply_voxel_ppe_row(r, lab, gh, gw, min_frac, neg_weight, bg_neg, bg_margin)
            overridden += 1
        elif lab is None and mask:
            i2, i5 = FACTORS.index("F2"), FACTORS.index("F5")
            if r["m"][i2] or r["m"][i5]:
                cleared += 1
            clear_f2_f5(r)
    return {"cleared": cleared, "overridden": overridden}


def parse_factor_list(text: str) -> list[str]:
    out = []
    for part in (text or "").split(","):
        f = part.strip()
        if not f:
            continue
        if f not in FACTORS:
            raise SystemExit(f"unknown factor {f!r} in --freeze-except")
        out.append(f)
    return out


def configure_freeze(model, factors: list[str]):
    """Freeze shared trunk and every factor row except ``factors``.

    Empty ``factors`` leaves the whole model trainable and returns None.
    Otherwise returns the output-row indexes that are allowed to change
    (attention, evidence, and both biases).
    """
    if not factors:
        for p in model.parameters():
            p.requires_grad = True
        return None
    keep = [FACTORS.index(f) for f in factors]
    for p in model.norm.parameters():
        p.requires_grad = False
    for p in model.proj.parameters():
        p.requires_grad = False
    model.att.weight.requires_grad = True
    model.att.bias.requires_grad = True
    model.val.weight.requires_grad = True
    model.val.bias.requires_grad = True
    return keep


def mask_factor_grads(model, keep: list[int]):
    """Zero gradients on factor rows that are not being fine-tuned."""
    keep_set = set(keep)
    for layer in (model.att, model.val):
        if layer.weight.grad is not None:
            for i in range(layer.weight.shape[0]):
                if i not in keep_set:
                    layer.weight.grad[i].zero_()
        if layer.bias is not None and layer.bias.grad is not None:
            for i in range(layer.bias.shape[0]):
                if i not in keep_set:
                    layer.bias.grad[i].zero_()


def snapshot_factor_rows(model):
    return {
        "att_w": model.att.weight.detach().clone(),
        "att_b": model.att.bias.detach().clone(),
        "val_w": model.val.weight.detach().clone(),
        "val_b": model.val.bias.detach().clone(),
    }


def restore_factor_rows(model, snap, keep: list[int]):
    """Put frozen factor rows back. AdamW weight decay would otherwise move them."""
    keep_set = set(keep)
    import torch
    with torch.no_grad():
        pairs = (
            (snap["att_w"], model.att.weight),
            (snap["att_b"], model.att.bias),
            (snap["val_w"], model.val.weight),
            (snap["val_b"], model.val.bias),
        )
        for src, dest in pairs:
            for i in range(dest.shape[0]):
                if i not in keep_set:
                    dest[i].copy_(src[i])


def guarded_output(path: Path) -> Path:
    """Refuse to replace the live h3_wide checkpoint or its results JSON."""
    name = path.name
    if name in ("h3_wide.pt", "h3_wide.json"):
        raise SystemExit(f"refusing to overwrite {path}")
    return path


def _shwd_geometry(src) -> bool:
    return bool(src) and src.get("bare_boxes") is not None and src.get("hat_boxes") is not None and src.get("width") and src.get("height")


def apply_train_sources(meta, use_zenodo=1, f2_box_source="", shwd_rows=None,
                        gh=None, gw=None, f2_min_frac=0.30, f2_hatneg_weight=2.0,
                        cache_label=""):
    """Attach optional SHWD box targets, or drop those rows.

    Defaults keep the voxel + zenodo setup: Zenodo stays, SHWD is removed, and no
    Glimmer label is touched (call ``mask_glimmer_f2`` separately).
    Rows already stored in the feature cache keep their boxes when the manifest
    row is missing that geometry. PPE rows in the same meta are left in place.
    """
    gh = GRID[0] // POOL if gh is None else gh
    gw = GRID[1] // POOL if gw is None else gw
    if not use_zenodo:
        meta = [r for r in meta if r["source"] != "zenodo"]
    if f2_box_source == "shwd":
        n_shwd = sum(1 for r in meta if r.get("source") == "shwd")
        if n_shwd == 0:
            where = f" ({cache_label})" if cache_label else ""
            raise SystemExit(
                f"--f2-box-source shwd but the cache has no shwd rows{where}. "
                "Extract SHWD features and rebuild the cache (use a new --cache path "
                "such as <SIGHTLINE_WORK_DIR>/cache/vjepa2_1_vitl_384_p2_shwd)."
            )
        by_id = {m["id"]: m for m in (shwd_rows or [])}
        for r in meta:
            if r.get("source") != "shwd":
                continue
            src = by_id.get(r["id"], r)
            if not _shwd_geometry(src) and _shwd_geometry(r):
                src = r
            if not _shwd_geometry(src):
                raise SystemExit(f"shwd row {r['id']} has no boxes; pass --shwd-manifest")
            target, weight, clip = shwd_patch_targets(
                src["bare_boxes"], src["hat_boxes"], int(src["width"]), int(src["height"]),
                gh, gw, f2_min_frac, f2_hatneg_weight)
            r["y"], r["m"] = shwd_targets(clip)
            r["patch"] = target
            r["patch_weight"] = weight
            r["patch_factor"] = 1
    elif f2_box_source:
        raise SystemExit(f"unknown --f2-box-source {f2_box_source!r} (expected '' or 'shwd')")
    else:
        meta = [r for r in meta if r.get("source") != "shwd"]
    return meta


def source_weights(rows, zenodo_share: float, shwd_weight: float = 1.0, ppe_share: float = 0.0) -> dict:
    """Per-source sampling weights.

    Zenodo is scaled so it is ``zenodo_share`` of the voxel+zenodo mass (the historical
    formula, counting Zenodo rows only). SHWD, when present, is an extra source at
    ``shwd_weight`` on the same scale as a voxel row's base weight of 1.
    PPE clips (``ppe_item``) are then scaled so they are ``ppe_share`` of the batch.
    With no PPE rows the returned voxel and zenodo weights match the historical formula.
    """
    nv = sum(1 for r in rows if r["source"] == "voxel51")
    nz = sum(1 for r in rows if r["source"] == "zenodo")
    ns = sum(1 for r in rows if r["source"] == "shwd")
    n_ppe = sum(1 for r in rows if r.get("ppe_item") or r.get("source") in ("ppe_video", "pexels"))
    src_w = {"voxel51": 1.0, "zenodo": 1.0}
    if nz:
        src_w["zenodo"] = zenodo_share * nv / max(1, nz) / max(1e-6, 1 - zenodo_share)
    if ns:
        src_w["shwd"] = float(shwd_weight)
    if n_ppe:
        mass = nv * src_w["voxel51"] + nz * src_w["zenodo"] + (ns * float(shwd_weight) if ns else 0.0)
        share = min(max(float(ppe_share), 0.0), 0.999)
        if mass <= 0:
            src_w["ppe_video"] = 1.0
        elif share <= 0:
            src_w["ppe_video"] = 0.0
        else:
            src_w["ppe_video"] = share * mass / (n_ppe * (1.0 - share))
        src_w["pexels"] = src_w["ppe_video"]
    return src_w


def cv_fold(item_id: str, cv: int) -> int:
    import hashlib
    return int(hashlib.md5(f"cv:{item_id}".encode()).hexdigest()[:8], 16) % cv


def assign_split(item: dict, val_frac: float, seed: int, cv: int = 0, fold: int = 0) -> str:
    """train/val/test. Voxel51: keep the dataset test split (or, with cv>0, use hash fold
    `fold` of all 658 unique clips as test); carve val from the rest by a stable hash.
    Zenodo and SHWD: train/validation/test as written in the manifest. SHWD's test split
    is the official test.txt and is never carved into train or val.
    PPE video (pexels): the split is taken from the PPE label file. demo_eval is held
    out of train, val and test, including under cross-validation."""
    s = item["split"]
    if item["source"] in ("zenodo", "shwd"):
        return {"train": "train", "validation": "val", "test": "test"}.get(s, "skip")
    if item["source"] in ("ppe_video", "pexels"):
        return {"train": "train", "validation": "val", "val": "val", "test": "test"}.get(s, "skip")
    if cv > 0:
        if cv_fold(item["id"], cv) == fold:
            return "test"
    elif s == "test":
        return "test"
    import hashlib
    h = int(hashlib.md5(f"{seed}:{item['id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


CLOSED_PANEL_LABEL = "Closed Panel Cover"
HARD_NEG_BOOST = 5.0


def zenodo_f9_keep(row_id: str, frac: float, seed: int = 0) -> bool:
    """Whether this Zenodo training row still supervises F9. 1 keeps all, 0 keeps none."""
    if frac >= 1:
        return True
    if frac <= 0:
        return False
    import hashlib
    h = int(hashlib.md5(f"f9:{seed}:{row_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def mask_zenodo_f9(rows, frac: float, seed: int = 0) -> int:
    """Clear F9 on Zenodo training rows outside the kept fraction. Val and test stay labeled."""
    i = FACTORS.index("F9")
    n = 0
    for r in rows:
        if r.get("source") != "zenodo" or r.get("set") != "train":
            continue
        if zenodo_f9_keep(r["id"], frac, seed):
            continue
        if r["m"][i]:
            r["m"][i] = 0.0
            n += 1
    return n


def sampling_weights(rows, src_w, scores_by_id=None, factor="F9", min_score=0.8, boost=HARD_NEG_BOOST):
    """Source weights, with a boost for hard negatives of `factor` and for closed-panel clips.

    A Voxel51 clip is boosted when it is labeled absent on `factor` and the reference
    head scores it at least `min_score`, or when its dataset class is Closed Panel Cover.
    """
    i = FACTORS.index(factor)
    out = []
    for r in rows:
        w = float(src_w[r["source"]])
        if scores_by_id is not None and r.get("source") == "voxel51":
            score = scores_by_id.get(r["id"])
            labeled_neg = r["m"][i] > 0 and r["y"][i] <= 0.5
            hard = labeled_neg and score is not None and score >= min_score
            closed = (r.get("source_label") or "") == CLOSED_PANEL_LABEL
            if hard or closed:
                w *= boost
        out.append(w)
    return out


def resolve_repo_file(path: str) -> Path:
    p = Path(os.path.expanduser(path))
    if p.is_file():
        return p
    q = TRAINING / path
    return q if q.is_file() else p


def resolve_ckpt(path: str, kind: str = "hard-neg checkpoint") -> Path:
    p = Path(os.path.expanduser(path))
    if p.is_file():
        return p
    for base in (TRAINING, TRAINING.parent, Path.cwd()):
        cand = base / path
        if cand.is_file():
            return cand
    raise SystemExit(f"{kind} not found: {path}")


def pos_weights(ys, ms, cap: float = 20.0):
    """Per-factor neg/pos ratio over labeled entries, clamped to [1, cap]."""
    out = []
    for i in range(len(FACTORS)):
        pos = sum(1 for y, m in zip(ys, ms) if m[i] and y[i] > 0.5)
        neg = sum(1 for y, m in zip(ys, ms) if m[i] and y[i] <= 0.5)
        out.append(float(min(cap, max(1.0, neg / pos))) if pos else 1.0)
    return out


def _counts(y, p, t):
    tp = sum(1 for a, b in zip(y, p) if a > 0.5 and b >= t)
    fp = sum(1 for a, b in zip(y, p) if a <= 0.5 and b >= t)
    fn = sum(1 for a, b in zip(y, p) if a > 0.5 and b < t)
    return tp, fp, fn


def best_f1_threshold(y, p):
    best = (0.0, 0.5)
    for t in sorted(set(round(float(v), 4) for v in p)):
        tp, fp, fn = _counts(y, p, t)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


def precision_threshold(y, p, floor=0.8):
    """Highest-recall threshold whose precision is at least `floor`.

    If none qualifies, the threshold with the highest precision (ties: higher cut).
    """
    best = None  # (recall, precision, t)
    fallback = (-1.0, 0.5)
    for t in sorted(set(round(float(v), 4) for v in p)):
        tp, fp, fn = _counts(y, p, t)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        if prec > fallback[0] or (prec == fallback[0] and t > fallback[1]):
            fallback = (prec, t)
        if tp and prec + 1e-9 >= floor and (best is None or rec > best[0] or (rec == best[0] and prec > best[1])):
            best = (rec, prec, t)
    return best[2] if best else fallback[1]


# ---------------------------------------------------------------- cache
def feature_array_path(feat: Path, row: dict) -> Path:
    """npz written by extract_features. PPE rows may live under the ppe_video subfolder."""
    primary = feat / row["source"] / f"{row['key']}.npz"
    if row.get("source") not in ("ppe_video", "pexels"):
        return primary
    for cand in (primary,
                 feat / "ppe_video" / row["source"] / f"{row['key']}.npz",
                 feat / "ppe_video" / f"{row['key']}.npz"):
        if cand.is_file():
            return cand
    return primary


def index_files_for(feat: Path, sources: list[str]) -> list:
    """Root index_r*.jsonl, plus the ppe_video subfolder when that source is requested."""
    files = sorted(feat.glob("index_r*.jsonl"))
    if "ppe_video" in sources or "pexels" in sources:
        files.extend(sorted((feat / "ppe_video").glob("index_r*.jsonl")))
    seen, out = set(), []
    for p in files:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def cmd_cache(a):
    import numpy as np
    feat = Path(a.features)
    sources = [s.strip() for s in (getattr(a, "sources", "") or "").split(",") if s.strip()]
    rows = {}
    for p in index_files_for(feat, sources):
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
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
    if sources:
        want = set(sources)
        items = [r for r in items if r["source"] in want]
        if not items:
            raise SystemExit(f"no index rows for --sources {','.join(sources)} under {feat}")
    shwd_man = {}
    if any(r["source"] == "shwd" for r in items):
        mp = resolve_repo_file(a.shwd_manifest)
        if not mp.is_file():
            raise SystemExit(f"shwd features in the index but manifest not found: {a.shwd_manifest}")
        with open(mp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    m = json.loads(line)
                    shwd_man[m["id"]] = m
        missing = [r["id"] for r in items if r["source"] == "shwd" and r["id"] not in shwd_man]
        if missing:
            raise SystemExit(f"{len(missing)} shwd feature rows are missing from {mp}, e.g. {missing[0]}")
    gh, gw = GRID[0] // POOL, GRID[1] // POOL
    total = sum(r["windows"] for r in items)
    out = Path(os.path.expanduser(getattr(a, "out", "") or a.cache)); out.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out / "tokens.npy", mode="w+", dtype=np.float16,
                                   shape=(total, gh * gw, 1024))
    meta, off, t0 = [], 0, time.time()
    for k, r in enumerate(items):
        z = np.load(feature_array_path(feat, r))
        g = z["grid"].astype(np.float32)                       # (W, 24, 42, D)
        w = g.shape[0]
        g = g.reshape(w, gh, POOL, gw, POOL, -1).mean((2, 4)).reshape(w, gh * gw, -1)
        mm[off:off + w] = g.astype(np.float16)
        extra = {}
        if r["source"] == "voxel51":
            y, m = voxel_targets(labels[r["id"]], ignore=[])
            pm = None
        elif r["source"] == "zenodo":
            mr = man[r["id"]]
            y, m = zenodo_targets(int(mr.get("n_hand_boxes") or 0))
            pm = box_patch_mask(read_yolo(str(TRAINING / mr["label_path"])) if mr.get("label_path") else [], gh, gw)
        elif r["source"] == "shwd":
            mr = shwd_man[r["id"]]
            y, m = shwd_targets(1.0 if int(mr.get("n_bare") or 0) > 0 else 0.0)
            pm = None
            extra = {"bare_boxes": mr.get("bare_boxes") or [], "hat_boxes": mr.get("hat_boxes") or [],
                     "width": int(mr["width"]), "height": int(mr["height"])}
        elif r["source"] in ("ppe_video", "pexels"):
            y, m = [0.0] * 10, [0.0] * 10
            pm = None
            extra = {"starts": r.get("starts") or [], "path": r.get("path") or ""}
            if r.get("width") and r.get("height"):
                extra["width"], extra["height"] = int(r["width"]), int(r["height"])
        else:
            raise SystemExit(f"unknown feature source {r['source']}")
        rec = {"id": r["id"], "source": r["source"], "split": r["split"], "off": off, "n": w,
               "y": y, "m": m, "patch": pm}
        rec.update(extra)
        meta.append(rec)
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


def ppe_patch_bce(s, valid, rs, kept, dev, patch_pw):
    """Aux BCE on s_F2 / s_F5. Weight 0 patches are ignored. Negatives carry their box weight."""
    import torch
    import torch.nn.functional as F
    P = (GRID[0] // POOL) * (GRID[1] // POOL)
    terms = []
    pw = torch.tensor(float(patch_pw), device=dev)
    for fi, fname in ((FACTORS.index("F2"), "F2"), (FACTORS.index("F5"), "F5")):
        ys, ws, idxs = [], [], []
        for i, r in enumerate(rs):
            wins = r.get("ppe_windows")
            if not wins:
                continue
            t_parts, w_parts, any_sup = [], [], False
            for wi in kept[i]:
                spec = wins[wi] if wi < len(wins) else None
                patch = None if not spec else (spec.get("patch") or {}).get(fname)
                labeled = spec is not None and spec.get(fname) is not None
                if not labeled or not patch:
                    t_parts.append([0.0] * P)
                    w_parts.append([0.0] * P)
                else:
                    if len(patch["y"]) != P:
                        raise SystemExit(
                            f"ppe patch target for {r.get('id')} has length {len(patch['y'])}, expected {P}"
                        )
                    t_parts.append(list(patch["y"]))
                    w_parts.append(list(patch["w"]))
                    if any(v > 0 for v in patch["w"]):
                        any_sup = True
            if not any_sup:
                continue
            ys.append([v for part in t_parts for v in part])
            ws.append([v for part in w_parts for v in part])
            idxs.append(i)
        if not idxs:
            continue
        maxlen = max(len(t) for t in ys)
        t = torch.zeros(len(idxs), maxlen, device=dev)
        wt = torch.zeros(len(idxs), maxlen, device=dev)
        for j, (yy, ww) in enumerate(zip(ys, ws)):
            t[j, :len(yy)] = torch.tensor(yy, device=dev)
            wt[j, :len(ww)] = torch.tensor(ww, device=dev)
        sp = s[idxs, :maxlen, fi]
        vp = valid[idxs, :maxlen].float()
        pl = F.binary_cross_entropy_with_logits(sp, t, reduction="none", pos_weight=pw)
        mask = (wt > 0).float() * vp
        if float(mask.sum()) <= 0:
            continue
        terms.append((pl * wt * vp).sum() / mask.sum().clamp(min=1))
    if not terms:
        return None
    return torch.stack(terms).mean()


def ppe_window_bce(s, valid, rs, kept, dev):
    """BCE on the max patch logit of each window with a non-null F2 or F5 label.

    The clip-level bag label stays as it is. A window labeled 0 pushes every
    patch in that window down, so a frame with no bare head inside a present
    clip is not forced positive. Null window labels are skipped. Returns None
    when the batch has no such window. The caller multiplies by --ppe-window-weight.
    """
    import torch
    import torch.nn.functional as F
    default_p = (GRID[0] // POOL) * (GRID[1] // POOL)
    terms = []
    for fi, fname in ((FACTORS.index("F2"), "F2"), (FACTORS.index("F5"), "F5")):
        logits, targets = [], []
        for i, r in enumerate(rs):
            wins = r.get("ppe_windows")
            if not wins:
                continue
            p_count = int(r.get("n_patches") or default_p)
            for k, wi in enumerate(kept[i]):
                spec = wins[wi] if wi < len(wins) else None
                if spec is None or spec.get(fname) is None:
                    continue
                start, end = k * p_count, (k + 1) * p_count
                if start >= s.shape[1]:
                    continue
                end = min(end, s.shape[1])
                sl = s[i, start:end, fi]
                vm = valid[i, start:end]
                if not bool(vm.any()):
                    continue
                logits.append(sl.masked_fill(~vm, float("-inf")).max())
                targets.append(float(spec[fname]))
        if not logits:
            continue
        lt = torch.stack(logits)
        tt = torch.tensor(targets, dtype=lt.dtype, device=dev)
        terms.append(F.binary_cross_entropy_with_logits(lt, tt))
    if not terms:
        return None
    return torch.stack(terms).mean()


def _labeled_metrics(yy, pp):
    from sklearn.metrics import average_precision_score, roc_auc_score
    pos = int(sum(1 for v in yy if v > 0.5))
    d = {"n": len(yy), "pos": pos, "neg": len(yy) - pos}
    if len(yy) and len({1 if v > 0.5 else 0 for v in yy}) == 2:
        d["auroc"] = float(roc_auc_score(yy, pp))
        d["ap"] = float(average_precision_score(yy, pp))
        d["base_rate"] = pos / len(yy)
    return d


def _with_threshold(d, yy, pp, t):
    d = dict(d)
    if not yy:
        d["threshold"] = t
        return d
    tp = sum(1 for q, w in zip(yy, pp) if q > 0.5 and w >= t)
    fp = sum(1 for q, w in zip(yy, pp) if q <= 0.5 and w >= t)
    fn = sum(1 for q, w in zip(yy, pp) if q > 0.5 and w < t)
    d.update({"threshold": t,
              "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
              "precision": tp / (tp + fp) if tp + fp else 0.0,
              "recall": tp / (tp + fn) if tp + fn else 0.0})
    return d


def _choose_threshold(yy, pp, precision_floor):
    if len({1 if v > 0.5 else 0 for v in yy}) != 2:
        return 0.5
    if precision_floor and precision_floor > 0:
        return precision_threshold(yy, pp, precision_floor)
    return best_f1_threshold(yy, pp)


def collect_ppe_scores(rows, probs, evid):
    """Clip probabilities, per-window max patch-evidence, and supervised patch scores.

    ``evid`` maps a factor index to a list of sigmoid(evidence) tensors, one per
    predict chunk of 32 rows. Window score is the max evidence in that window.
    """
    factors = ("F2", "F5")
    index = {"F2": FACTORS.index("F2"), "F5": FACTORS.index("F5")}
    buckets = ("all", "pexels", "voxel51")
    clip = {b: {f: {"y": [], "p": []} for f in factors} for b in buckets}
    window = {b: {f: {"y": [], "p": []} for f in factors} for b in buckets}
    patch = {f: {"y": [], "p": []} for f in factors}
    if not rows:
        return clip, window, patch
    for k, r in enumerate(rows):
        src = r.get("ppe_source") or ppe_metric_source(r)
        for f in factors:
            i = index[f]
            if not r["m"][i]:
                continue
            for b in ("all", src):
                clip[b][f]["y"].append(float(r["y"][i]))
                clip[b][f]["p"].append(float(probs[k, i]))
    offset = 0
    for f in factors:
        maps = evid[index[f]]
        bi = 0
        mi = 0
        while bi < len(rows):
            chunk = rows[bi:bi + 32]
            pm = maps[mi]
            mi += 1
            for j, r in enumerate(chunk):
                sig = pm[j]
                n = int(r["n"])
                p_count = int(r.get("n_patches") or (sig.shape[0] // max(n, 1)))
                wins = r.get("ppe_windows") or [None] * n
                src = r.get("ppe_source") or ppe_metric_source(r)
                for w in range(n):
                    spec = wins[w] if w < len(wins) else None
                    sl = sig[w * p_count:(w + 1) * p_count]
                    if spec is None or spec.get(f) is None or sl.numel() == 0:
                        continue
                    score = float(sl.max())
                    for b in ("all", src):
                        window[b][f]["y"].append(float(spec[f]))
                        window[b][f]["p"].append(score)
                    packed = (spec.get("patch") or {}).get(f)
                    if not packed:
                        continue
                    for yv, wv, sv in zip(packed["y"], packed["w"], sl.tolist()):
                        if wv > 0:
                            patch[f]["y"].append(float(yv))
                            patch[f]["p"].append(float(sv))
            bi += 32
        offset += 0
    return clip, window, patch


def pack_ppe_block(val_rows, val_probs, val_evid, test_rows, test_probs, test_evid, precision_floor):
    """PPE section: clip and window AUROC/AP, thresholds fit on val, patch AUROC on test boxes."""
    v_clip, v_win, _v_patch = collect_ppe_scores(val_rows, val_probs, val_evid) if val_rows else (
        None, None, None)
    t_clip, t_win, t_patch = collect_ppe_scores(test_rows, test_probs, test_evid) if test_rows else (
        {b: {f: {"y": [], "p": []} for f in ("F2", "F5")} for b in ("all", "pexels", "voxel51")},
        {b: {f: {"y": [], "p": []} for f in ("F2", "F5")} for b in ("all", "pexels", "voxel51")},
        {f: {"y": [], "p": []} for f in ("F2", "F5")})
    if v_clip is None:
        v_clip = {b: {f: {"y": [], "p": []} for f in ("F2", "F5")} for b in ("all", "pexels", "voxel51")}
        v_win = {b: {f: {"y": [], "p": []} for f in ("F2", "F5")} for b in ("all", "pexels", "voxel51")}
    clip_thr, win_thr = {}, {}
    for f in ("F2", "F5"):
        clip_thr[f] = _choose_threshold(v_clip["all"][f]["y"], v_clip["all"][f]["p"], precision_floor)
        win_thr[f] = _choose_threshold(v_win["all"][f]["y"], v_win["all"][f]["p"], precision_floor)
    def pack(store, thr):
        out = {}
        for b in ("all", "pexels", "voxel51"):
            out[b] = {}
            for f in ("F2", "F5"):
                yy, pp = store[b][f]["y"], store[b][f]["p"]
                out[b][f] = _with_threshold(_labeled_metrics(yy, pp), yy, pp, thr[f])
        return out
    patch_out = {}
    for f in ("F2", "F5"):
        yy, pp = t_patch[f]["y"], t_patch[f]["p"]
        d = {"n_patches": len(yy), "pos": int(sum(1 for v in yy if v > 0.5)),
             "neg": int(sum(1 for v in yy if v <= 0.5))}
        if len(yy) and len(set(int(v > 0.5) for v in yy)) == 2:
            from sklearn.metrics import average_precision_score, roc_auc_score
            d["auroc"] = float(roc_auc_score(yy, pp))
            d["ap"] = float(average_precision_score(yy, pp))
        patch_out[f] = d
    return {"score_window": "max_patch_evidence",
            "thresholds_clip": clip_thr, "thresholds_window": win_thr,
            "clip": pack(t_clip, clip_thr), "window": pack(t_win, win_thr),
            "patch": patch_out,
            "val_counts": {b: {f: _labeled_metrics(v_clip[b][f]["y"], v_clip[b][f]["p"])
                               for f in ("F2", "F5")} for b in ("all", "pexels", "voxel51")}}


def load_meta(a):
    """Labels, splits and token banks. PPE is opt-in and does not change the default sets."""
    import numpy as np
    cache = Path(os.path.expanduser(a.cache))
    tok = np.load(cache / "tokens.npy", mmap_mode="r")
    meta = [json.loads(l) for l in open(cache / "meta.jsonl", encoding="utf-8") if l.strip()]
    banks = [tok]
    for r in meta:
        r["bank"] = 0
        r["n_patches"] = int(tok.shape[1])
    ignore = [f for f in a.ignore.split(",") if f]
    labels = {}
    with open(TRAINING / a.labels, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                labels[r["clip_id"]] = r
    ppe_labels = load_ppe_label_file(a.ppe_labels) if getattr(a, "ppe_labels", "") else None
    if getattr(a, "ppe_cache", ""):
        if ppe_labels is None:
            raise SystemExit("--ppe-cache requires --ppe-labels")
        ppe_cache = Path(os.path.expanduser(a.ppe_cache))
        ppe_tok = np.load(ppe_cache / "tokens.npy", mmap_mode="r")
        have = {r["id"] for r in meta}
        banks.append(ppe_tok)
        with open(ppe_cache / "meta.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["id"] in have:
                    print(f"ppe cache skips {r['id']}: already in the main cache", flush=True)
                    continue
                r["bank"] = len(banks) - 1
                r["n_patches"] = int(ppe_tok.shape[1])
                meta.append(r)
    gh, gw = GRID[0] // POOL, GRID[1] // POOL
    for r in meta:
        if r["source"] == "voxel51":
            lab = labels[r["id"]]
            r["y"], r["m"] = voxel_targets(lab, ignore=ignore)
            r["source_label"] = lab.get("source_label") or ""
        elif r["source"] == "zenodo":
            # Labels are stored in the feature cache; rebuild them so a label-policy
            # change does not require rewriting the memmap.
            r["y"], r["m"] = zenodo_targets(1 if r["y"][0] > 0.5 else 0)
        elif r["source"] in ("ppe_video", "pexels"):
            if ppe_labels is None:
                raise SystemExit(f"{r['id']} is a ppe feature row; pass --ppe-labels")
            lab = ppe_labels.get(r["id"])
            if lab is None:
                raise SystemExit(f"ppe feature {r['id']} has no row in {a.ppe_labels}")
            apply_pexels_row(
                r, lab, gh, gw, a.f2_min_frac, a.f2_hatneg_weight,
                bg_neg=getattr(a, "ppe_bg_neg", 0.0), bg_margin=getattr(a, "ppe_bg_margin", 1.0))
        r["set"] = assign_split(r, a.val_frac, a.seed, a.cv, a.fold)
    return {"banks": banks, "meta": meta, "labels": labels, "ppe_labels": ppe_labels, "gh": gh, "gw": gw}


def cmd_train(a):
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    prep = load_meta(a)
    banks, meta = prep["banks"], prep["meta"]
    ppe_labels = prep["ppe_labels"]
    shwd_rows = None
    if a.f2_box_source == "shwd":
        mp = resolve_repo_file(a.shwd_manifest)
        if not mp.is_file():
            raise SystemExit(f"shwd manifest not found: {a.shwd_manifest}")
        with open(mp, encoding="utf-8") as f:
            shwd_rows = [json.loads(line) for line in f if line.strip()]
    meta = apply_train_sources(meta, use_zenodo=a.use_zenodo, f2_box_source=a.f2_box_source,
                               shwd_rows=shwd_rows, f2_min_frac=a.f2_min_frac,
                               f2_hatneg_weight=a.f2_hatneg_weight, cache_label=str(a.cache))
    if a.f2_box_source == "shwd":
        n_shwd = sum(1 for r in meta if r["source"] == "shwd")
        print(f"shwd F2 boxes on {n_shwd} rows (min_frac={a.f2_min_frac}, hatneg={a.f2_hatneg_weight})",
              flush=True)
    if a.drop_glimmer_f2:
        cleared = mask_glimmer_f2(meta)
        print(f"glimmer F2 masked on {cleared} voxel51 train rows", flush=True)
    if a.zenodo_f9_frac < 1:
        cleared = mask_zenodo_f9(meta, a.zenodo_f9_frac, a.seed)
        print(f"zenodo F9 mask cleared on {cleared} train rows (frac={a.zenodo_f9_frac})", flush=True)
    if ppe_labels is not None and (a.ppe_override_voxel or a.mask_glimmer_f2f5):
        stats = apply_voxel_ppe_policy(
            meta, ppe_labels, a.ppe_override_voxel, a.mask_glimmer_f2f5,
            prep["gh"], prep["gw"], a.f2_min_frac, a.f2_hatneg_weight,
            bg_neg=getattr(a, "ppe_bg_neg", 0.0), bg_margin=getattr(a, "ppe_bg_margin", 1.0))
        print(f"ppe voxel F2/F5 override={stats['overridden']} glimmer masked={stats['cleared']}", flush=True)
    tr = [r for r in meta if r["set"] == "train"]
    va = [r for r in meta if r["set"] == "val"]
    te = [r for r in meta if r["set"] == "test"]
    ppe_active = bool(getattr(a, "ppe_labels", "") or getattr(a, "ppe_cache", "")
                      or getattr(a, "ppe_override_voxel", False) or getattr(a, "mask_glimmer_f2f5", False))
    # Headline val/test stay the historical sources. Pexels clips are reported under "ppe".
    va_main = [r for r in va if not r.get("ppe_item")]
    te_main = [r for r in te if not r.get("ppe_item")]
    ppe_val = [r for r in va if r.get("ppe_item") or r.get("ppe_labeled")]
    ppe_test = [r for r in te if r.get("ppe_item") or r.get("ppe_labeled")]
    pw = torch.tensor(pos_weights([r["y"] for r in tr], [r["m"] for r in tr], a.pw_cap), device=dev)
    print("split sizes", {k: len(v) for k, v in (("train", tr), ("val", va), ("test", te))}, "pos_weight",
          [round(v, 2) for v in pw.tolist()], flush=True)
    if ppe_active:
        print("ppe sizes", {"train": sum(1 for r in tr if r.get("ppe_item")),
                            "val": len(ppe_val), "test": len(ppe_test)}, flush=True)
        if getattr(a, "ppe_bg_neg", 0.0) > 0 or getattr(a, "ppe_window_labels", False):
            print(f"ppe targets bg_neg={a.ppe_bg_neg} margin={a.ppe_bg_margin} "
                  f"window_labels={bool(a.ppe_window_labels)} window_weight={a.ppe_window_weight}",
                  flush=True)
    # load everything needed into RAM (pooled tokens are small: ~0.5 MB per window)
    def batchify(rs, train):
        xs, kepts = [], []
        for r in rs:
            bank = banks[r.get("bank", 0)]
            x = np.asarray(bank[r["off"]:r["off"] + r["n"]])          # (W, P, D)
            keep = list(range(r["n"]))
            if train and r["n"] > 1 and a.window_drop > 0:
                keep = [i for i in range(r["n"]) if random.random() > a.window_drop] or [random.randrange(r["n"])]
                x = x[keep]
            kepts.append(keep)
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
        PF = [int(r.get("patch_factor") or 0) for r in rs]
        PW = [r.get("patch_weight") for r in rs]
        return X, V, y, m, P, PF, PW, kepts

    model = build_model(hidden=a.hidden, drop=a.dropout).to(dev)
    if getattr(a, "init_checkpoint", ""):
        ck_path = resolve_ckpt(a.init_checkpoint, "init checkpoint")
        init_blob = torch.load(ck_path, map_location=dev, weights_only=False)
        ck_h = int((init_blob.get("config") or {}).get("hidden") or 0)
        if ck_h and ck_h != a.hidden:
            raise SystemExit(
                f"init checkpoint hidden {ck_h} != --hidden {a.hidden}. "
                "Pass the checkpoint's --hidden (h3_wide uses 512)."
            )
        model.load_state_dict(init_blob["state_dict"])
        print(f"loaded init checkpoint {ck_path}", flush=True)
    keep = configure_freeze(model, parse_factor_list(getattr(a, "freeze_except", "")))
    frozen_rows = snapshot_factor_rows(model) if keep is not None else None
    if keep is not None:
        print(f"freeze-except {[FACTORS[i] for i in keep]} (shared trunk frozen)", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)
    src_w = source_weights(tr, a.zenodo_share, a.shwd_weight, getattr(a, "ppe_weight", 0.0))

    def predict(rs, net=None):
        net = model if net is None else net
        net.eval(); out = []
        evid = {0: [], 1: [], 4: []}
        with torch.no_grad():
            for i in range(0, len(rs), 32):
                X, V, y, m, P, PF, PW, _kept = batchify(rs[i:i + 32], False)
                z, s, w = net(X, V)
                out.append(torch.sigmoid(z).cpu())
                evid[0].append(torch.sigmoid(s[..., 0]).cpu())   # F1 patch evidence
                evid[1].append(torch.sigmoid(s[..., 1]).cpu())   # F2 patch evidence
                evid[4].append(torch.sigmoid(s[..., 4]).cpu())   # F5 patch evidence
        net.train()
        return torch.cat(out).numpy(), evid

    scores_by_id = None
    if a.hard_neg_checkpoint:
        if a.hard_neg_factor not in FACTORS:
            raise SystemExit(f"unknown hard-neg factor {a.hard_neg_factor}")
        ck_path = resolve_ckpt(a.hard_neg_checkpoint)
        scorer = build_model(hidden=a.hidden, drop=a.dropout).to(dev)
        blob = torch.load(ck_path, map_location=dev, weights_only=False)
        scorer.load_state_dict(blob["state_dict"])
        voxel_tr = [r for r in tr if r["source"] == "voxel51"]
        pred, _evid = predict(voxel_tr, scorer)
        fi = FACTORS.index(a.hard_neg_factor)
        scores_by_id = {r["id"]: float(pred[k, fi]) for k, r in enumerate(voxel_tr)}
        del scorer
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"hard-neg scored {len(scores_by_id)} train clips from {ck_path.name}", flush=True)
    weights = sampling_weights(tr, src_w, scores_by_id, a.hard_neg_factor, a.hard_neg_min)
    if scores_by_id is not None:
        n_boost = sum(1 for r, w in zip(tr, weights) if r["source"] == "voxel51" and w > src_w["voxel51"] * 1.01)
        print(f"hard-neg boosted {n_boost} voxel train clips x{HARD_NEG_BOOST}", flush=True)

    def evaluate(rs):
        if not rs:
            return {}, None, None
        p, evid = predict(rs)
        f1_maps, f2_maps = evid[0], evid[1]
        res = {}
        for i, f in enumerate(FACTORS):
            yy = [r["y"][i] for r, mm in zip(rs, [r["m"] for r in rs]) if mm[i]]
            pp = [float(p[k, i]) for k, r in enumerate(rs) if r["m"][i]]
            d = {"n": len(yy), "pos": int(sum(yy))}
            if len(set(yy)) == 2:
                d["auroc"] = float(roc_auc_score(yy, pp)); d["ap"] = float(average_precision_score(yy, pp))
                d["base_rate"] = d["pos"] / d["n"]
            res[f] = d
        # Patch maps vs boxes. Factor 0 is Zenodo hands (F1); factor 1 is SHWD bare heads (F2).
        # A Zenodo-only eval has no factor-1 rows, so F1_patch_vs_boxes is unchanged.
        for factor, maps, key in ((0, f1_maps, "F1_patch_vs_boxes"), (1, f2_maps, "F2_patch_vs_boxes")):
            py, pp = [], []
            k = 0
            for bi in range(0, len(rs), 32):
                chunk = rs[bi:bi + 32]
                pm = maps[k]
                k += 1
                for j, r in enumerate(chunk):
                    if r.get("patch") is None or int(r.get("patch_factor") or 0) != factor:
                        continue
                    t = np.asarray(r["patch"], np.float32).reshape(-1)
                    py.extend(t.tolist())
                    pp.extend(pm[j, :len(t)].numpy().tolist())
            if py and len(set(py)) == 2:
                res[key] = {"n_patches": len(py), "pos": int(sum(py)),
                            "auroc": float(roc_auc_score(py, pp)),
                            "ap": float(average_precision_score(py, pp))}
        return res, p, evid

    def summary(res, extra=()):
        aps = [v["ap"] for f, v in res.items() if f in FACTORS and "ap" in v]
        aps.extend(extra)
        return float(np.mean(aps)) if aps else 0.0

    best, best_state, hist, t0 = -1.0, None, [], time.time()
    for ep in range(a.epochs):
        tot = 0.0
        for st in range(a.steps):
            rs = random.choices(tr, weights=weights, k=a.batch)
            X, V, y, m, P, PF, PW, kept = batchify(rs, True)
            z, s, w = model(X, V)
            loss = masked_bce(z, y, m, pw)
            if a.patch_loss > 0:
                # F1 hand boxes. PF is 0 for every Zenodo row, so this matches the previous loss.
                idx = [i for i, p in enumerate(P) if p is not None and PF[i] == 0]
                if idx:
                    t = torch.tensor([sum(P[i], []) for i in idx], device=dev)
                    sp = s[idx, :t.shape[1], 0]
                    vp = V[idx, :t.shape[1]].float()
                    pl = torch.nn.functional.binary_cross_entropy_with_logits(
                        sp, t, reduction="none", pos_weight=torch.tensor(a.patch_pw, device=dev))
                    loss = loss + a.patch_loss * (pl * vp).sum() / vp.sum().clamp(min=1)
                idx2 = [i for i, p in enumerate(P) if p is not None and PF[i] == 1]
                if idx2:
                    t = torch.tensor([sum(P[i], []) for i in idx2], device=dev)
                    wt = torch.tensor(
                        [sum(PW[i], []) if PW[i] is not None else [1.0] * t.shape[1] for i in idx2],
                        device=dev)
                    sp = s[idx2, :t.shape[1], 1]
                    vp = V[idx2, :t.shape[1]].float()
                    pl = torch.nn.functional.binary_cross_entropy_with_logits(
                        sp, t, reduction="none", pos_weight=torch.tensor(a.patch_pw, device=dev))
                    loss = loss + a.patch_loss * (pl * vp * wt).sum() / vp.sum().clamp(min=1)
            if getattr(a, "ppe_patch_loss", 0.0) > 0:
                extra_loss = ppe_patch_bce(s, V, rs, kept, dev, a.patch_pw)
                if extra_loss is not None:
                    loss = loss + a.ppe_patch_loss * extra_loss
            if getattr(a, "ppe_window_labels", False):
                wloss = ppe_window_bce(s, V, rs, kept, dev)
                if wloss is not None:
                    loss = loss + float(a.ppe_window_weight) * wloss
            opt.zero_grad(); loss.backward()
            if keep is not None:
                mask_factor_grads(model, keep)
            opt.step()
            if keep is not None:
                restore_factor_rows(model, frozen_rows, keep)
            sched.step()
            tot += float(loss)
        base_res, _, _ = evaluate(va_main)
        extra_ap = []
        ppe_ep = None
        if ppe_val:
            ppe_ep, _, _ = evaluate(ppe_val)
            for f in ("F2", "F5"):
                if "ap" in ppe_ep.get(f, {}):
                    extra_ap.append(ppe_ep[f]["ap"])
        sc = summary(base_res, extra_ap)
        row = {"epoch": ep + 1, "loss": tot / a.steps, "val_mean_ap": sc}
        msg = f"ep {ep+1} loss {tot/a.steps:.4f} val_mAP {sc:.3f}"
        if ppe_active:
            f1 = base_res.get("F1", {})
            row["val_f1_ap"] = f1.get("ap")
            row["val_f1_auroc"] = f1.get("auroc")
            if ppe_ep:
                row["ppe_val_f2_ap"] = ppe_ep.get("F2", {}).get("ap")
                row["ppe_val_f5_ap"] = ppe_ep.get("F5", {}).get("ap")
            msg += f" val_F1_AP {f1.get('ap', float('nan')):.3f}"
            if ppe_ep:
                msg += f" ppe_F2_AP {ppe_ep.get('F2', {}).get('ap', float('nan')):.3f}"
                msg += f" ppe_F5_AP {ppe_ep.get('F5', {}).get('ap', float('nan')):.3f}"
        hist.append(row)
        print(f"{msg} ({time.time()-t0:.0f}s)", flush=True)
        if sc > best:
            best, best_state = sc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    report = build_report(a, evaluate, tr, va, te, va_main, te_main, ppe_val, ppe_test, ppe_active, pw, hist, best, t0)
    save_training_outputs(a, best_state, report)


def build_report(a, evaluate, tr, va, te, va_main, te_main, ppe_val, ppe_test, ppe_active, pw, hist, best, t0):
    """Validation thresholds, headline test metrics, and the optional PPE block."""
    import numpy as np
    # Thresholds see every val row, so PPE val F2/F5 move F2/F5 cuts. F1 on PPE rows is masked.
    vres_all, vp, _ = evaluate(va) if va else ({}, None, None)
    if vp is None:
        vp = np.zeros((0, len(FACTORS)))
        vres_all = {f: {"n": 0, "pos": 0} for f in FACTORS}
    thr = {}
    for i, f in enumerate(FACTORS):
        yy = [r["y"][i] for r in va if r["m"][i]]
        pp = [float(vp[k, i]) for k, r in enumerate(va) if r["m"][i]]
        if len(set(yy)) != 2:
            thr[f] = 0.5
        elif a.precision_floor > 0:
            thr[f] = precision_threshold(yy, pp, a.precision_floor)
        else:
            thr[f] = best_f1_threshold(yy, pp)
    vres = vres_all if len(va_main) == len(va) else (evaluate(va_main)[0] if va_main else vres_all)
    tres, tp, _ = evaluate(te_main) if te_main else ({}, None, None)
    if tp is None:
        tp = np.zeros((0, len(FACTORS)))
    for i, f in enumerate(FACTORS):
        yy = [r["y"][i] for r in te_main if r["m"][i]]
        pp = [float(tp[k, i]) for k, r in enumerate(te_main) if r["m"][i]]
        if yy and f in tres:
            t = thr[f]
            tpn = sum(1 for q, w in zip(yy, pp) if q > .5 and w >= t)
            fp = sum(1 for q, w in zip(yy, pp) if q <= .5 and w >= t)
            fn = sum(1 for q, w in zip(yy, pp) if q > .5 and w < t)
            tres[f].update({"threshold": t, "f1": 2 * tpn / (2 * tpn + fp + fn) if tpn else 0.0,
                            "precision": tpn / (tpn + fp) if tpn + fp else 0.0,
                            "recall": tpn / (tpn + fn) if tpn + fn else 0.0})
    tv = [r for r in te_main if r["source"] == "voxel51"]
    tz = [r for r in te_main if r["source"] == "zenodo"]
    ts = [r for r in te_main if r["source"] == "shwd"]
    vs = [r for r in va_main if r["source"] == "shwd"]
    sub = {"voxel51": evaluate(tv)[0] if tv else {}, "zenodo": evaluate(tz)[0] if tz else {}}
    shwd_block = None
    if ts or vs:
        shwd_test = evaluate(ts)[0] if ts else {}
        sub["shwd"] = shwd_test
        shwd_block = {"validation": evaluate(vs)[0] if vs else {}, "test": shwd_test}
    ppe_block = None
    if ppe_active:
        _, val_p, val_e = evaluate(ppe_val) if ppe_val else (None, None, None)
        _, te_p, te_e = evaluate(ppe_test) if ppe_test else (None, None, None)
        if val_p is None:
            val_p = np.zeros((0, len(FACTORS)))
            val_e = {0: [], 1: [], 4: []}
        if te_p is None:
            te_p = np.zeros((0, len(FACTORS)))
            te_e = {0: [], 1: [], 4: []}
        ppe_block = pack_ppe_block(ppe_val, val_p, val_e, ppe_test, te_p, te_e, a.precision_floor)
    preds = [{"id": r["id"], "p": [round(float(v), 5) for v in tp[k]], "y": r["y"], "m": r["m"]}
             for k, r in enumerate(te_main) if r["source"] == "voxel51"]
    return {"vres": vres, "tres": tres, "sub": sub, "thr": thr, "preds": preds,
            "shwd_block": shwd_block, "ppe_block": ppe_block, "tp": tp,
            "hist": hist, "best": best, "pw": pw, "t0": t0,
            "tr": tr, "va": va, "te": te}


def save_training_outputs(a, state, report):
    import torch as _t
    run = a.name or time.strftime("head_%Y%m%d_%H%M")
    ck = TRAINING / "04-head/checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    ck_path = guarded_output(ck / f"{run}.pt")
    _t.save({"state_dict": state, "config": vars(a), "factors": FACTORS, "thresholds": report["thr"],
             "pool": POOL, "grid": GRID}, ck_path)
    resd = Path(a.results_dir) if getattr(a, "results_dir", "") else TRAINING / "04-head/results"
    resd.mkdir(parents=True, exist_ok=True)
    out_path = guarded_output(resd / f"{run}.json")
    _write_result_json(out_path, run, a, report, eval_only=False)
    print(f"saved {ck_path} and {out_path}")


def _write_result_json(path, run, a, report, eval_only=False):
    pw = report["pw"]
    out = {"run": run, "config": vars(a), "best_val_mean_ap": report["best"], "history": report["hist"],
           "val": report["vres"], "test_predictions_voxel51": report["preds"],
           "test": report["tres"], "test_by_source": report["sub"], "thresholds": report["thr"],
           "sizes": {"train": len(report["tr"]), "val": len(report["va"]), "test": len(report["te"])},
           "pos_weight": pw.tolist() if hasattr(pw, "tolist") else list(pw),
           "minutes": (time.time() - report["t0"]) / 60}
    if eval_only:
        out["eval_only"] = True
        out["checkpoint"] = str(getattr(a, "checkpoint", ""))
    if report["shwd_block"] is not None:
        out["shwd"] = report["shwd_block"]
    if report["ppe_block"] is not None:
        out["ppe"] = report["ppe_block"]
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    printable = {f: {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}
                 for f, d in report["tres"].items() if isinstance(d, dict)}
    print("TEST", json.dumps(printable), flush=True)
    if report["ppe_block"] is not None:
        clip = report["ppe_block"]["clip"]["all"]
        brief = {f: {k: round(v, 3) if isinstance(v, float) else v for k, v in clip[f].items()}
                 for f in ("F2", "F5")}
        print("PPE_TEST_CLIP", json.dumps(brief), flush=True)


def cmd_eval(a):
    """Score a saved checkpoint on the same splits. Does not write a new checkpoint."""
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    if not getattr(a, "checkpoint", ""):
        raise SystemExit("eval requires --checkpoint")
    if not getattr(a, "name", ""):
        raise SystemExit("eval requires --name")
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    ck_path = resolve_ckpt(a.checkpoint, "checkpoint")
    blob = torch.load(ck_path, map_location=dev, weights_only=False)
    cfg = blob.get("config") or {}
    if "hidden" not in cfg:
        raise SystemExit(f"checkpoint {ck_path} has no config.hidden")
    hidden = int(cfg["hidden"])
    drop = float(cfg.get("dropout", 0.0))
    prep = load_meta(a)
    banks, meta = prep["banks"], prep["meta"]
    ppe_labels = prep["ppe_labels"]
    shwd_rows = None
    if a.f2_box_source == "shwd":
        mp = resolve_repo_file(a.shwd_manifest)
        if not mp.is_file():
            raise SystemExit(f"shwd manifest not found: {a.shwd_manifest}")
        with open(mp, encoding="utf-8") as f:
            shwd_rows = [json.loads(line) for line in f if line.strip()]
    meta = apply_train_sources(meta, use_zenodo=a.use_zenodo, f2_box_source=a.f2_box_source,
                               shwd_rows=shwd_rows, f2_min_frac=a.f2_min_frac,
                               f2_hatneg_weight=a.f2_hatneg_weight, cache_label=str(a.cache))
    if a.drop_glimmer_f2:
        cleared = mask_glimmer_f2(meta)
        print(f"glimmer F2 masked on {cleared} voxel51 train rows", flush=True)
    if a.zenodo_f9_frac < 1:
        cleared = mask_zenodo_f9(meta, a.zenodo_f9_frac, a.seed)
        print(f"zenodo F9 mask cleared on {cleared} train rows (frac={a.zenodo_f9_frac})", flush=True)
    if ppe_labels is not None and (a.ppe_override_voxel or a.mask_glimmer_f2f5):
        stats = apply_voxel_ppe_policy(
            meta, ppe_labels, a.ppe_override_voxel, a.mask_glimmer_f2f5,
            prep["gh"], prep["gw"], a.f2_min_frac, a.f2_hatneg_weight,
            bg_neg=getattr(a, "ppe_bg_neg", 0.0), bg_margin=getattr(a, "ppe_bg_margin", 1.0))
        print(f"ppe voxel F2/F5 override={stats['overridden']} glimmer masked={stats['cleared']}", flush=True)
    tr = [r for r in meta if r["set"] == "train"]
    va = [r for r in meta if r["set"] == "val"]
    te = [r for r in meta if r["set"] == "test"]
    ppe_active = bool(a.ppe_labels or a.ppe_cache or a.ppe_override_voxel or a.mask_glimmer_f2f5)
    va_main = [r for r in va if not r.get("ppe_item")]
    te_main = [r for r in te if not r.get("ppe_item")]
    ppe_val = [r for r in va if r.get("ppe_item") or r.get("ppe_labeled")]
    ppe_test = [r for r in te if r.get("ppe_item") or r.get("ppe_labeled")]
    pw = torch.tensor(pos_weights([r["y"] for r in tr], [r["m"] for r in tr], a.pw_cap), device=dev)
    print("split sizes", {k: len(v) for k, v in (("train", tr), ("val", va), ("test", te))}, "pos_weight",
          [round(v, 2) for v in pw.tolist()], flush=True)
    model = build_model(hidden=hidden, drop=drop).to(dev)
    model.load_state_dict(blob["state_dict"])
    print(f"eval checkpoint {ck_path} hidden={hidden} dropout={drop} device={dev}", flush=True)

    def batchify(rs, train):
        xs, kepts = [], []
        for r in rs:
            bank = banks[r.get("bank", 0)]
            x = np.asarray(bank[r["off"]:r["off"] + r["n"]])
            keep = list(range(r["n"]))
            if train and r["n"] > 1 and a.window_drop > 0:
                keep = [i for i in range(r["n"]) if random.random() > a.window_drop] or [random.randrange(r["n"])]
                x = x[keep]
            kepts.append(keep)
            xs.append(x.reshape(-1, x.shape[-1]))
        n = max(len(x) for x in xs)
        X = np.zeros((len(xs), n, xs[0].shape[-1]), np.float16)
        V = np.zeros((len(xs), n), bool)
        for i, x in enumerate(xs):
            X[i, :len(x)] = x; V[i, :len(x)] = True
        X, V = torch.from_numpy(X).to(dev).float(), torch.from_numpy(V).to(dev)
        y = torch.tensor([r["y"] for r in rs], device=dev)
        m = torch.tensor([r["m"] for r in rs], device=dev)
        P = [r.get("patch") for r in rs]
        PF = [int(r.get("patch_factor") or 0) for r in rs]
        PW = [r.get("patch_weight") for r in rs]
        return X, V, y, m, P, PF, PW, kepts

    def predict(rs, net=None):
        net = model if net is None else net
        net.eval(); out = []
        evid = {0: [], 1: [], 4: []}
        with torch.no_grad():
            for i in range(0, len(rs), 32):
                X, V, y, m, P, PF, PW, _kept = batchify(rs[i:i + 32], False)
                z, s, w = net(X, V)
                out.append(torch.sigmoid(z).cpu())
                evid[0].append(torch.sigmoid(s[..., 0]).cpu())
                evid[1].append(torch.sigmoid(s[..., 1]).cpu())
                evid[4].append(torch.sigmoid(s[..., 4]).cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, len(FACTORS))), evid

    def evaluate(rs):
        if not rs:
            return {}, None, None
        p, evid = predict(rs)
        f1_maps, f2_maps = evid[0], evid[1]
        res = {}
        for i, f in enumerate(FACTORS):
            yy = [r["y"][i] for r in rs if r["m"][i]]
            pp = [float(p[k, i]) for k, r in enumerate(rs) if r["m"][i]]
            d = {"n": len(yy), "pos": int(sum(yy))}
            if len(set(yy)) == 2:
                d["auroc"] = float(roc_auc_score(yy, pp)); d["ap"] = float(average_precision_score(yy, pp))
                d["base_rate"] = d["pos"] / d["n"]
            res[f] = d
        for factor, maps, key in ((0, f1_maps, "F1_patch_vs_boxes"), (1, f2_maps, "F2_patch_vs_boxes")):
            py, pp = [], []
            k = 0
            for bi in range(0, len(rs), 32):
                chunk = rs[bi:bi + 32]
                pm = maps[k]
                k += 1
                for j, r in enumerate(chunk):
                    if r.get("patch") is None or int(r.get("patch_factor") or 0) != factor:
                        continue
                    t = np.asarray(r["patch"], np.float32).reshape(-1)
                    py.extend(t.tolist())
                    pp.extend(pm[j, :len(t)].numpy().tolist())
            if py and len(set(py)) == 2:
                res[key] = {"n_patches": len(py), "pos": int(sum(py)),
                            "auroc": float(roc_auc_score(py, pp)),
                            "ap": float(average_precision_score(py, pp))}
        return res, p, evid

    t0 = time.time()
    base, _, _ = evaluate(va_main) if va_main else ({}, None, None)
    extra = []
    if ppe_val:
        ppe_ep, _, _ = evaluate(ppe_val)
        for f in ("F2", "F5"):
            if "ap" in (ppe_ep or {}).get(f, {}):
                extra.append(ppe_ep[f]["ap"])
    aps = [v["ap"] for f, v in (base or {}).items() if f in FACTORS and isinstance(v, dict) and "ap" in v]
    best = float(np.mean(aps + extra)) if aps or extra else 0.0
    report = build_report(a, evaluate, tr, va, te, va_main, te_main, ppe_val, ppe_test, ppe_active, pw, [], best, t0)
    resd = Path(a.results_dir) if a.results_dir else TRAINING / "04-head/results"
    resd.mkdir(parents=True, exist_ok=True)
    out_path = guarded_output(resd / f"{a.name}_eval.json")
    _write_result_json(out_path, a.name, a, report, eval_only=True)
    print(f"saved {out_path}")


def build_arg_parser():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("cache")
    c.add_argument("--features", default=DEFAULT_FEATURES)
    c.add_argument("--cache", default=DEFAULT_CACHE)
    c.add_argument("--labels", default="02-labeling/outputs/weak_labels_v2_vllm.jsonl")
    c.add_argument("--zenodo-train-stride", type=int, default=2)
    c.add_argument("--shwd-manifest", default="data/shwd_manifest.jsonl",
                   help="read only when the feature index contains shwd rows")
    c.add_argument("--sources", default="",
                   help="comma list of feature sources to cache; empty keeps every row in the root index")
    c.add_argument("--out", default="", help="cache directory; overrides --cache when set")
    t = sp.add_parser("train")
    t.add_argument("--cache", default=DEFAULT_CACHE)
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
    t.add_argument("--precision-floor", type=float, default=0.0,
                   help="if > 0, pick val thresholds at this precision instead of best F1")
    t.add_argument("--hard-neg-checkpoint", default="",
                   help="score Voxel51 train clips with this head and upweight its misses")
    t.add_argument("--hard-neg-factor", default="F9")
    t.add_argument("--hard-neg-min", type=float, default=0.8)
    t.add_argument("--zenodo-f9-frac", type=float, default=1.0,
                   help="fraction of Zenodo train rows that still supervise F9 (0 = none)")
    t.add_argument("--f2-box-source", default="", choices=["", "shwd"],
                   help="F2 box supervision (empty = off, current behaviour)")
    t.add_argument("--shwd-manifest", default="data/shwd_manifest.jsonl")
    t.add_argument("--shwd-weight", type=float, default=1.0,
                   help="sampling weight for an SHWD row, on the same scale as a voxel row (1)")
    t.add_argument("--f2-min-frac", type=float, default=0.30,
                   help="a pooled patch is a bare-head positive when a box covers at least this fraction")
    t.add_argument("--f2-hatneg-weight", type=float, default=2.0,
                   help="loss multiplier on patches inside a helmet (hat) box")
    t.add_argument("--drop-glimmer-f2", action="store_true",
                   help="mask Glimmer F2 on Voxel51 training rows (val and test stay labeled)")
    _add_ppe_args(t)
    t.add_argument("--results-dir", default="", help="results directory (default: 04-head/results)")
    e = sp.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--name", required=True)
    _add_eval_data_args(e)
    _add_ppe_args(e)
    e.add_argument("--results-dir", default="", help="results directory (default: 04-head/results)")
    return ap


def _add_ppe_args(p):
    p.add_argument("--ppe-labels", default="", help="jsonl of real-video F2/F5 labels (off by default)")
    p.add_argument("--ppe-cache", default="", help="memmap cache of ppe_video features, loaded beside --cache")
    p.add_argument("--ppe-weight", type=float, default=0.25,
                   help="sampling share of ppe clips in each batch, used only when those clips are loaded")
    p.add_argument("--ppe-override-voxel", action="store_true",
                   help="use PPE label rows for Voxel51 F2/F5 instead of Glimmer")
    p.add_argument("--mask-glimmer-f2f5", action="store_true",
                   help="mask Glimmer F2/F5 on Voxel51 clips that have no PPE label row")
    p.add_argument("--ppe-patch-loss", type=float, default=0.0,
                   help="aux patch BCE weight on s_F2/s_F5 from PPE window boxes (0 = off)")
    p.add_argument("--ppe-bg-neg", type=float, default=0.0,
                   help="weight on patches farther than --ppe-bg-margin from any box of that "
                        "factor in the window (0 = ignore background, previous behaviour)")
    p.add_argument("--ppe-bg-margin", type=float, default=1.0,
                   help="Chebyshev patch distance. Cells this close to a box stay unlabeled "
                        "by --ppe-bg-neg; farther cells are negatives")
    p.add_argument("--ppe-window-labels", action="store_true",
                   help="BCE on the max patch evidence of each window, using that window's "
                        "non-null F2/F5 label")
    p.add_argument("--ppe-window-weight", type=float, default=0.5,
                   help="weight of --ppe-window-labels (used only when that flag is set)")
    p.add_argument("--init-checkpoint", default="",
                   help="load these weights and fine-tune (h3_wide: also pass --hidden 512 --dropout 0.2)")
    p.add_argument("--freeze-except", default="",
                   help="comma list, e.g. F2,F5. Trains only those attention/evidence/bias rows. "
                        "Empty trains every parameter at --lr (pass a lower --lr for a full fine-tune)")


def _add_eval_data_args(p):
    """Same data-policy defaults as train, so an eval of h3_wide reproduces its splits."""
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--labels", default="02-labeling/outputs/weak_labels_v2_vllm.jsonl")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ignore", default="F3,F4")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--cv", type=int, default=0)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--use-zenodo", type=int, default=1)
    p.add_argument("--pw-cap", type=float, default=20.0)
    p.add_argument("--window-drop", type=float, default=0.25)
    p.add_argument("--precision-floor", type=float, default=0.0)
    p.add_argument("--zenodo-f9-frac", type=float, default=1.0)
    p.add_argument("--f2-box-source", default="", choices=["", "shwd"])
    p.add_argument("--shwd-manifest", default="data/shwd_manifest.jsonl")
    p.add_argument("--f2-min-frac", type=float, default=0.30)
    p.add_argument("--f2-hatneg-weight", type=float, default=2.0)
    p.add_argument("--drop-glimmer-f2", action="store_true")


def main(argv=None):
    a = build_arg_parser().parse_args(argv)
    {"cache": cmd_cache, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
