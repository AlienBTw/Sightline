"""Replay held-out clips through the live pipeline and summarize F1, F2 and F5.

Mirrors the f2 replay helper. F5 is left live (muted factors are
F6-F10) and outputs go under ``<SIGHTLINE_WORK_DIR>/ppe_eval/<run>/``
(work dir default ``<repo>/work``). GPU 1 only
(CUDA_VISIBLE_DEVICES=1, so the process cuda:0 is that card). Never writes
data/live and does not talk to the backend on port 8000.

  python tools/labeling/replay_eval.py --checkpoint training/04-head/checkpoints/<run>.pt

Clips come from training/data/ppe_demo_eval.jsonl when that file exists.
The demo video and the bare-head Pexels clip are always included. Missing
files are skipped. Reference boxes, when present on demo_eval rows in
training/data/ppe_video_labels.jsonl, are scored as the mean pixel distance
from each F2/F5 circle centre to the nearest positive box (0 if inside).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("SIGHTLINE_WORK_DIR", REPO / "work"))
EVAL_ROOT = WORK / "ppe_eval"
FACTORS = ("F1", "F2", "F5")
DEFAULT_CLIPS = (
    {"id": "sightline_demo", "path": "data/demo/sightline_demo.mp4", "camera_id": "demo"},
    {"id": "pexels_8987012", "path": "data/demo_extra/pexels_8987012_bare_head_grinder.mp4",
     "camera_id": "pexels_8987012"},
)


def repo_path(path: str) -> Path:
    raw = Path(os.path.expanduser(str(path)))
    if raw.is_file() or raw.is_absolute():
        return raw
    for base in (REPO, REPO / "training", Path.cwd()):
        cand = base / path
        if cand.is_file():
            return cand
    return REPO / path


def checkpoint_for_yaml(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path)


def override_yaml(checkpoint_rel: str) -> str:
    """Temp pipeline config: F1, F2 and F5 stay in the mute list's complement; escalation off."""
    return (
        "model:\n"
        f"  head_checkpoint: {checkpoint_rel}\n"
        "events:\n"
        "  muted_factors: [F6, F7, F8, F9, F10]\n"
        "escalation:\n"
        "  enabled: false\n"
    )


def assert_not_live(path: Path):
    """The running demo owns data/live. Replay writes only under the eval root."""
    live = (REPO / "data" / "live").resolve()
    try:
        p = path.resolve()
    except OSError:
        p = path
    if p == live or live in getattr(p, "parents", ()):
        raise SystemExit(f"refusing to write pipeline data under {live}")
    text = p.as_posix().replace("\\", "/")
    if text.rstrip("/").endswith("/data/live") or "/data/live/" in text:
        raise SystemExit(f"refusing to write pipeline data under data/live ({path})")


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def camera_id_for(row: dict, fallback: str) -> str:
    raw = str(row.get("camera_id") or row.get("id") or fallback)
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in raw)
    return (safe[:40] or "clip").strip("_") or "clip"


def load_demo_clips(list_path: Path | None = None) -> list[dict]:
    """Held-out replay list. The two canonical clips are always present."""
    path = list_path or (REPO / "training" / "data" / "ppe_demo_eval.jsonl")
    clips = []
    seen = set()
    for row in load_jsonl(path):
        rel = row.get("path") or row.get("file") or row.get("video") or row.get("source")
        if not rel or str(rel).startswith("rtsp"):
            continue
        item = {"id": row.get("id") or Path(str(rel)).stem, "path": str(rel),
                "camera_id": camera_id_for(row, Path(str(rel)).stem)}
        key = item["path"].replace("\\", "/")
        if key not in seen:
            seen.add(key)
            clips.append(item)
    for row in DEFAULT_CLIPS:
        key = row["path"].replace("\\", "/")
        if key not in seen:
            seen.add(key)
            clips.append(dict(row))
    return clips


def summarize_steps(steps: list[dict], thresholds: dict, factors=FACTORS) -> dict:
    """Max, p95 and fraction of raw probabilities at or above the run's thresholds."""
    import numpy as np
    out = {}
    for f in factors:
        vals = []
        for step in steps:
            probs = step.get("p") if isinstance(step, dict) else None
            if isinstance(probs, dict) and f in probs and probs[f] is not None:
                vals.append(float(probs[f]))
        thr = thresholds.get(f)
        thr_f = float(thr) if thr is not None else None
        if not vals:
            out[f] = {"n": 0, "max": None, "p95": None, "frac_above": None, "threshold": thr_f}
            continue
        out[f] = {
            "n": len(vals),
            "max": max(vals),
            "p95": float(np.percentile(vals, 95)),
            "frac_above": (sum(v >= thr_f for v in vals) / len(vals)) if thr_f is not None else None,
            "threshold": thr_f,
        }
    return out


def load_run_thresholds(data_dir: Path) -> dict:
    runs = data_dir / "runs"
    files = sorted(runs.glob("*_summary.json")) if runs.is_dir() else []
    if not files:
        return {}
    summary = json.loads(files[-1].read_text(encoding="utf-8"))
    thr = summary.get("thresholds") or {}
    return {k: float(v) for k, v in thr.items() if v is not None}


def load_steps(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def count_incidents(data_dir: Path, factors=FACTORS) -> dict:
    """Incidents per factor from event.json (one event can count for several factors)."""
    counts = {f: 0 for f in factors}
    root = Path(data_dir) / "events"
    if not root.is_dir():
        return counts
    for path in sorted(root.glob("*/event.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        have = set(rec.get("factors") or [])
        for seg in rec.get("segments") or []:
            if isinstance(seg, dict) and seg.get("factor"):
                have.add(seg["factor"])
        for f in factors:
            if f in have:
                counts[f] += 1
    return counts


def circles_by_factor(data_dir: Path, factors=("F2", "F5")) -> dict:
    """One circle per overlay step. Deduped on time and centre so a repeated write counts once."""
    found = {f: [] for f in factors}
    root = Path(data_dir) / "events"
    if not root.is_dir():
        return found
    seen = set()
    for path in sorted(root.glob("*/event.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        for ov in rec.get("overlays") or []:
            if not isinstance(ov, dict):
                continue
            f = ov.get("factor")
            circle = ov.get("circle")
            if f not in found or not isinstance(circle, dict) or "cx" not in circle:
                continue
            key = (f, ov.get("t_ms"), round(float(circle["cx"]), 2), round(float(circle["cy"]), 2))
            if key in seen:
                continue
            seen.add(key)
            found[f].append(circle)
    return found


def point_box_distance(cx: float, cy: float, box, w: float, h: float) -> float:
    """Pixel distance from a point to an xyxy box normalised on a w x h frame. 0 if inside."""
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    left, right = sorted((x0 * w, x1 * w))
    top, bottom = sorted((y0 * h, y1 * h))
    dx = max(left - cx, 0.0, cx - right)
    dy = max(top - cy, 0.0, cy - bottom)
    return math.hypot(dx, dy)


def mean_circle_distance(circles: list[dict], boxes: list) -> float | None:
    if not circles or not boxes:
        return None
    dists = []
    for circle in circles:
        if "cx" not in circle or "w" not in circle or "h" not in circle:
            continue
        w, h = float(circle["w"]), float(circle["h"])
        if w <= 0 or h <= 0:
            continue
        cx, cy = float(circle["cx"]), float(circle["cy"])
        dists.append(min(point_box_distance(cx, cy, b, w, h) for b in boxes))
    if not dists:
        return None
    return sum(dists) / len(dists)


def reference_boxes(label: dict | None) -> dict:
    """Positive F2/F5 boxes (the hazard region) pooled over the clip's windows."""
    acc = {"F2": [], "F5": []}
    if not label:
        return acc
    for win in label.get("windows") or []:
        boxes = (win or {}).get("boxes") or {}
        for factor, key in (("F2", "F2_pos"), ("F5", "F5_pos")):
            for box in boxes.get(key) or []:
                if box and len(box) >= 4:
                    acc[factor].append(list(box[:4]))
    return acc


def match_reference(labels: list[dict], clip: dict) -> dict | None:
    """demo_eval row for this clip, matched on id or path."""
    cid = clip.get("id")
    path = str(clip.get("path") or "").replace("\\", "/")
    name = Path(path).name if path else ""
    for row in labels:
        if row.get("split") not in ("demo_eval", None):
            continue
        if cid and row.get("id") == cid:
            return row
        rp = str(row.get("path") or "").replace("\\", "/")
        if path and rp and (rp == path or rp.endswith("/" + path) or path.endswith("/" + rp) or Path(rp).name == name):
            return row
    return None


def summarize_clip(steps, thresholds, data_dir: Path, reference: dict | None) -> dict:
    scores = summarize_steps(steps, thresholds)
    incidents = count_incidents(data_dir)
    circles = circles_by_factor(data_dir)
    boxes = reference_boxes(reference)
    distance = {}
    n_circles = {}
    for f in ("F2", "F5"):
        n_circles[f] = len(circles[f])
        distance[f] = mean_circle_distance(circles[f], boxes[f])
    return {"scores": scores, "incidents": incidents, "circle_distance_px": distance, "n_circles": n_circles}


def run_one(clip: dict, override: Path, out_dir: Path, python: str) -> dict:
    rel = clip["path"]
    src = repo_path(rel)
    cam = clip["camera_id"]
    rec = {"id": clip.get("id"), "path": rel, "camera_id": cam, "resolved": str(src)}
    if not src.is_file():
        rec.update({"skipped": True, "reason": "missing file"})
        print(f"skip missing {rel}", flush=True)
        return rec
    data_dir = out_dir / f"{cam}_data"
    steps_path = out_dir / f"{cam}_steps.jsonl"
    log_path = out_dir / f"{cam}_run.log"
    assert_not_live(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-m", "sightline_pipeline.run",
           "--source", str(src), "--camera-id", cam, "--config", str(override),
           "--data-dir", str(data_dir), "--fast", "--no-escalation",
           "--steps-out", str(steps_path)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    print(" ".join(cmd), flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT)
    labels = load_jsonl(REPO / "training" / "data" / "ppe_video_labels.jsonl")
    ref = match_reference(labels, clip)
    body = summarize_clip(load_steps(steps_path), load_run_thresholds(data_dir), data_dir, ref)
    rec.update({"skipped": False, "returncode": proc.returncode, "reference": bool(ref), **body})
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
        rec["error_tail"] = tail
        print(f"{cam} failed rc={proc.returncode}", flush=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay held-out clips with one head checkpoint")
    ap.add_argument("--checkpoint", required=True, help="head .pt, repo-relative or absolute")
    ap.add_argument("--name", default="", help="eval folder name (default: checkpoint stem)")
    ap.add_argument("--clips", default="", help="ppe_demo_eval.jsonl (default: training/data/ppe_demo_eval.jsonl)")
    ap.add_argument("--out-root", default=str(EVAL_ROOT))
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    ckpt = repo_path(a.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {a.checkpoint}")
    run = a.name or ckpt.stem
    out_dir = Path(a.out_root) / run
    assert_not_live(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel = checkpoint_for_yaml(ckpt)
    override = out_dir / "override.yaml"
    override.write_text(override_yaml(rel), encoding="utf-8")
    list_path = repo_path(a.clips) if a.clips else (REPO / "training" / "data" / "ppe_demo_eval.jsonl")
    clips = load_demo_clips(list_path if list_path.is_file() else Path("__missing_ppe_demo_eval__"))
    results = [run_one(clip, override, out_dir, a.python) for clip in clips]
    summary = {
        "run": run,
        "checkpoint": rel,
        "override": override.read_text(encoding="utf-8"),
        "cuda_visible_devices": "1",
        "clips": results,
        "note": (
            "Scores are raw factor probabilities from the steps jsonl. "
            "Incidents and circles are read from event.json under each clip data dir. "
            "The live tracker also mutes verify-gate factors (F2, F5) even when escalation "
            "is disabled, so those incident and circle counts stay zero until that mute is lifted. "
            "F1 still opens incidents."
        ),
    }
    dest = out_dir / "replay_summary.json"
    dest.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {dest}", flush=True)
    failed = [r for r in results if not r.get("skipped") and r.get("returncode") not in (0, None)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
