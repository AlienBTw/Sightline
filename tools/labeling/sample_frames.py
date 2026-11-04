"""Save three JPEG keyframes for every feature window of a manifest clip.

Windows are ``extract_features.window_starts`` / ``window_indices`` (16 frames,
step 4, at most 8). Keyframes are the first, middle, and last index of that
window. JPEGs are longest-side <= 1280 and land in
``<SIGHTLINE_WORK_DIR>/vidlabel/frames/<safe id>/w<k>_<j>.jpg``. An index jsonl records
the frame numbers so SAM and Muse label the same windows the encoder will see.

  python tools/labeling/sample_frames.py --source ppe_video
  python tools/labeling/sample_frames.py --source voxel51_safe_unsafe --split train
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("SIGHTLINE_WORK_DIR", REPO / "work"))
sys.path.insert(0, str(REPO / "training" / "01-data"))
sys.path.insert(0, str(REPO / "training" / "03-features"))
sys.path.insert(0, str(REPO / "tools" / "labeling"))

import extract_features as ef  # noqa: E402
import ppe_rules  # noqa: E402
import ppe_video as pv  # noqa: E402

DEFAULT_FRAMES = WORK / "vidlabel" / "frames"
DEFAULT_PPE = REPO / "training" / "data" / "ppe_video_manifest.jsonl"
DEFAULT_VOXEL = REPO / "training" / "data" / "manifest.jsonl"
DEFAULT_WEAK = REPO / "training" / "02-labeling" / "outputs" / "weak_labels_v2_vllm.jsonl"
MAX_SIDE = 1280


def resolve_media(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    for root in (REPO / "training", REPO):
        candidate = root / path_str
        if candidate.is_file():
            return candidate
    return path


def select_rows(source: str, splits: set[str] | None, manifest: Path | None,
                weak_labels: Path | None) -> list[dict]:
    if source == pv.SOURCE:
        path = manifest or DEFAULT_PPE
        rows = [row for row in pv.read_jsonl(path) if row.get("source") == pv.SOURCE]
    elif source in (pv.VOXEL_SOURCE, "voxel51"):
        path = manifest or DEFAULT_VOXEL
        manifest_rows = pv.read_jsonl(path)
        weak = pv.read_jsonl(weak_labels) if weak_labels and weak_labels.is_file() else None
        if weak is None and DEFAULT_WEAK.is_file() and manifest is None:
            weak = pv.read_jsonl(DEFAULT_WEAK)
        rows = pv.voxel_rows_for_labeling(manifest_rows, weak)
    else:
        raise SystemExit(f"unknown --source {source} (use {pv.SOURCE} or {pv.VOXEL_SOURCE})")
    if splits:
        rows = [row for row in rows if row.get("split") in splits]
    return rows


def resize_long(image, max_side: int):
    import cv2
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def read_frames(path: Path, indices: list[int]) -> dict[int, object]:
    """Grab the same way as extract_features.read_video_windows so indices match."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    need = sorted(set(indices))
    got = {}
    if not need:
        cap.release()
        return got
    want = set(need)
    idx = 0
    while idx <= need[-1]:
        if not cap.grab():
            break
        if idx in want:
            ok, frame = cap.retrieve()
            if ok:
                got[idx] = frame
        idx += 1
    cap.release()
    return got


def windows_for(n_frames: int) -> list[tuple[int, list[int], list[int]]]:
    starts = ef.window_starts(n_frames)
    out = []
    for start in starts:
        indices = ef.window_indices(start, n_frames)
        out.append((start, indices, ppe_rules.keyframe_indices(indices)))
    return out


def sample_clip(row: dict, frames_root: Path, max_side: int) -> list[dict]:
    import cv2
    media = resolve_media(row["path"])
    if not media.is_file():
        raise RuntimeError(f"missing media {row['path']}")
    probed = pv.probe_video(media)
    n_frames = int(probed["frames"])
    safe = ef.item_key(row["id"])
    folder = frames_root / safe
    folder.mkdir(parents=True, exist_ok=True)
    planned = windows_for(n_frames)
    wanted = []
    records = []
    for w_i, (start, _indices, keys) in enumerate(planned):
        paths = [folder / f"w{w_i}_{j}.jpg" for j in range(len(keys))]
        missing = [k for k, path in zip(keys, paths) if not path.is_file() or path.stat().st_size < 32]
        wanted.extend(missing)
        records.append({
            "id": row["id"],
            "key": safe,
            "source": row.get("source"),
            "split": row.get("split"),
            "window": w_i,
            "start": int(start),
            "frame_indices": [int(k) for k in keys],
            "paths": [str(p) for p in paths],
            "width": None,
            "height": None,
            "n_frames": n_frames,
        })
    decoded = read_frames(media, wanted) if wanted else {}
    for record, (_start, _indices, keys) in zip(records, planned):
        paths = [Path(p) for p in record["paths"]]
        for key, path in zip(keys, paths):
            if path.is_file() and path.stat().st_size >= 32:
                continue
            frame = decoded.get(key)
            if frame is None:
                raise RuntimeError(f"{row['id']} missing frame {key}")
            image = resize_long(frame, max_side)
            if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
                raise RuntimeError(f"jpeg write failed: {path}")
        sample = cv2.imread(str(paths[0]))
        if sample is None:
            raise RuntimeError(f"cannot read back {paths[0]}")
        record["height"], record["width"] = int(sample.shape[0]), int(sample.shape[1])
    return records


def write_index(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Save 3 keyframes per V-JEPA window")
    parser.add_argument("--source", default=pv.SOURCE, help="ppe_video or voxel51_safe_unsafe")
    parser.add_argument("--split", default="", help="comma list; empty keeps every split")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--weak-labels", type=Path, default=DEFAULT_WEAK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--index", type=Path, default=None, help="default: <out-dir>/index.jsonl")
    parser.add_argument("--max-side", type=int, default=MAX_SIDE)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    splits = {part.strip() for part in args.split.split(",") if part.strip()} or None
    rows = select_rows(args.source, splits, args.manifest, args.weak_labels)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("no clips selected")
        return 0
    index_path = args.index or (args.out_dir / "index.jsonl")
    existing = pv.read_jsonl(index_path)
    by_key = {(row["id"], row["window"]): row for row in existing}
    errors = 0
    print(f"sampling {len(rows)} clip(s) -> {args.out_dir}", flush=True)
    for n, row in enumerate(rows, 1):
        try:
            records = sample_clip(row, args.out_dir, args.max_side)
        except Exception as exc:
            errors += 1
            print(f"ERROR {row.get('id')}: {exc}", flush=True)
            continue
        for record in records:
            by_key[(record["id"], record["window"])] = record
        print(f"[{n}/{len(rows)}] {row['id']} windows={len(records)}", flush=True)
    ordered = [by_key[key] for key in sorted(by_key)]
    write_index(index_path, ordered)
    print(f"index {len(ordered)} windows -> {index_path}  errors={errors}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
