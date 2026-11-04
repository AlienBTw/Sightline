"""Merge SAM window labels and Muse clip verdicts into training/data/ppe_video_labels.jsonl.

Covers Pexels (``ppe_video``) and the de-duplicated Voxel51 clips phase 03/04
already use. Muse, when present, sets the clip label. SAM windows are kept
only where they agree with that verdict (absent forces 0; present keeps SAM
positives and clears the rest). With SAM alone the clip label is the max over
windows. With neither, both factors are null. With Muse and no SAM, every
window inherits the clip label and has no boxes.

  python tools/labeling/merge_labels.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training" / "01-data"))
sys.path.insert(0, str(REPO / "tools" / "labeling"))

import ppe_rules  # noqa: E402
import ppe_video as pv  # noqa: E402

DEFAULT_PPE = REPO / "training" / "data" / "ppe_video_manifest.jsonl"
DEFAULT_VOXEL = REPO / "training" / "data" / "manifest.jsonl"
DEFAULT_WEAK = REPO / "training" / "02-labeling" / "outputs" / "weak_labels_v2_vllm.jsonl"
DEFAULT_SAM = Path("/mnt/f/models/vidlabel/sam_windows.jsonl")
DEFAULT_MUSE = Path("/mnt/f/models/vidlabel/muse_verdicts.jsonl")
DEFAULT_INDEX = Path("/mnt/f/models/vidlabel/frames/index.jsonl")
DEFAULT_OUT = REPO / "training" / "data" / "ppe_video_labels.jsonl"


def meta_of(row: dict) -> dict:
    return {
        "id": row["id"],
        "source": row.get("source"),
        "split": row.get("split"),
        "path": row.get("path"),
        "license": row.get("license"),
        "attribution": row.get("attribution"),
        "url": row.get("url"),
    }


def window_specs(index_rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list] = {}
    for row in index_rows:
        grouped.setdefault(row["id"], []).append({"w": int(row["window"]), "start": int(row["start"])})
    for specs in grouped.values():
        specs.sort(key=lambda spec: spec["w"])
    return grouped


def build_labels(ppe_rows, voxel_rows, sam_rows, muse_rows, specs) -> list[dict]:
    sam = {row["id"]: row for row in sam_rows}
    muse = {row["id"]: row for row in muse_rows}
    out = []
    for row in list(ppe_rows) + list(voxel_rows):
        out.append(ppe_rules.merge_record(
            meta_of(row),
            sam.get(row["id"]),
            muse.get(row["id"]),
            specs.get(row["id"]),
        ))
    return out


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Merge SAM and Muse PPE labels")
    parser.add_argument("--ppe-manifest", type=Path, default=DEFAULT_PPE)
    parser.add_argument("--voxel-manifest", type=Path, default=DEFAULT_VOXEL)
    parser.add_argument("--weak-labels", type=Path, default=DEFAULT_WEAK)
    parser.add_argument("--sam", type=Path, default=DEFAULT_SAM)
    parser.add_argument("--muse", type=Path, default=DEFAULT_MUSE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-voxel", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    ppe_rows = [row for row in pv.read_jsonl(args.ppe_manifest) if row.get("source") == pv.SOURCE]
    if args.skip_voxel:
        voxel_rows = []
    else:
        weak = pv.read_jsonl(args.weak_labels) if args.weak_labels.is_file() else None
        voxel_rows = pv.voxel_rows_for_labeling(pv.read_jsonl(args.voxel_manifest), weak)
    labels = build_labels(
        ppe_rows,
        voxel_rows,
        pv.read_jsonl(args.sam),
        pv.read_jsonl(args.muse),
        window_specs(pv.read_jsonl(args.index)),
    )
    pv.write_jsonl(args.out, labels)
    print(ppe_rules.format_summary(labels))
    print(f"wrote {len(labels)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
