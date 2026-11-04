"""Build training/data/ppe_video_manifest.jsonl from transcoded Pexels clips.

One row per clip. Splits are by source group (pexels id // 10000), never by a
single clip, so frames from one shoot cannot land in both train and test.
Held-out demo/eval groups are split ``demo_eval`` only. Also writes
training/data/ppe_demo_eval.jsonl for replay (those Pexels clips plus the
bare-head grinder extra and the stitched demo video).

  python tools/labeling/build_ppe_manifest.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training" / "01-data"))

import ppe_video as pv  # noqa: E402

DEFAULT_CLIPS = Path("/mnt/f/models/vidlabel/clips")
DEFAULT_URLS = Path("/mnt/f/models/vidlabel/pexels_urls.txt")
DEFAULT_OUT = REPO / "training" / "data" / "ppe_video_manifest.jsonl"
DEFAULT_DEMO = REPO / "training" / "data" / "ppe_demo_eval.jsonl"
DEMO_EXTRA = REPO / "data" / "demo_extra" / "pexels_8987012_bare_head_grinder.mp4"
DEMO_VIDEO = REPO / "data" / "demo" / "sightline_demo.mp4"
DEMO_ATTR = (
    "Held-out test footage. Voxel51 Safe_and_Unsafe_Behaviours "
    "(Önal & Dandıl 2024, CC BY 4.0) and Fraunhofer IPA table-saw frames "
    "(Steckhan et al. 2026, Zenodo, CC BY 4.0). See data/demo/ATTRIBUTION.json."
)


def clip_ids(clips_dir: Path) -> dict[int, Path]:
    found = {}
    if not clips_dir.is_dir():
        return found
    for path in sorted(clips_dir.glob("pexels_*.mp4")):
        stem = path.name[len("pexels_"):-len(".mp4")]
        if stem.isdigit() and ".tmp" not in path.name:
            found[int(stem)] = path
    return found


def replay_row(item_id: str, path: Path, group: int, license_name: str, attribution: str, url: str) -> dict:
    row = {
        "id": item_id,
        "source": "demo_replay",
        "kind": "video",
        "path": str(path.relative_to(REPO)).replace("\\", "/") if path.is_relative_to(REPO) else str(path),
        "split": "demo_eval",
        "fps": None,
        "frames": None,
        "width": None,
        "height": None,
        "license": license_name,
        "attribution": attribution,
        "url": url,
        "sha256": None,
        "bytes": None,
        "group": group,
        "role": "replay_eval",
    }
    if path.is_file():
        probed = pv.probe_video(path)
        row.update(probed)
        row["sha256"] = pv.file_sha256(path)
        row["bytes"] = path.stat().st_size
    else:
        print(f"WARNING: replay file missing, listed anyway: {path}", flush=True)
    return row


def build_rows(clips_dir: Path, urls_path: Path) -> tuple[list[dict], list[str]]:
    clips = clip_ids(clips_dir)
    if not clips:
        raise SystemExit(f"no transcoded clips in {clips_dir} — run tools/labeling/prep_clips.py first")
    urls = pv.load_url_list(urls_path)
    errors = []
    probed = {}
    for pexels_id, path in clips.items():
        try:
            probed[pexels_id] = (pv.probe_video(path), pv.file_sha256(path))
        except Exception as exc:
            errors.append(f"pexels/{pexels_id}: {exc}")
            print(f"ERROR pexels/{pexels_id}: {exc}", flush=True)
    usable = {pexels_id: clips[pexels_id] for pexels_id in probed}
    sizes: dict[int, int] = {}
    for pexels_id in usable:
        sizes[pv.pexels_group(pexels_id)] = sizes.get(pv.pexels_group(pexels_id), 0) + 1
    splits = pv.assign_group_splits(sizes)
    rows = []
    for pexels_id in sorted(usable):
        group = pv.pexels_group(pexels_id)
        meta, digest = probed[pexels_id]
        rows.append(pv.manifest_row(
            pexels_id, usable[pexels_id], splits[group], urls.get(pexels_id) or pv.page_url(pexels_id), meta, digest,
        ))
    leaked = pv.leakage_errors(rows)
    if leaked:
        raise SystemExit("split leakage:\n" + "\n".join(leaked))
    counts = pv.split_counts(rows)
    non_demo = sum(n for split, n in counts.items() if split != "demo_eval")
    if non_demo >= pv.MIN_TEST_CLIPS and counts.get("test", 0) < pv.MIN_TEST_CLIPS:
        raise SystemExit(f"test split has {counts.get('test', 0)} clips; need at least {pv.MIN_TEST_CLIPS}")
    return rows, errors


def demo_eval_rows(manifest_rows: list[dict]) -> list[dict]:
    rows = [dict(row) for row in manifest_rows if row["split"] == "demo_eval"]
    rows.append(replay_row(
        "demo_extra/pexels_8987012",
        DEMO_EXTRA,
        pv.pexels_group(8987012),
        pv.LICENSE,
        pv.page_url(8987012),
        pv.page_url(8987012),
    ))
    rows.append(replay_row(
        "demo/sightline_demo",
        DEMO_VIDEO,
        -1,
        "CC-BY-4.0",
        DEMO_ATTR,
        "https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours",
    ))
    return rows


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Write the PPE video manifest and the demo-eval list")
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--urls", type=Path, default=DEFAULT_URLS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--demo-out", type=Path, default=DEFAULT_DEMO)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows, errors = build_rows(args.clips_dir, args.urls)
    pv.write_jsonl(args.out, rows)
    demo_rows = demo_eval_rows(rows)
    pv.write_jsonl(args.demo_out, demo_rows)
    counts = pv.split_counts(rows)
    print(f"wrote {len(rows)} clips -> {args.out}")
    for split in ("train", "val", "test", "demo_eval"):
        print(f"  {split:<10} {counts.get(split, 0)}")
    groups = {}
    for row in rows:
        groups.setdefault(row["split"], set()).add(row["group"])
    for split, group_ids in sorted(groups.items()):
        print(f"  groups {split}: {sorted(group_ids)}")
    print(f"wrote {len(demo_rows)} demo-eval rows -> {args.demo_out}")
    if errors:
        print(f"{len(errors)} clip(s) skipped because probe failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
