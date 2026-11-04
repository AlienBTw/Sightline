"""PPE video source (Pexels clips) for F2 / F5.

Group rule: ``group = pexels_id // 10000``. Clips from the same shoot or creator
series share that bucket and must stay in one split. Groups that contain any of
the held-out demo/eval ids are ``demo_eval`` and are never train, val, or test.
Every other group gets a stable hash into about 70/15/15 train/val/test (seed 0).
If that hash leaves test with fewer than ``min_test`` clips, the non-test groups
closest to the test bucket (highest hash) move to test until the floor is met.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SOURCE = "ppe_video"
KIND = "video"
LICENSE = "Pexels License"
GROUP_DIVISOR = 10000
SPLIT_SEED = 0
MIN_TEST_CLIPS = 3
TRAIN_CUT = 0.70
VAL_CUT = 0.85

# These clips (and every other clip in the same id//10000 group) stay out of train/val/test.
DEMO_EVAL_IDS = (
    8987012,
    8987022,
    8987023,
    4292900,
    5100044,
    5100048,
    7457419,
    10817415,
)

VOXEL_SOURCE = "voxel51_safe_unsafe"
VOXEL_LICENSE = "CC-BY-4.0"
VOXEL_ATTRIBUTION = (
    "Önal & Dandıl (2024), Video dataset for the detection of safe and unsafe behaviours "
    "in workplaces, Data in Brief 56:110756, doi:10.1016/j.dib.2024.110756 (Voxel51 FiftyOne export)"
)

MANIFEST_KEYS = (
    "id", "source", "kind", "path", "split", "fps", "frames", "width", "height",
    "license", "attribution", "url", "sha256", "bytes", "group",
)


def pexels_group(pexels_id: int) -> int:
    return int(pexels_id) // GROUP_DIVISOR


def held_out_groups() -> frozenset[int]:
    return frozenset(pexels_group(i) for i in DEMO_EVAL_IDS)


def page_url(pexels_id: int) -> str:
    return f"https://www.pexels.com/video/{int(pexels_id)}/"


def group_unit(group: int, seed: int = SPLIT_SEED) -> float:
    """Stable unit interval. sha256, not Python's salted hash()."""
    digest = hashlib.sha256(f"ppe-video-group:{int(seed)}:{int(group)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def split_from_unit(unit: float) -> str:
    if unit < TRAIN_CUT:
        return "train"
    if unit < VAL_CUT:
        return "val"
    return "test"


def assign_group_splits(group_sizes: dict[int, int], seed: int = SPLIT_SEED,
                        min_test: int = MIN_TEST_CLIPS, unit_fn=None) -> dict[int, str]:
    """Map each group id to train, val, test, or demo_eval.

    ``group_sizes`` maps group -> number of clips. A held-out group is always
    ``demo_eval``. The repair only moves whole groups, so a group cannot leak
    across splits.
    """
    sizes = {int(g): int(n) for g, n in group_sizes.items()}
    held = held_out_groups()
    unit_of = unit_fn or (lambda g: group_unit(g, seed))
    out: dict[int, str] = {}
    units: dict[int, float] = {}
    for g in sizes:
        if g in held:
            out[g] = "demo_eval"
            continue
        units[g] = float(unit_of(g))
        out[g] = split_from_unit(units[g])

    def count(split: str) -> int:
        return sum(sizes[g] for g, s in out.items() if s == split)

    guard = 0
    while count("test") < min_test and guard < 10000:
        guard += 1
        donors = [g for g, s in out.items() if s in ("train", "val")]
        if not donors:
            break
        chosen = max(donors, key=lambda g: (units[g], g))
        out[chosen] = "test"
    return out


def leakage_errors(rows: list[dict]) -> list[str]:
    """Rows that put a held-out group in train/val/test, or split one group two ways."""
    held = held_out_groups()
    seen: dict[int, str] = {}
    errors = []
    for row in rows:
        group = int(row["group"])
        split = row["split"]
        if group in held and split != "demo_eval":
            errors.append(f"{row.get('id')} group {group} is held out but split is {split}")
        prev = seen.get(group)
        if prev is not None and prev != split:
            errors.append(f"group {group} is in both {prev} and {split}")
        seen[group] = split
    return errors


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def probe_video(path: Path) -> dict:
    """cv2 probe. Frame count matches extract_features.read_video_windows."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"bad probe {path}: {width}x{height} frames={frames} fps={fps}")
    return {"frames": frames, "width": width, "height": height, "fps": round(fps, 3)}


def manifest_row(pexels_id: int, path: Path, split: str, url: str, probed: dict, digest: str) -> dict:
    pid = int(pexels_id)
    return {
        "id": f"pexels/{pid}",
        "source": SOURCE,
        "kind": KIND,
        "path": str(path),
        "split": split,
        "fps": probed["fps"],
        "frames": probed["frames"],
        "width": probed["width"],
        "height": probed["height"],
        "license": LICENSE,
        "attribution": page_url(pid),
        "url": url or page_url(pid),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "group": pexels_group(pid),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dedup_voxel51(rows: list[dict]) -> list[dict]:
    """First row per sha256 (else per id). Same rule as label_glimmer.keep_duplicates off."""
    seen = set()
    kept = []
    for row in rows:
        if row.get("source") != VOXEL_SOURCE:
            continue
        key = row.get("sha256") or row["id"]
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def voxel_rows_for_labeling(manifest_rows: list[dict], weak_rows: list[dict] | None) -> list[dict]:
    """The de-duplicated Voxel51 clips phase 03/04 already use.

    When the weak-label file is present, its clip ids are the set (and order)
    extract_features.build_items reads. Otherwise fall back to the sha256 dedup.
    """
    by_id = {}
    for row in manifest_rows:
        if row.get("source") == VOXEL_SOURCE and row.get("id") not in by_id:
            by_id[row["id"]] = row
    if weak_rows:
        out = []
        for weak in weak_rows:
            clip_id = weak.get("clip_id") or weak.get("id")
            row = by_id.get(clip_id)
            if row is not None:
                out.append(row)
        return out
    return dedup_voxel51(manifest_rows)


def load_url_list(path: Path) -> dict[int, str]:
    """``pexels_urls.txt`` lines: ``<id> <url>``."""
    out: dict[int, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ident, _, url = line.partition(" ")
        ident = ident.strip()
        url = url.strip()
        if ident.isdigit():
            out[int(ident)] = url or page_url(int(ident))
    return out


def split_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    return counts
