"""Build a jsonl manifest of SHWD for F2 ('visible bare head') box supervision.

SHWD (Safety Helmet Wearing Dataset, MIT) is Pascal VOC under ``VOC2028/``:

  Annotations/*.xml
  ImageSets/Main/trainval.txt
  ImageSets/Main/test.txt
  JPEGImages/

``person`` is a bare head (F2 positive). ``hat`` is a helmeted head (hard negative).
``test.txt`` is held out (never used for training or model selection). ``trainval.txt``
is split by a stable hash into train (~85%) and validation (~15%). The official
``val.txt`` is not used; those images sit inside ``trainval.txt`` and follow the hash.

Output rows:
  id, source, kind, path, width, height, split,
  bare_boxes, hat_boxes, n_bare, n_hat

Boxes are ``[x0, y0, x1, y1]`` normalized to the original image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

TRAINING = Path(__file__).resolve().parents[1]
REPO = TRAINING.parent
WORK = Path(os.environ.get("SIGHTLINE_WORK_DIR", REPO / "work"))
SPLITS = ("train", "validation", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def hash_unit(text: str) -> float:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def hash_rank(row_id: str, seed: int, salt: str) -> str:
    return hashlib.md5(f"{salt}:{seed}:{row_id}".encode("utf-8")).hexdigest()


def trainval_split(stem: str, val_frac: float = 0.15, seed: int = 0) -> str:
    """Hash split of a trainval stem. test.txt never goes through here."""
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    return "validation" if hash_unit(f"shwd:{seed}:{stem}") < val_frac else "train"


def norm_stem(token: str) -> str:
    stem = token.strip().replace("\\", "/").split("/")[-1]
    for ext in IMAGE_EXTS:
        if stem.endswith(ext):
            return stem[: -len(ext)]
    return stem


def read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing image list: {path}")
    ids = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            ids.append(norm_stem(line.split()[0]))
    if not ids:
        raise ValueError(f"empty image list: {path}")
    return ids


def find_voc(root: Path) -> Path:
    root = Path(root)
    if (root / "Annotations").is_dir() and (root / "JPEGImages").is_dir():
        return root
    for cand in (root / "VOC2028", root / "shwd" / "VOC2028"):
        if (cand / "Annotations").is_dir() and (cand / "JPEGImages").is_dir():
            return cand
    if root.is_dir():
        for cand in sorted(p for p in root.iterdir() if p.is_dir()):
            if (cand / "Annotations").is_dir() and (cand / "JPEGImages").is_dir():
                return cand
    raise FileNotFoundError(
        f"no VOC2028 tree under {root} (expected Annotations/ and JPEGImages/)"
    )


def _clip01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def parse_voc_xml(path: Path) -> dict:
    """One SHWD annotation. person -> bare_boxes, hat -> hat_boxes, xyxy normalized."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ValueError(f"{path}: {e}") from e
    size = root.find("size")
    if size is None or size.findtext("width") is None or size.findtext("height") is None:
        raise ValueError(f"{path}: missing <size><width>/<height>")
    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: bad size {width}x{height}")
    bare, hat, unknown = [], [], {}
    skipped = 0
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"{path}: <object> without <bndbox>")
        try:
            x0 = float(box.findtext("xmin")) / width
            y0 = float(box.findtext("ymin")) / height
            x1 = float(box.findtext("xmax")) / width
            y1 = float(box.findtext("ymax")) / height
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path}: bad bndbox") from e
        x0, x1 = sorted((_clip01(x0), _clip01(x1)))
        y0, y1 = sorted((_clip01(y0), _clip01(y1)))
        if x1 <= x0 or y1 <= y0:
            skipped += 1
            continue
        coords = [x0, y0, x1, y1]
        if name == "person":
            bare.append(coords)
        elif name == "hat":
            hat.append(coords)
        elif name:
            # SHWD is person/hat. A handful of other names (one image labeled 'dog')
            # are not heads, so they are not F2 boxes. The count is printed, not dropped quietly.
            unknown[name] = unknown.get(name, 0) + 1
        else:
            raise ValueError(f"{path}: object with an empty class name")
    filename = (root.findtext("filename") or "").strip() or None
    return {
        "filename": filename,
        "width": width,
        "height": height,
        "bare_boxes": bare,
        "hat_boxes": hat,
        "skipped_degenerate": skipped,
        "unknown": unknown,
    }


def find_image(jpeg_dir: Path, stem: str, xml_filename: str | None) -> Path:
    names = []
    if xml_filename:
        names.append(xml_filename)
    names.extend(stem + ext for ext in IMAGE_EXTS)
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        cand = jpeg_dir / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no image for {stem} in {jpeg_dir}")


def subsample_hash(rows: list[dict], n: int, seed: int, salt: str) -> list[dict]:
    """Deterministic cap. n <= 0 keeps every row. Order of the result is by id."""
    if n <= 0 or len(rows) <= n:
        return list(rows)
    picked = sorted(rows, key=lambda r: (hash_rank(r["id"], seed, salt), r["id"]))[:n]
    return sorted(picked, key=lambda r: r["id"])


def subsample_stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Cap ``rows`` so about half of the kept images contain >= 1 bare head.

    When one side is smaller than n/2, the rest is filled from the other side.
    n <= 0 keeps every row.
    """
    if n <= 0 or len(rows) <= n:
        return list(rows)
    pos = [r for r in rows if r["n_bare"] >= 1]
    neg = [r for r in rows if r["n_bare"] < 1]
    n_pos = min(len(pos), n // 2)
    n_neg = min(len(neg), n - n_pos)
    if n_pos + n_neg < n and len(pos) > n_pos:
        n_pos += min(n - n_pos - n_neg, len(pos) - n_pos)
    pos_s = sorted(pos, key=lambda r: (hash_rank(r["id"], seed, "bare"), r["id"]))[:n_pos]
    neg_s = sorted(neg, key=lambda r: (hash_rank(r["id"], seed, "helmet"), r["id"]))[:n_neg]
    return sorted(pos_s + neg_s, key=lambda r: r["id"])


def build_manifest(root: Path, max_train: int = 3000, max_eval: int = 1000,
                   val_frac: float = 0.15, seed: int = 0) -> list[dict]:
    voc = find_voc(Path(root))
    ann_dir = voc / "Annotations"
    jpeg_dir = voc / "JPEGImages"
    sets = voc / "ImageSets" / "Main"
    test_ids = read_ids(sets / "test.txt")
    trainval_ids = read_ids(sets / "trainval.txt")
    test_set = set(test_ids)
    # test.txt wins if an id is listed twice; it must not enter train.
    trainval_ids = [s for s in trainval_ids if s not in test_set]

    rows = []
    degenerate = 0
    unknown = {}
    seen = set()

    def add(stem: str, split: str) -> None:
        nonlocal degenerate
        if stem in seen:
            return
        seen.add(stem)
        xml_path = ann_dir / f"{stem}.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(f"missing annotation for {stem}: {xml_path}")
        parsed = parse_voc_xml(xml_path)
        degenerate += parsed["skipped_degenerate"]
        for name, n in parsed["unknown"].items():
            unknown[name] = unknown.get(name, 0) + n
        image = find_image(jpeg_dir, stem, parsed["filename"])
        rows.append({
            "id": f"shwd/{stem}",
            "source": "shwd",
            "kind": "image",
            "path": image.resolve().as_posix(),
            "width": parsed["width"],
            "height": parsed["height"],
            "split": split,
            "bare_boxes": parsed["bare_boxes"],
            "hat_boxes": parsed["hat_boxes"],
            "n_bare": len(parsed["bare_boxes"]),
            "n_hat": len(parsed["hat_boxes"]),
        })

    for stem in test_ids:
        add(stem, "test")
    for stem in trainval_ids:
        add(stem, trainval_split(stem, val_frac, seed))

    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]
    kept = (
        subsample_stratified(train, max_train, seed)
        + subsample_hash(validation, max_eval, seed, "validation")
        + subsample_hash(test, max_eval, seed, "test")
    )
    kept.sort(key=lambda r: (SPLITS.index(r["split"]), r["id"]))
    # Parsing runs on the full lists, before the caps. main() prints this count.
    build_manifest.last_degenerate = degenerate
    build_manifest.last_unknown = unknown
    build_manifest.last_pool = {
        "train": len(train), "validation": len(validation), "test": len(test),
    }
    return kept


def format_counts(rows: list[dict]) -> str:
    lines = []
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        bare = sum(1 for r in part if r["n_bare"] >= 1)
        lines.append(
            f"{split:12} {len(part):5d}  with_bare={bare:5d}  "
            f"bare_boxes={sum(r['n_bare'] for r in part):6d}  "
            f"hat_boxes={sum(r['n_hat'] for r in part):6d}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(WORK / "datasets" / "shwd"),
                    help="SHWD root (the folder that contains VOC2028/) or the VOC2028 folder itself")
    ap.add_argument("--out", default=str(TRAINING / "data" / "shwd_manifest.jsonl"))
    ap.add_argument("--max-train", type=int, default=3000,
                    help="cap on the train split after the hash split (0 = no cap); stratified ~half bare-head")
    ap.add_argument("--max-eval", type=int, default=1000,
                    help="cap on validation and on test separately (0 = no cap)")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of trainval held out as validation by hash")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    try:
        rows = build_manifest(Path(args.root), args.max_train, args.max_eval, args.val_frac, args.seed)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pool = getattr(build_manifest, "last_pool", {})
    print(f"shwd manifest: {len(rows)} rows -> {out}", flush=True)
    if pool:
        print("pool before caps: " + ", ".join(f"{k}={v}" for k, v in pool.items()), flush=True)
    print(format_counts(rows), flush=True)
    skipped = getattr(build_manifest, "last_degenerate", 0)
    if skipped:
        print(f"skipped {skipped} degenerate boxes", flush=True)
    unknown = getattr(build_manifest, "last_unknown", {})
    if unknown:
        detail = ", ".join(f"{name}={n}" for name, n in sorted(unknown.items()))
        print(f"skipped non-head boxes ({detail}); not used as F2 targets", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
