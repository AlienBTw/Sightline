#!/usr/bin/env python3
"""Phase 01 - download and verify the Sightline training datasets.

Sources (both CC BY 4.0):
  voxel51  Voxel51/Safe_and_Unsafe_Behaviours on Hugging Face (691 MP4 clips, ~10 GB)
  zenodo   Fraunhofer IPA table-saw hand set, Zenodo record 18936355
           (one data.tar.gz split into 39 parts, ~38.6 GiB, YOLO boxes for "Hand")

What it does
  * lists the real files from the HF tree API / Zenodo API (no hard-coded names)
  * resumable downloads (.part files + HTTP Range), retries with backoff
  * verifies every file: sha256 (HF git-lfs oid) or md5 (Zenodo) + size
  * zenodo: stream-extracts the downloaded parts in order; with --zenodo-parts N
    only the first N parts are fetched and the files fully contained in them are extracted
  * writes data/manifest.jsonl (one line per clip/image: source, licence, path, split, label...)
  * fails loudly: any checksum mismatch, HTTP failure or missing file -> exit code 1

Usage (from the repo root):
  python training/01-data/download.py --dry-run
  python training/01-data/download.py --sources voxel51 --limit 5
  python training/01-data/download.py --sources voxel51
  python training/01-data/download.py --sources zenodo --zenodo-parts 1
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

TRAINING_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = TRAINING_DIR / "data"
TUNIS = timezone(timedelta(hours=1), "Africa/Tunis")  # Tunisia has no DST

HF_REPO = "Voxel51/Safe_and_Unsafe_Behaviours"
HF_REVISION = "7369ff1830b5f6c00f7d7383eeeaa90704bfc5b5"  # pinned 2026-10-30
HF_LICENSE = "CC-BY-4.0"
HF_ATTRIBUTION = (
    "Önal & Dandıl (2024), Video dataset for the detection of safe and unsafe behaviours "
    "in workplaces, Data in Brief 56:110756, doi:10.1016/j.dib.2024.110756 (Voxel51 FiftyOne export)"
)
HF_EXPECTED_CLIPS = 691

ZENODO_RECORD = "18936355"
ZENODO_LICENSE = "CC-BY-4.0"
ZENODO_ATTRIBUTION = (
    "Steckhan, Prajapati, Shao, Heinle, Vock (2026), Dataset for Hand Detection Near a Sliding "
    "Table Saw in Safety-Critical Woodworking Scenarios, Fraunhofer IPA, doi:10.5281/zenodo.18936355"
)
ZENODO_EXPECTED_PARTS = 39
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

USER_AGENT = "sightline-training/0.1 (+hackathon dataset downloader)"
CHUNK = 1 << 20          # read/hash block
COPY_CHUNK = 4 << 20     # socket -> disk copy block
DEFAULT_WORKERS = {"voxel51": 16, "zenodo": 4}


class DownloadError(RuntimeError):
    """Raised for anything that must stop the run (checksum, HTTP, layout)."""


def now_tunis() -> str:
    return datetime.now(TUNIS).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{datetime.now(TUNIS).strftime('%H:%M:%S')} Tunis] {msg}", flush=True)


# --------------------------------------------------------------------------- HTTP

def _request(url: str, headers: dict | None = None) -> urllib.request.Request:
    h = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN")  # optional; the dataset is public
    if token and "huggingface.co" in url:
        h["Authorization"] = f"Bearer {token}"
    if headers:
        h.update(headers)
    return urllib.request.Request(url, headers=h)


def http_json(url: str, retries: int = 4) -> tuple[object, dict]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_request(url), timeout=60) as r:
                return json.loads(r.read().decode("utf-8")), dict(r.headers)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise DownloadError(f"GET {url} failed after {retries} tries: {last}")


def _next_link(headers: dict) -> str | None:
    link = headers.get("Link") or headers.get("link")
    if not link:
        return None
    for part in link.split(","):
        seg = part.split(";")
        if len(seg) >= 2 and 'rel="next"' in seg[1]:
            return seg[0].strip().strip("<>")
    return None


def hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class RemoteFile:
    url: str
    dest: Path
    size: int
    algo: str        # "sha256" | "md5"
    digest: str


class VerifiedCache:
    """Remembers files already hashed (path -> size, mtime, digest) so reruns are fast."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()  # downloads run in a thread pool
        self.data: dict = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.data = {}

    def ok(self, f: RemoteFile) -> bool:
        if not f.dest.exists():
            return False
        st = f.dest.stat()
        with self.lock:
            rec = self.data.get(str(f.dest))
        return bool(rec and rec["size"] == st.st_size == f.size
                    and rec["mtime"] == int(st.st_mtime) and rec["digest"] == f.digest)

    def mark(self, f: RemoteFile) -> None:
        st = f.dest.stat()
        with self.lock:
            self.data[str(f.dest)] = {"size": st.st_size, "mtime": int(st.st_mtime), "digest": f.digest}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            text = json.dumps(self.data, indent=0)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.path)


def download_file(f: RemoteFile, cache: VerifiedCache, retries: int = 5) -> str:
    """Download with resume + verify. Returns 'cached' | 'verified' | 'downloaded'."""
    if cache.ok(f):
        return "cached"
    f.dest.parent.mkdir(parents=True, exist_ok=True)
    if f.dest.exists() and f.dest.stat().st_size == f.size:
        if hash_file(f.dest, f.algo) == f.digest:
            cache.mark(f)
            return "verified"
        log(f"  checksum mismatch on existing {f.dest.name}, re-downloading")
        f.dest.unlink()

    part = f.dest.with_name(f.dest.name + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have > f.size:
            part.unlink()
            have = 0
        if have == f.size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with urllib.request.urlopen(_request(f.url, headers), timeout=120) as r:
                status = getattr(r, "status", 200)
                mode = "ab" if (have and status == 206) else "wb"
                if have and status != 206:
                    log(f"  server ignored Range for {f.dest.name}; restarting from 0")
                with open(part, mode) as out:
                    shutil.copyfileobj(r, out, COPY_CHUNK)
            if part.stat().st_size == f.size:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            wait = min(60, 2 ** attempt)
            log(f"  {f.dest.name}: {e} - retry {attempt + 1}/{retries} in {wait}s (resumes)")
            time.sleep(wait)
    else:
        raise DownloadError(f"download failed for {f.url}: {last}")

    got = part.stat().st_size
    if got != f.size:
        raise DownloadError(f"size mismatch for {f.dest.name}: got {got}, expected {f.size}")
    digest = hash_file(part, f.algo)
    if digest != f.digest:
        part.unlink()
        raise DownloadError(
            f"{f.algo} mismatch for {f.dest.name}: got {digest}, expected {f.digest} (partial file deleted)")
    os.replace(part, f.dest)
    cache.mark(f)
    return "downloaded"


def download_many(files: list[RemoteFile], cache: VerifiedCache, workers: int, label: str) -> tuple[dict, list[str]]:
    """Download/verify files in a thread pool. Each file keeps its own resume + checksum.
    Returns (counts, errors). Prints throughput every ~25 files."""
    counts = {"cached": 0, "verified": 0, "downloaded": 0}
    errors: list[str] = []
    total = len(files)
    t0 = time.time()
    got_bytes = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(download_file, f, cache): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            f = futs[fut]
            try:
                kind = fut.result()
                counts[kind] += 1
                if kind == "downloaded":
                    got_bytes += f.size
            except DownloadError as e:
                errors.append(str(e))
                log(f"  ERROR {e}")
            if i % 25 == 0 or i == total:
                cache.save()
                dt = max(time.time() - t0, 1e-6)
                log(f"  {label} {i}/{total} {counts} {fmt_bytes(got_bytes)} new in {dt:.0f}s "
                    f"({fmt_bytes(got_bytes / dt)}/s, {workers} workers)")
    return counts, errors


# --------------------------------------------------------------------------- Voxel51 (HF)

def hf_list_files(repo: str, revision: str) -> list[dict]:
    url = (f"https://huggingface.co/api/datasets/{repo}/tree/{revision}"
           f"?recursive=true&expand=false")
    files: list[dict] = []
    while url:
        page, headers = http_json(url)
        if not isinstance(page, list):
            raise DownloadError(f"unexpected HF tree response: {str(page)[:200]}")
        files.extend(x for x in page if x.get("type") == "file")
        url = _next_link(headers)
    return files


def hf_url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"


def voxel51_plan(root: Path, revision: str) -> tuple[list[RemoteFile], list[RemoteFile]]:
    """Returns (metadata files, clip files)."""
    listing = hf_list_files(HF_REPO, revision)
    meta, clips = [], []
    for x in listing:
        p = x["path"]
        if p.endswith(".mp4"):
            lfs = x.get("lfs")
            if not lfs:
                raise DownloadError(f"{p} has no LFS sha256 in the HF listing - cannot verify")
            clips.append(RemoteFile(hf_url(HF_REPO, revision, p), root / p, int(lfs["size"]),
                                    "sha256", lfs["oid"]))
        elif p in ("samples.json", "README.md"):
            # small git (non-LFS) files: verify by size; git blob oid is sha1 of a header+content
            clips_or_meta = RemoteFile(hf_url(HF_REPO, revision, p), root / p, int(x["size"]),
                                       "git-blob", x["oid"])
            meta.append(clips_or_meta)
    clips.sort(key=lambda r: r.dest.name)
    if len(clips) != HF_EXPECTED_CLIPS:
        log(f"WARNING: HF listing has {len(clips)} clips, expected {HF_EXPECTED_CLIPS}")
    if not any(m.dest.name == "samples.json" for m in meta):
        raise DownloadError("samples.json (labels + splits) not found in the HF listing")
    return meta, clips


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download_git_file(f: RemoteFile) -> None:
    if f.dest.exists() and git_blob_sha1(f.dest) == f.digest:
        return
    f.dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_request(f.url), timeout=60) as r:
        data = r.read()
    tmp = f.dest.with_suffix(f.dest.suffix + ".tmp")
    tmp.write_bytes(data)
    if git_blob_sha1(tmp) != f.digest:
        tmp.unlink()
        raise DownloadError(f"git blob sha1 mismatch for {f.dest.name}")
    os.replace(tmp, f.dest)


def merge_hints(labels: Iterable[str], hints: dict) -> dict:
    """Factor hints for a clip published under several classes (byte-identical duplicates).
    Multiple-instance view: if any class says a factor is present, the clip is positive."""
    out: dict[str, str] = {}
    for lab in labels:
        for fid, state in (hints.get(lab) or {}).items():
            if out.get(fid) != "present":
                out[fid] = state
    return out


def voxel51_records(root: Path, clips: list[RemoteFile], hints: dict, cache: VerifiedCache) -> list[dict]:
    """One record per clip that is on disk and verified. `clips` must be the FULL listing
    so byte-identical duplicates (the HF export has some, even across train/test) are found."""
    samples_path = root / "samples.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))["samples"]
    by_path = {s["filepath"]: s for s in samples}
    by_digest: dict[str, list[str]] = {}
    for c in clips:
        by_digest.setdefault(c.digest, []).append(c.dest.stem)
    out = []
    for c in clips:
        if not cache.ok(c):
            continue
        rel = c.dest.relative_to(root).as_posix()
        s = by_path.get(rel)
        if s is None:
            raise DownloadError(f"{rel} has no entry in samples.json")
        label = (s.get("ground_truth") or {}).get("label")
        split = "test" if "test" in s.get("tags", []) else "train" if "train" in s.get("tags", []) else None
        md = s.get("metadata") or {}
        dups = [d for d in by_digest[c.digest] if d != c.dest.stem]
        dup_labels = sorted({(by_path.get(f"data/{d}.mp4") or {}).get("ground_truth", {}).get("label")
                             for d in dups} - {None, label})
        out.append({
            "id": f"voxel51/{c.dest.stem}",
            "source": "voxel51_safe_unsafe",
            "kind": "video",
            "path": c.dest.relative_to(TRAINING_DIR).as_posix() if c.dest.is_relative_to(TRAINING_DIR) else str(c.dest),
            "split": split,
            "source_label": label,
            "factor_hints": merge_hints([label, *dup_labels], hints),  # merged over duplicates
            "duplicate_of": [f"voxel51/{d}" for d in dups],
            "duplicate_labels": dup_labels,   # same bytes published under another class
            "sha256": c.digest,
            "bytes": c.size,
            "fps": md.get("frame_rate"),
            "frames": md.get("total_frame_count"),
            "width": md.get("frame_width"),
            "height": md.get("frame_height"),
            "license": HF_LICENSE,
            "attribution": HF_ATTRIBUTION,
            "url": c.url,
        })
    return out


# --------------------------------------------------------------------------- Zenodo

def zenodo_plan(root: Path, record: str) -> list[RemoteFile]:
    rec, _ = http_json(f"https://zenodo.org/api/records/{record}")
    lic = ((rec.get("metadata") or {}).get("license") or {}).get("id", "")
    if lic.lower() != "cc-by-4.0":
        raise DownloadError(f"Zenodo record licence is '{lic}', expected cc-by-4.0 - stopping")
    parts = []
    for f in rec.get("files", []):
        algo, _, digest = f["checksum"].partition(":")
        parts.append(RemoteFile(f["links"]["self"], root / "parts" / f["key"], int(f["size"]), algo, digest))
    parts.sort(key=lambda r: r.dest.name)
    if len(parts) != ZENODO_EXPECTED_PARTS:
        log(f"WARNING: Zenodo lists {len(parts)} parts, expected {ZENODO_EXPECTED_PARTS}")
    return parts


class ChainedReader(io.RawIOBase):
    """Reads several files back to back as one stream (split archive parts)."""

    def __init__(self, paths: Iterable[Path]):
        self._paths = list(paths)
        self._fh = None
        self._i = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while True:
            if self._fh is None:
                if self._i >= len(self._paths):
                    return 0
                self._fh = open(self._paths[self._i], "rb")
                self._i += 1
            n = self._fh.readinto(b)
            if n:
                return n
            self._fh.close()
            self._fh = None

    def close(self) -> None:
        if self._fh:
            self._fh.close()
        super().close()


def safe_member_path(dest_root: Path, name: str) -> Path:
    target = (dest_root / name).resolve()
    if not target.is_relative_to(dest_root.resolve()):
        raise DownloadError(f"unsafe path in archive: {name}")
    return target


COMPLETE_MARKER = ".extraction_complete.json"


def extract_split_tar(parts: list[Path], dest_root: Path, complete: bool,
                      stats: dict | None = None) -> tuple[int, int, bool]:
    """Stream-extract a split .tar.gz. Returns (written, skipped, truncated).
    If `stats` is given it is filled with {"members": n_files, "bytes": total} for the files seen."""
    dest_root.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    seen_files = seen_bytes = 0
    truncated = False
    stream = io.BufferedReader(ChainedReader(parts), CHUNK)
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for m in tar:
                if not (m.isfile() or m.isdir()):
                    continue  # no links/devices
                target = safe_member_path(dest_root, m.name)
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                seen_files += 1
                seen_bytes += m.size
                if target.exists() and target.stat().st_size == m.size:
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".tmp")
                src = tar.extractfile(m)
                with open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out, CHUNK)
                if tmp.stat().st_size != m.size:
                    tmp.unlink()
                    raise tarfile.ReadError("short member")
                os.replace(tmp, target)
                written += 1
    except (tarfile.ReadError, EOFError, OSError) as e:
        if complete:
            raise DownloadError(f"archive extraction failed: {e}")
        truncated = True  # expected when only a prefix of the parts was downloaded
    finally:
        stream.close()
    for leftover in dest_root.rglob("*.tmp"):
        leftover.unlink()
    if stats is not None:
        stats.update({"members": seen_files, "bytes": seen_bytes})
    return written, skipped, truncated


def write_complete_marker(dest_root: Path, parts: list[RemoteFile], stats: dict) -> Path:
    """Written only after a full (all parts) extraction finished with every member on disk."""
    on_disk = sum(1 for p in dest_root.rglob("*") if p.is_file() and p.name != COMPLETE_MARKER)
    if on_disk != stats["members"]:
        raise DownloadError(f"extraction check failed: archive has {stats['members']} files, "
                            f"{on_disk} on disk under {dest_root}")
    marker = dest_root / COMPLETE_MARKER
    marker.write_text(json.dumps({"checked_at": now_tunis(), "members": stats["members"], "bytes": stats["bytes"],
                                  "parts": {p.dest.name: f"{p.algo}:{p.digest}" for p in parts}}, indent=1),
                      encoding="utf-8")
    return marker


def delete_parts_if_safe(dest_root: Path, parts: list[RemoteFile]) -> int:
    """Delete archive parts only if the completion marker lists exactly these parts."""
    marker = dest_root / COMPLETE_MARKER
    if not marker.exists():
        raise DownloadError("refusing to delete parts: no extraction-complete marker (run a full extraction first)")
    listed = json.loads(marker.read_text(encoding="utf-8"))["parts"]
    if listed != {p.dest.name: f"{p.algo}:{p.digest}" for p in parts}:
        raise DownloadError("refusing to delete parts: marker does not match the current part list")
    freed = 0
    for p in parts:
        if p.dest.exists():
            freed += p.dest.stat().st_size
            p.dest.unlink()
    return freed


def zenodo_records(extract_root: Path) -> list[dict]:
    out = []
    for img in sorted(extract_root.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS or not img.is_file():
            continue
        label = img.with_suffix(".txt")
        parent = img.parent.name  # YOLO 1.1 export: obj_<Split>_data
        split = None
        if parent.startswith("obj_") and parent.endswith("_data"):
            split = parent[4:-5].lower()
        n_boxes = None
        if label.exists():
            n_boxes = sum(1 for ln in label.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip())
        out.append({
            # frame names repeat across export folders, so the id carries the relative path
            "id": "zenodo_tablesaw/" + img.relative_to(extract_root).with_suffix("").as_posix(),
            "source": "zenodo_tablesaw",
            "kind": "image",
            "path": img.relative_to(TRAINING_DIR).as_posix() if img.is_relative_to(TRAINING_DIR) else str(img),
            "label_path": (label.relative_to(TRAINING_DIR).as_posix() if label.exists() and label.is_relative_to(TRAINING_DIR) else (str(label) if label.exists() else None)),
            "label_format": "yolo1.1 (class 0 = Hand, normalized cx cy w h)",
            "split": split,
            "source_label": "hand_boxes",
            "n_hand_boxes": n_boxes,   # None = no label file found (meaning not confirmed yet)
            "factor_hints": {},
            "bytes": img.stat().st_size,
            "license": ZENODO_LICENSE,
            "attribution": ZENODO_ATTRIBUTION,
            "url": f"https://doi.org/10.5281/zenodo.{ZENODO_RECORD}",
        })
    return out


# --------------------------------------------------------------------------- manifest

def write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_hints() -> dict:
    cfg = TRAINING_DIR / "configs" / "factors.yaml"
    try:
        import yaml  # type: ignore
    except ImportError:
        log("PyYAML not installed - factor_hints left empty")
        return {}
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    return (data.get("dataset_hints") or {}).get("voxel51_safe_unsafe") or {}


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def check_disk(path: Path, need: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < need * 1.05:
        raise DownloadError(f"not enough disk at {path}: need ~{fmt_bytes(need)}, free {fmt_bytes(free)}")


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", choices=["voxel51", "zenodo"], default=["voxel51"])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--limit", type=int, default=None, help="voxel51: only the first N clips (smoke test)")
    ap.add_argument("--hf-revision", default=HF_REVISION)
    ap.add_argument("--zenodo-parts", default="all",
                    help="'all' or N = download only the first N of the 39 parts (~1 GiB each)")
    ap.add_argument("--workers", type=int, default=None,
                    help=f"parallel downloads (default {DEFAULT_WORKERS['voxel51']} for voxel51, "
                         f"{DEFAULT_WORKERS['zenodo']} for zenodo parts)")
    ap.add_argument("--delete-parts", action="store_true",
                    help="zenodo: after a full, verified extraction delete the archive parts (frees ~38.6 GiB; "
                         "a later re-extraction would need a re-download)")
    ap.add_argument("--no-extract", action="store_true", help="zenodo: download parts only")
    ap.add_argument("--dry-run", action="store_true", help="list what would be downloaded, write nothing")
    args = ap.parse_args(argv)

    data_dir: Path = args.data_dir.resolve()
    manifest_path = data_dir / "manifest.jsonl"
    cache = VerifiedCache(data_dir / ".verified.json")
    errors: list[str] = []
    records: list[dict] = []

    # keep records of sources not requested this run
    if manifest_path.exists() and not args.dry_run:
        wanted = {"voxel51": "voxel51_safe_unsafe", "zenodo": "zenodo_tablesaw"}
        skip = {wanted[s] for s in args.sources}
        for ln in manifest_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                if r.get("source") not in skip:
                    records.append(r)

    if "voxel51" in args.sources:
        root = data_dir / "raw" / "voxel51_safe_unsafe"
        log(f"voxel51: listing {HF_REPO}@{args.hf_revision[:10]}")
        meta, all_clips = voxel51_plan(root, args.hf_revision)
        clips = all_clips
        if args.limit is not None:
            clips = clips[: args.limit]
        total = sum(c.size for c in clips)
        log(f"voxel51: {len(clips)} clips, {fmt_bytes(total)} -> {root}")
        if args.dry_run:
            for c in clips[:10]:
                print(f"  {c.dest.name}  {fmt_bytes(c.size)}  sha256={c.digest[:12]}...")
            if len(clips) > 10:
                print(f"  ... and {len(clips) - 10} more")
        else:
            check_disk(root, total)
            for m in meta:
                download_git_file(m)
            _, errs = download_many(clips, cache, args.workers or DEFAULT_WORKERS["voxel51"], "voxel51")
            errors.extend(errs)
            records.extend(voxel51_records(root, all_clips, load_hints(), cache))

    if "zenodo" in args.sources:
        root = data_dir / "raw" / "zenodo_tablesaw"
        log(f"zenodo: reading record {ZENODO_RECORD}")
        parts = zenodo_plan(root, ZENODO_RECORD)
        n = len(parts) if args.zenodo_parts == "all" else int(args.zenodo_parts)
        if not 1 <= n <= len(parts):
            raise DownloadError(f"--zenodo-parts must be 1..{len(parts)} or 'all'")
        chosen = parts[:n]
        total = sum(p.size for p in chosen)
        log(f"zenodo: {n}/{len(parts)} parts, {fmt_bytes(total)} (+ about the same again once extracted)")
        if args.dry_run:
            for p in chosen[:5]:
                print(f"  {p.dest.name}  {fmt_bytes(p.size)}  md5={p.digest}")
        else:
            check_disk(root, total * (1 if args.no_extract else 2))
            _, errs = download_many(chosen, cache, args.workers or DEFAULT_WORKERS["zenodo"], "zenodo")
            errors.extend(errs)  # any failed part blocks extraction (the stream needs every part in order)
            if not args.no_extract and not errors:
                complete = n == len(parts)
                log(f"zenodo: extracting {n} part(s){'' if complete else ' (prefix only)'}")
                stats: dict = {}
                w, s, trunc = extract_split_tar([p.dest for p in chosen], root / "extracted", complete, stats)
                log(f"zenodo: {w} files written, {s} already present, {stats['members']} files seen in the archive"
                    + (", stopped at end of downloaded parts (expected for a prefix)" if trunc else ""))
                if complete:
                    mk = write_complete_marker(root / "extracted", parts, stats)
                    log(f"zenodo: extraction verified complete ({stats['members']} files, "
                        f"{fmt_bytes(stats['bytes'])}) -> {mk.name}")
                    if args.delete_parts:
                        log(f"zenodo: deleted verified parts, freed {fmt_bytes(delete_parts_if_safe(root / 'extracted', parts))}")
            recs = zenodo_records(root / "extracted")
            log(f"zenodo: {len(recs)} images in manifest")
            records.extend(recs)

    if args.dry_run:
        log("dry run - nothing written")
        return 0

    cache.save()
    write_manifest(manifest_path, records)
    log(f"manifest: {len(records)} records -> {manifest_path}")
    if errors:
        log(f"FAILED: {len(errors)} error(s). Re-run the same command to resume.")
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    log("done, all files verified")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DownloadError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("interrupted - re-run the same command to resume", file=sys.stderr)
        sys.exit(130)
