"""Re-verify the full Zenodo extraction, then delete only the archive parts.

Why a separate script: `download.py --sources zenodo --delete-parts` re-runs check_disk(total * 2),
which needs ~81 GiB free even when every part is already on disk, so it refuses on a tight C: drive.

Steps (fails loud, deletes nothing unless every check passes):
  1. every part is on disk with the size from the Zenodo record and is in the md5-verified cache
  2. the whole split archive is streamed again (gzip CRC checked); every member must already exist
     on disk with the right size (0 written), and the member count must match the existing marker
  3. the extraction-complete marker is rewritten, then delete_parts_if_safe() removes the parts only.
The extracted images, labels and data/manifest.jsonl are never touched.
"""
import importlib.util, json, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("download", HERE / "download.py")
dl = importlib.util.module_from_spec(spec); sys.modules["download"] = dl; spec.loader.exec_module(dl)

def main() -> int:
    data_dir = dl.DEFAULT_DATA_DIR.resolve()
    root = data_dir / "raw" / "zenodo_tablesaw"
    ext = root / "extracted"
    marker = ext / dl.COMPLETE_MARKER
    if not marker.exists():
        raise dl.DownloadError("no extraction-complete marker; run the full download first")
    old = json.loads(marker.read_text(encoding="utf-8"))
    free0 = shutil.disk_usage(root).free
    dl.log(f"cleanup: free before {dl.fmt_bytes(free0)}")
    parts = dl.zenodo_plan(root, dl.ZENODO_RECORD)
    cache = dl.VerifiedCache(data_dir / ".verified.json")
    for p in parts:
        if not p.dest.exists() or p.dest.stat().st_size != p.size:
            raise dl.DownloadError(f"part missing or wrong size: {p.dest.name}")
        if not cache.ok(p):
            raise dl.DownloadError(f"part not in the verified cache: {p.dest.name}")
    dl.log(f"cleanup: {len(parts)} parts present, sizes match, md5-verified cache OK")
    t = time.time(); stats: dict = {}
    w, s, trunc = dl.extract_split_tar([p.dest for p in parts], ext, True, stats)
    dl.log(f"cleanup: re-streamed archive in {time.time()-t:.0f}s: {stats['members']} members, {s} already on disk, {w} written")
    if w or trunc or stats["members"] != old["members"]:
        raise dl.DownloadError(f"re-verification mismatch (written={w}, truncated={trunc}, members={stats['members']} vs marker {old['members']}); nothing deleted")
    dl.write_complete_marker(ext, parts, stats)
    freed = dl.delete_parts_if_safe(ext, parts)
    free1 = shutil.disk_usage(root).free
    dl.log(f"cleanup: deleted {len(parts)} parts, freed {dl.fmt_bytes(freed)}; free after {dl.fmt_bytes(free1)}")
    left = list((root / "parts").glob("*"))
    dl.log(f"cleanup: files left in parts/: {len(left)}; images in extracted/: {sum(1 for x in ext.rglob('*') if x.suffix.lower()=='.png')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
