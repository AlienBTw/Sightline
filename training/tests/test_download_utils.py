"""Offline tests for 01-data/download.py helpers (no network)."""
import hashlib
import io
import os
import json
import tarfile

import pytest

import download as dl


def test_chained_split_tar_full_and_prefix(tmp_path):
    src = io.BytesIO()
    with tarfile.open(fileobj=src, mode="w:gz") as t:
        for i in range(30):
            data = os.urandom(4000)  # incompressible, so a prefix really cuts the archive
            ti = tarfile.TarInfo(f"proj/obj_Train_data/frame_{i:06d}.PNG")
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    blob = src.getvalue()
    n = len(blob) // 3 + 1
    parts = []
    for k in range(3):
        p = tmp_path / f"data.tar.gz.part-a{chr(97 + k)}"
        p.write_bytes(blob[k * n:(k + 1) * n])
        parts.append(p)

    w, s, trunc = dl.extract_split_tar(parts, tmp_path / "full", complete=True)
    assert (w, s, trunc) == (30, 0, False)
    # rerun skips existing files
    assert dl.extract_split_tar(parts, tmp_path / "full", complete=True) == (0, 30, False)
    # prefix: fewer files, no exception, no .tmp leftovers
    w2, _, trunc2 = dl.extract_split_tar(parts[:1], tmp_path / "prefix", complete=False)
    assert trunc2 is True and 0 <= w2 < 30
    assert not list((tmp_path / "prefix").rglob("*.tmp"))
    recs = dl.zenodo_records(tmp_path / "full")
    assert len(recs) == 30 and recs[0]["split"] == "train" and recs[0]["license"] == "CC-BY-4.0"
    assert len({r["id"] for r in recs}) == 30


def test_incomplete_archive_fails_loudly_when_complete_expected(tmp_path):
    p = tmp_path / "p"
    p.write_bytes(b"\x1f\x8b\x08\x00garbage")
    with pytest.raises(dl.DownloadError):
        dl.extract_split_tar([p], tmp_path / "x", complete=True)


def test_rejects_path_traversal(tmp_path):
    with pytest.raises(dl.DownloadError):
        dl.safe_member_path(tmp_path, "../evil.txt")


def test_download_verifies_and_restarts_partial(tmp_path):
    payload = b"sightline" * 1000
    src = tmp_path / "src.bin"
    src.write_bytes(payload)
    dest = tmp_path / "out" / "clip.mp4"
    dest.parent.mkdir()
    (dest.parent / "clip.mp4.part").write_bytes(b"garbage-prefix")  # stale partial
    f = dl.RemoteFile(src.as_uri(), dest, len(payload), "sha256", hashlib.sha256(payload).hexdigest())
    cache = dl.VerifiedCache(tmp_path / "v.json")
    assert dl.download_file(f, cache) == "downloaded"
    assert dest.read_bytes() == payload
    assert dl.download_file(f, cache) == "cached"


def test_download_checksum_mismatch_is_loud(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"abc")
    f = dl.RemoteFile(src.as_uri(), tmp_path / "o.bin", 3, "md5", "0" * 32)
    with pytest.raises(dl.DownloadError, match="mismatch"):
        dl.download_file(f, dl.VerifiedCache(tmp_path / "v.json"))
    assert not (tmp_path / "o.bin").exists() and not (tmp_path / "o.bin.part").exists()


def test_git_blob_sha1(tmp_path):
    p = tmp_path / "a"
    p.write_bytes(b"hello\n")
    assert dl.git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a"  # `git hash-object`


def test_manifest_written_atomically(tmp_path):
    out = tmp_path / "m.jsonl"
    dl.write_manifest(out, [{"id": "a", "license": "CC-BY-4.0"}])
    assert json.loads(out.read_text(encoding="utf-8")) == {"id": "a", "license": "CC-BY-4.0"}


def test_download_many_parallel_reports_each_failure(tmp_path):
    files = []
    for i in range(12):
        payload = os.urandom(5000)
        src = tmp_path / f"src{i}.bin"
        src.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest() if i != 5 else "0" * 64  # one corrupt
        files.append(dl.RemoteFile(src.as_uri(), tmp_path / "out" / f"c{i}.mp4", len(payload), "sha256", digest))
    cache = dl.VerifiedCache(tmp_path / "v.json")
    counts, errors = dl.download_many(files, cache, workers=8, label="t")
    assert counts["downloaded"] == 11 and len(errors) == 1 and "c5.mp4" in errors[0]
    counts2, errors2 = dl.download_many(files[:5], cache, workers=8, label="t")
    assert counts2["cached"] == 5 and not errors2
    assert len(json.loads((tmp_path / "v.json").read_text())) == 11


def test_merge_hints_present_wins_over_absent():
    hints = {"Safe Walkway": {"F8": "absent"}, "Safe Walkway Violation": {"F8": "present"},
             "Opened Panel Cover": {"F9": "present"}, "Unauthorized Intervention": {}}
    assert dl.merge_hints(["Safe Walkway", "Safe Walkway Violation"], hints) == {"F8": "present"}
    assert dl.merge_hints(["Safe Walkway Violation", "Safe Walkway"], hints) == {"F8": "present"}
    assert dl.merge_hints(["Unauthorized Intervention", "Opened Panel Cover"], hints) == {"F9": "present"}
    assert dl.merge_hints(["Safe Walkway"], hints) == {"F8": "absent"}


def test_complete_marker_and_safe_delete(tmp_path):
    src = io.BytesIO()
    with tarfile.open(fileobj=src, mode="w:gz") as t:
        for i in range(5):
            data = os.urandom(3000)
            ti = tarfile.TarInfo(f"proj/obj_Validation_data/frame_{i:06d}.PNG")
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    blob = src.getvalue()
    p1, p2 = tmp_path / "data.tar.gz.part-aa", tmp_path / "data.tar.gz.part-ab"
    p1.write_bytes(blob[: len(blob) // 2])
    p2.write_bytes(blob[len(blob) // 2:])
    parts = [dl.RemoteFile("u", p, p.stat().st_size, "md5", hashlib.md5(p.read_bytes()).hexdigest()) for p in (p1, p2)]
    dest = tmp_path / "x"
    with pytest.raises(dl.DownloadError, match="no extraction-complete marker"):
        dl.delete_parts_if_safe(dest, parts)
    stats = {}
    dl.extract_split_tar([p1, p2], dest, complete=True, stats=stats)
    assert stats["members"] == 5
    (dest / "proj" / "obj_Validation_data" / "frame_000003.PNG").unlink()
    with pytest.raises(dl.DownloadError, match="extraction check failed"):
        dl.write_complete_marker(dest, parts, stats)
    dl.extract_split_tar([p1, p2], dest, complete=True, stats=stats)  # re-extract fills the gap
    dl.write_complete_marker(dest, parts, stats)
    assert dl.zenodo_records(dest)[0]["split"] == "validation"
    assert dl.delete_parts_if_safe(dest, parts) == len(blob) and not p1.exists() and not p2.exists()
