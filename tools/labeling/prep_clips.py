"""Transcode Pexels raw clips to a training size with PyAV.

Writes ``<SIGHTLINE_WORK_DIR>/vidlabel/clips/pexels_<id>.mp4``: longest side 1280,
about 25 fps (frames are dropped when the source is faster; slower sources are
not duplicated), at most 30 seconds, H.264. Safe to re-run while downloads are
still landing: a clip is skipped when the output stamp matches the source size,
mtime, and these settings. One bad file is logged and the rest continue.

  python tools/labeling/prep_clips.py
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("SIGHTLINE_WORK_DIR", REPO / "work"))
_VID = WORK / "vidlabel"
DEFAULT_RAW = _VID / "pexels"
DEFAULT_EXTRA = WORK / "vest_eval" / "src"
DEFAULT_OUT = _VID / "clips"
DEFAULT_ERRORS = _VID / "prep_errors.jsonl"

LONG_SIDE = 1280
FPS_CAP = 25
MAX_SECONDS = 30.0
STAMP_VERSION = 1


def settings() -> dict:
    return {
        "version": STAMP_VERSION,
        "long_side": LONG_SIDE,
        "fps_cap": FPS_CAP,
        "max_seconds": MAX_SECONDS,
        "codec": "h264",
    }


def fit_side(width: int, height: int, long_side: int = LONG_SIDE) -> tuple[int, int]:
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"bad size {width}x{height}")
    longest = max(width, height)
    if longest > long_side:
        scale = long_side / float(longest)
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
    width -= width % 2
    height -= height % 2
    return max(2, width), max(2, height)


def stream_rotation(stream) -> int:
    """Degrees clockwise to display the frame upright. 0 when the pixels are already oriented."""
    try:
        raw = (getattr(stream, "metadata", None) or {}).get("rotate")
        if raw not in (None, ""):
            return int(round(float(raw))) % 360
        side_data = getattr(stream, "side_data", None) or {}
        matrix = side_data.get("DISPLAYMATRIX") if isinstance(side_data, dict) else None
        if matrix is None:
            return 0
        # av stores a 3x3 display matrix as 9 int32s. Phone clips are 0/90/180/270.
        import numpy as np
        values = np.frombuffer(bytes(matrix), dtype=np.int32)
        if values.size < 9:
            return 0
        a, b = int(values[0]), int(values[1])
        if abs(a) >= abs(b) and a > 0:
            return 0
        if abs(b) > abs(a) and b < 0:
            return 90
        if abs(a) >= abs(b) and a < 0:
            return 180
        if abs(b) > abs(a) and b > 0:
            return 270
    except Exception:
        return 0
    return 0


def apply_rotation(frame, degrees: int):
    degrees = int(degrees) % 360
    if degrees == 0:
        return frame
    import av
    import cv2
    image = frame.to_ndarray(format="bgr24")
    if degrees == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif degrees == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif degrees == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return frame
    return av.VideoFrame.from_ndarray(image, format="bgr24")


def frame_time(frame, stream, index: int, fallback_fps: float) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and stream.time_base is not None:
        return float(frame.pts * stream.time_base)
    return index / fallback_fps if fallback_fps else float(index)


def encoder_name() -> str:
    import av
    names = getattr(av, "codecs_available", ())
    for name in ("libx264", "h264"):
        if name in names:
            return name
    raise RuntimeError("this PyAV build has no h264 encoder (looked for libx264 and h264)")


def stamp_path(dest: Path) -> Path:
    return dest.with_suffix(".json")


def stamp_matches(src: Path, dest: Path) -> bool:
    stamp = stamp_path(dest)
    if not dest.is_file() or dest.stat().st_size < 1024 or not stamp.is_file():
        return False
    try:
        meta = json.loads(stamp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    stat = src.stat()
    return (
        meta.get("ok") is True
        and meta.get("src_size") == stat.st_size
        and meta.get("src_mtime_ns") == stat.st_mtime_ns
        and meta.get("settings") == settings()
    )


def write_stamp(src: Path, dest: Path, info: dict) -> None:
    stat = src.stat()
    payload = {
        "ok": True,
        "src": str(src),
        "src_size": stat.st_size,
        "src_mtime_ns": stat.st_mtime_ns,
        "settings": settings(),
        **info,
    }
    stamp = stamp_path(dest)
    tmp = stamp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, stamp)


def transcode(src: Path, dest: Path, long_side: int = LONG_SIDE, fps_cap: int = FPS_CAP,
              max_seconds: float = MAX_SECONDS) -> dict:
    """One clip. Writes ``dest`` via a temp file so a crash cannot leave a half mp4 as done."""
    import av
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    container = av.open(str(src))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        src_w = int(stream.codec_context.width or stream.width)
        src_h = int(stream.codec_context.height or stream.height)
        rotation = stream_rotation(stream)
        if rotation in (90, 270):
            src_w, src_h = src_h, src_w
        out_w, out_h = fit_side(src_w, src_h, long_side)
        rate = stream.average_rate or stream.guessed_rate
        src_fps = float(rate) if rate else float(fps_cap)
        if src_fps <= 0:
            src_fps = float(fps_cap)
        out_fps = int(fps_cap) if src_fps >= fps_cap - 0.01 else max(1, int(round(src_fps)))
        time_base = Fraction(1, out_fps)
        out = av.open(str(tmp), "w")
        try:
            codec = encoder_name()
            out_stream = out.add_stream(codec, rate=out_fps)
            out_stream.width = out_w
            out_stream.height = out_h
            out_stream.pix_fmt = "yuv420p"
            out_stream.time_base = time_base
            out_stream.options = {"crf": "23", "preset": "veryfast"}
            interval = 1.0 / out_fps
            next_t = 0.0
            written = 0
            seen = 0
            max_frames = int(round(out_fps * max_seconds))
            for frame in container.decode(stream):
                timestamp = frame_time(frame, stream, seen, src_fps)
                seen += 1
                if timestamp >= max_seconds - 1e-3:
                    break
                if timestamp + 1e-4 < next_t:
                    continue
                frame = apply_rotation(frame, rotation)
                frame = frame.reformat(width=out_w, height=out_h, format="yuv420p")
                frame.pts = written
                frame.time_base = time_base
                for packet in out_stream.encode(frame):
                    out.mux(packet)
                written += 1
                next_t += interval
                if written >= max_frames:
                    break
            for packet in out_stream.encode(None):
                out.mux(packet)
        finally:
            out.close()
        if written <= 0:
            raise RuntimeError(f"no frames encoded from {src}")
    finally:
        container.close()
    os.replace(tmp, dest)
    info = {
        "frames": written,
        "fps": out_fps,
        "width": out_w,
        "height": out_h,
        "rotation": rotation,
        "src_fps": round(src_fps, 3),
    }
    write_stamp(src, dest, info)
    return info


def discover(raw_dir: Path, extra_dirs: list[Path]) -> dict[int, Path]:
    """Raw ``raw_<id>.mp4`` wins over an extra copy of the same id."""
    jobs: dict[int, Path] = {}
    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("raw_*.mp4")):
            stem = path.name[len("raw_"):-len(".mp4")]
            if stem.isdigit():
                jobs[int(stem)] = path
    for extra in extra_dirs:
        if not extra.is_dir():
            continue
        for path in sorted(extra.glob("*.mp4")):
            stem = path.stem
            if stem.isdigit():
                jobs.setdefault(int(stem), path)
    return jobs


def _worker(job: tuple) -> dict:
    pexels_id, src, dest = job
    src, dest = Path(src), Path(dest)
    try:
        if stamp_matches(src, dest):
            return {"id": pexels_id, "status": "skipped", "path": str(dest)}
        info = transcode(src, dest)
        return {"id": pexels_id, "status": "ok", "path": str(dest), **info}
    except Exception as exc:
        tmp = dest.with_suffix(".tmp.mp4")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return {
            "id": pexels_id,
            "status": "error",
            "src": str(src),
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=8),
        }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Transcode Pexels raw clips to 1280p / ~25 fps / 30 s H.264")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--extra-dir", type=Path, action="append", default=None,
                        help="also pick up <id>.mp4 here (default: vest_eval). Raw files win on id clashes.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--jobs", type=int, default=0, help="worker processes (default: min(4, cpu/2))")
    parser.add_argument("--limit", type=int, default=0, help="only the first N ids, for a smoke test")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    extras = args.extra_dir if args.extra_dir is not None else [DEFAULT_EXTRA]
    jobs_map = discover(args.raw_dir, extras)
    ids = sorted(jobs_map)
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        print(f"no raw clips under {args.raw_dir} or {extras}")
        return 0
    jobs = [(pexels_id, str(jobs_map[pexels_id]), str(args.out_dir / f"pexels_{pexels_id}.mp4")) for pexels_id in ids]
    workers = args.jobs or max(1, min(4, (os.cpu_count() or 4) // 2))
    workers = max(1, min(workers, len(jobs)))
    print(f"prep {len(jobs)} clip(s) -> {args.out_dir}  workers={workers}", flush=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    ok = skipped = errors = 0
    # spawn so a worker does not inherit a half-open decoder from the parent
    import multiprocessing as mp
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for fut in as_completed(futures):
            result = fut.result()
            status = result["status"]
            if status == "ok":
                ok += 1
                print(
                    f"ok  pexels/{result['id']}  {result.get('width')}x{result.get('height')} "
                    f"{result.get('fps')}fps frames={result.get('frames')} rot={result.get('rotation')}",
                    flush=True,
                )
            elif status == "skipped":
                skipped += 1
                print(f"skip pexels/{result['id']}", flush=True)
            else:
                errors += 1
                print(f"ERROR pexels/{result['id']}: {result.get('error')}", flush=True)
                with args.errors.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"done ok={ok} skipped={skipped} errors={errors}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
