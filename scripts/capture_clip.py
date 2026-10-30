"""Capture a short MP4 from webcam or RTSP into data/clips/ (gitignored)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from sightline.errors import SightlineError


def main() -> int:
    p = argparse.ArgumentParser(description="Capture demo clip for Sightline")
    p.add_argument("--out", required=True, help="Output MP4 path under data/clips/")
    p.add_argument("--source", default="0", help="Webcam index or RTSP/file URL")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--fps", type=float, default=15.0)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src: int | str
    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        err = SightlineError(
            error_code="capture_open_failed",
            component="scripts.capture_clip",
            message=f"Cannot open source {args.source!r}",
            hint="Check webcam index / RTSP URL / permissions",
        )
        print(err.to_dict(), file=sys.stderr)
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, args.fps, (w, h))
    if not writer.isOpened():
        print(
            SightlineError(
                error_code="capture_writer_failed",
                component="scripts.capture_clip",
                message=f"Cannot open writer for {out}",
                hint="Check path and codecs",
            ).to_dict(),
            file=sys.stderr,
        )
        return 1

    t_end = time.time() + args.seconds
    n = 0
    try:
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok:
                print(
                    SightlineError(
                        error_code="capture_frame_drop",
                        component="scripts.capture_clip",
                        message="Frame read failed — source disconnected",
                        hint="Reconnect camera or shorten --seconds",
                    ).to_dict(),
                    file=sys.stderr,
                )
                return 1
            writer.write(frame)
            n += 1
    finally:
        cap.release()
        writer.release()

    print(f"OK: wrote {n} frames → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
