"""Run the configured detector on an image or video; write JSON results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from sightline.config import get_settings
from sightline.errors import SightlineError
from sightline.vision.backends import build_detector


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", help="Path to image")
    p.add_argument("--video", help="Path to video")
    p.add_argument("--out", default="detections_out.json")
    p.add_argument("--camera-id", default="cam_01")
    args = p.parse_args()
    if not args.image and not args.video:
        print("Provide --image or --video", file=sys.stderr)
        return 1
    try:
        settings = get_settings()
        det = build_detector(settings)
        results = []
        if args.image:
            frame = cv2.imread(args.image)
            if frame is None:
                raise SightlineError(
                    error_code="image_open_failed",
                    component="scripts.run_detector",
                    message=f"Cannot read {args.image}",
                    hint="Check path",
                )
            dets = det.detect(frame, camera_id=args.camera_id, frame_id="img_0", timestamp=0.0)
            results.append({"frame_id": "img_0", "detections": [d.__dict__ for d in dets]})
        else:
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                raise SightlineError(
                    error_code="video_open_failed",
                    component="scripts.run_detector",
                    message=f"Cannot open {args.video}",
                    hint="Check path/codec",
                )
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(round(fps / max(settings.sample_fps, 0.1))))
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    fid = f"frm_{idx:05d}"
                    dets = det.detect(
                        frame, camera_id=args.camera_id, frame_id=fid, timestamp=idx / fps
                    )
                    results.append({"frame_id": fid, "detections": [d.__dict__ for d in dets]})
                idx += 1
            cap.release()
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"OK: backend={settings.detector_backend} frames={len(results)} → {args.out}")
        return 0
    except SightlineError as e:
        print(f"FAIL: {e.to_dict()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
