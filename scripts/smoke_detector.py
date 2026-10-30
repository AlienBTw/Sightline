"""Run only the configured DETECTOR_BACKEND on a synthetic frame. Fail loud."""

from __future__ import annotations

import sys

import numpy as np

from sightline.config import get_settings
from sightline.errors import SightlineError
from sightline.vision.backends import build_detector


def main() -> int:
    try:
        settings = get_settings()
        det = build_detector(settings)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        fid = "frm_000" if settings.detector_backend == "replay" else "smoke_0"
        dets = det.detect(frame, camera_id="cam_01", frame_id=fid, timestamp=0.0)
        print(f"OK: backend={settings.detector_backend} detections={len(dets)}")
        for d in dets[:5]:
            print(f"  {d.label} score={d.score:.3f} bbox={d.bbox_xyxy}")
        return 0
    except SightlineError as e:
        print(f"FAIL: {e.to_dict()}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
