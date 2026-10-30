"""Validate demo_workshop camera calibration (metres scale)."""

from __future__ import annotations

import sys

from sightline.errors import SightlineError
from sightline.geometry.calibrate import load_cameras_file, validate_camera_calibration


def main() -> int:
    try:
        data = load_cameras_file()
        cams = data.get("cameras") or []
        if not cams:
            raise SightlineError(
                error_code="calibration_no_cameras",
                component="scripts.validate_calibration",
                message="cameras.json has no cameras",
                hint="Add at least cam_01",
            )
        for cam in cams:
            summary = validate_camera_calibration(cam)
            print(summary)
            if not summary["scale_ok"]:
                print("WARN:", summary["hint"], file=sys.stderr)
        print("OK: calibration validated")
        return 0
    except SightlineError as e:
        print(f"FAIL: {e.to_dict()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
