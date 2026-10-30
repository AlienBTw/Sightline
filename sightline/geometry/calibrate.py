"""Camera floor-plan calibration: 4-point homography + known-length scale check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sightline.config import ROOT
from sightline.errors import SightlineError
from sightline.geometry.homography import (
    compute_homography,
    metres_per_pixel_from_known_length,
    plan_distance,
)


SITE_CAMERAS = ROOT / "config" / "sites" / "demo_workshop" / "cameras.json"


def validate_camera_calibration(cam: dict[str, Any]) -> dict[str, Any]:
    """Validate and return calibration summary including metres scale factor."""
    required = [
        "camera_id",
        "pixel_points",
        "plan_points",
        "known_length_m",
        "known_length_plan_p1",
        "known_length_plan_p2",
    ]
    missing = [k for k in required if k not in cam]
    if missing:
        raise SightlineError(
            error_code="calibration_incomplete",
            component="geometry.calibrate",
            message=f"Camera missing fields: {missing}",
            hint="Set four pixel↔plan points and a known real-world length",
        )
    H = compute_homography(cam["pixel_points"], cam["plan_points"])
    scale = metres_per_pixel_from_known_length(
        cam["known_length_plan_p1"],
        cam["known_length_plan_p2"],
        float(cam["known_length_m"]),
    )
    measured = plan_distance(
        tuple(cam["known_length_plan_p1"]),  # type: ignore[arg-type]
        tuple(cam["known_length_plan_p2"]),  # type: ignore[arg-type]
    )
    return {
        "camera_id": cam["camera_id"],
        "homography_ok": H is not None,
        "known_length_m": float(cam["known_length_m"]),
        "plan_segment_m": measured,
        "scale_factor": scale,
        "scale_ok": abs(scale - 1.0) <= 0.25,
        "hint": (
            None
            if abs(scale - 1.0) <= 0.25
            else "Plan segment length differs from known_length_m by >25% — check units"
        ),
    }


def load_cameras_file(path: Path | None = None) -> dict[str, Any]:
    path = path or SITE_CAMERAS
    if not path.exists():
        raise SightlineError(
            error_code="cameras_file_missing",
            component="geometry.calibrate",
            message=f"Missing {path}",
            hint="Create config/sites/demo_workshop/cameras.json",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def save_camera_calibration(camera_id: str, update: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    path = path or SITE_CAMERAS
    data = load_cameras_file(path)
    cams = data.get("cameras") or []
    found = False
    for i, cam in enumerate(cams):
        if cam.get("camera_id") == camera_id:
            cams[i] = {**cam, **update, "camera_id": camera_id}
            summary = validate_camera_calibration(cams[i])
            found = True
            break
    if not found:
        raise SightlineError(
            error_code="camera_unknown",
            component="geometry.calibrate",
            message=f"No camera {camera_id}",
            hint="Add the camera entry first",
        )
    data["cameras"] = cams
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return summary
