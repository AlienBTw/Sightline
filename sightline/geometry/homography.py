"""Four-point homography and metre-scale projection."""

from __future__ import annotations

import numpy as np

from sightline.errors import SightlineError


def compute_homography(
    pixel_points: list[list[float]],
    plan_points: list[list[float]],
) -> np.ndarray:
    if len(pixel_points) != 4 or len(plan_points) != 4:
        raise SightlineError(
            error_code="homography_bad_points",
            component="geometry.homography",
            message="Homography requires exactly 4 pixel points and 4 plan points",
            hint="Calibrate the camera with four matched corners",
        )
    src = np.array(pixel_points, dtype=np.float64)
    dst = np.array(plan_points, dtype=np.float64)
    try:
        import cv2

        H, _ = cv2.findHomography(src, dst, method=0)
    except Exception as e:
        raise SightlineError(
            error_code="homography_failed",
            component="geometry.homography",
            message=str(e),
            hint="Install opencv-python-headless and check point coordinates",
        ) from e
    if H is None:
        raise SightlineError(
            error_code="homography_failed",
            component="geometry.homography",
            message="cv2.findHomography returned None",
            hint="Points may be collinear or degenerate",
        )
    return H


def pixel_to_plan(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    pt = np.array([u, v, 1.0], dtype=np.float64)
    out = H @ pt
    if abs(out[2]) < 1e-9:
        raise SightlineError(
            error_code="homography_project_failed",
            component="geometry.homography",
            message="Homogeneous coordinate w≈0",
            hint="Point may be at infinity relative to the plane",
        )
    return float(out[0] / out[2]), float(out[1] / out[2])


def plan_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def metres_per_pixel_from_known_length(
    plan_p1: list[float],
    plan_p2: list[float],
    known_length_m: float,
) -> float:
    """For axis-aligned plan coords already in metres, returns 1.0; validates length."""
    d = plan_distance((plan_p1[0], plan_p1[1]), (plan_p2[0], plan_p2[1]))
    if d <= 0:
        raise SightlineError(
            error_code="calibration_bad_length",
            component="geometry.homography",
            message="Known-length plan segment has zero length",
            hint="Set known_length_plan_p1/p2 to distinct plan points",
        )
    # plan_points are stored in metres already; scale factor for validation
    scale = known_length_m / d
    if abs(scale - 1.0) > 0.25:
        # Soft warning via details — still return corrected scale for callers that need it
        pass
    return float(scale)
