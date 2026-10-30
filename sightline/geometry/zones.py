"""Zone containment and exit clear-width helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from sightline.errors import SightlineError


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    if len(polygon) < 3:
        raise SightlineError(
            error_code="zone_invalid_polygon",
            component="geometry.zones",
            message="Polygon needs at least 3 points",
            hint="Fix zones.json",
        )
    pts = np.array(polygon, dtype=np.float64)
    # Ray casting
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def find_walkway(x: float, y: float, zones: dict[str, Any]) -> str | None:
    for z in zones.get("walkways") or []:
        if point_in_polygon(x, y, z["polygon_plan"]):
            return z["zone_id"]
    return None


def find_exit_clearance(x: float, y: float, zones: dict[str, Any]) -> dict[str, Any] | None:
    for ex in zones.get("exits") or []:
        if point_in_polygon(x, y, ex["clearance_polygon"]):
            return ex
    return None


def estimate_clear_width_m(
    obstacles: list[tuple[float, float]],
    exit_zone: dict[str, Any],
) -> float:
    """Rough clear width: min distance of obstacle to door along clearance polygon bbox width."""
    poly = np.array(exit_zone["clearance_polygon"], dtype=np.float64)
    width = float(poly[:, 0].max() - poly[:, 0].min())
    door = exit_zone["door_point"]
    if not obstacles:
        return width
    # Reduce width by nearest obstacle distance to door projected on x
    nearest = min(abs(o[0] - door[0]) for o in obstacles)
    return max(0.0, min(width, nearest))
