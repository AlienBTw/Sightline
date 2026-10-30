"""Shared feature row schema for HistGB (and future Laya)."""

from __future__ import annotations

from typing import Any


FEATURE_KEYS = [
    "detector_score",
    "duration_seconds",
    "in_walkway",
    "in_exit_clearance",
    "distance_to_heat_m",
    "distance_to_exit_m",
    "distance_to_power_m",
    "clear_width_m",
    "vlm_called",
    "vlm_source_active",
    "vlm_material_combustible",
    "vlm_liquid_present",
    "rule_score",
    "severity_prior",
]

FAMILY_ONEHOT = ["slip_trip", "fire_heat", "egress", "electrical", "other"]


def row_from_candidate(c: dict[str, Any]) -> dict[str, float]:
    f = c.get("features") or {}
    row: dict[str, float] = {}
    for k in FEATURE_KEYS:
        v = f.get(k, c.get(k))
        if v is None:
            row[k] = -1.0
        elif isinstance(v, bool):
            row[k] = 1.0 if v else 0.0
        else:
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = -1.0
    fam = c.get("hazard_family") or "other"
    for name in FAMILY_ONEHOT:
        row[f"family_{name}"] = 1.0 if fam == name else 0.0
    return row


def vectorize(rows: list[dict[str, float]]) -> list[list[float]]:
    keys = FEATURE_KEYS + [f"family_{n}" for n in FAMILY_ONEHOT]
    return [[r.get(k, -1.0) for k in keys] for r in rows]


def feature_names() -> list[str]:
    return FEATURE_KEYS + [f"family_{n}" for n in FAMILY_ONEHOT]
