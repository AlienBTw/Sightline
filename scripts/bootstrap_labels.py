"""Generate honest bootstrap training rows for HistGB (synthetic features, not site footage).

Writes CSV used by train_histgb.py --bootstrap. Report n on the calibration page.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from sightline.decision.features import feature_names

FAMILIES = ["slip_trip", "fire_heat", "egress", "electrical", "other"]


def synth_row(rng: random.Random) -> dict[str, float | int]:
    fam = rng.choice(FAMILIES)
    # Higher detector/rule score + duration → more likely real hazard
    detector_score = rng.uniform(0.2, 0.98)
    duration = rng.uniform(0, 30)
    rule_score = detector_score * rng.uniform(0.4, 0.9)
    severity = rng.uniform(0.3, 0.9)
    in_walkway = 1.0 if fam == "slip_trip" and rng.random() > 0.3 else float(rng.random() > 0.7)
    clear_w = rng.uniform(0.3, 1.5) if fam == "egress" else -1.0
    dist_heat = rng.uniform(0.1, 3.0) if fam == "fire_heat" else -1.0
    dist_power = rng.uniform(0.05, 2.0) if fam == "electrical" else -1.0
    vlm_active = 1.0 if fam == "fire_heat" and rng.random() > 0.5 else 0.0
    vlm_liquid = 1.0 if fam == "electrical" and rng.random() > 0.6 else 0.0

    # Soft label rule
    score = rule_score + 0.02 * duration + 0.15 * in_walkway
    if clear_w >= 0:
        score += max(0.0, 0.9 - clear_w)
    if dist_heat >= 0:
        score += max(0.0, 1.0 - dist_heat) + 0.2 * vlm_active
    if dist_power >= 0:
        score += max(0.0, 0.5 - dist_power) + 0.2 * vlm_liquid
    if fam == "other":
        score *= 0.5
    is_real = 1 if score + rng.uniform(-0.15, 0.15) > 0.55 else 0

    row: dict[str, float | int] = {
        "detector_score": detector_score,
        "duration_seconds": duration,
        "in_walkway": in_walkway,
        "in_exit_clearance": 1.0 if fam == "egress" else 0.0,
        "distance_to_heat_m": dist_heat,
        "distance_to_exit_m": rng.uniform(0.5, 8.0) if fam == "egress" else -1.0,
        "distance_to_power_m": dist_power,
        "clear_width_m": clear_w,
        "vlm_called": 1.0 if vlm_active or vlm_liquid else 0.0,
        "vlm_source_active": vlm_active,
        "vlm_material_combustible": vlm_active,
        "vlm_liquid_present": vlm_liquid,
        "rule_score": rule_score,
        "severity_prior": severity,
        "is_real_hazard": is_real,
    }
    for name in FAMILIES:
        row[f"family_{name}"] = 1.0 if fam == name else 0.0
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("data/models/bootstrap_labels.csv"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = feature_names() + ["is_real_hazard"]
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for _ in range(args.n):
            w.writerow(synth_row(rng))
    print(f"OK: wrote {args.n} bootstrap rows -> {args.out}")
    print("Note: synthetic features for MVP training — report n honestly on Calibration page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
