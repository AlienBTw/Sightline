"""Simple spatial persistence for candidates/hazards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sightline.config import load_yaml
from sightline.geometry.homography import plan_distance


@dataclass
class Track:
    track_id: str
    hazard_family: str
    label: str
    floor_x: float
    floor_y: float
    first_seen: float
    last_seen: float
    consecutive_present: int = 1
    consecutive_absent: int = 0
    features: dict[str, Any] = field(default_factory=dict)
    status: str = "candidate"


class SpatialTracker:
    def __init__(self) -> None:
        th = load_yaml("thresholds.yaml")
        self.match_radius_m = float((th.get("candidate") or {}).get("match_radius_m", 0.5))
        self.max_gap_s = float((th.get("candidate") or {}).get("max_gap_seconds", 5))
        self.tracks: list[Track] = []

    def update(self, candidates: list[dict[str, Any]], timestamp: float) -> list[Track]:
        matched: set[int] = set()
        for c in candidates:
            best_i = None
            best_d = self.match_radius_m
            for i, t in enumerate(self.tracks):
                if i in matched:
                    continue
                if t.hazard_family != c["hazard_family"]:
                    continue
                if timestamp - t.last_seen > self.max_gap_s:
                    continue
                d = plan_distance((t.floor_x, t.floor_y), (c["floor_x"], c["floor_y"]))
                if d <= best_d:
                    best_d = d
                    best_i = i
            if best_i is None:
                self.tracks.append(
                    Track(
                        track_id=c["candidate_id"],
                        hazard_family=c["hazard_family"],
                        label=c["detector_label"],
                        floor_x=c["floor_x"],
                        floor_y=c["floor_y"],
                        first_seen=timestamp,
                        last_seen=timestamp,
                        features=c.get("features") or {},
                    )
                )
            else:
                matched.add(best_i)
                t = self.tracks[best_i]
                t.last_seen = timestamp
                t.consecutive_present += 1
                t.consecutive_absent = 0
                t.floor_x = c["floor_x"]
                t.floor_y = c["floor_y"]
                t.features = c.get("features") or t.features

        for i, t in enumerate(self.tracks):
            if i not in matched and timestamp - t.last_seen > 0:
                if timestamp - t.last_seen >= 1.0 / 2.0:
                    t.consecutive_absent += 1
                    t.consecutive_present = 0
        return list(self.tracks)
