"""Config-driven hazard candidate rules."""

from __future__ import annotations

import uuid
from typing import Any

from sightline.config import load_yaml
from sightline.geometry import (
    estimate_clear_width_m,
    find_exit_clearance,
    find_walkway,
    plan_distance,
)
from sightline.vision.backends.base import Detection


def _norm(label: str) -> str:
    return label.strip().lower().replace("_", " ").replace("-", " ")


class RulesEngine:
    def __init__(self) -> None:
        self.hazards = load_yaml("hazards.yaml")
        self.thresholds = load_yaml("thresholds.yaml")

    def propose(
        self,
        detections: list[Detection],
        *,
        floor_xy: dict[str, tuple[float, float]],
        zones: dict[str, Any],
        timestamp: float,
    ) -> list[dict[str, Any]]:
        """Return rule candidates with evidence features (not final decisions)."""
        candidates: list[dict[str, Any]] = []
        by_label: dict[str, list[tuple[Detection, tuple[float, float]]]] = {}
        for d in detections:
            key = f"{d.frame_id}:{d.label}:{d.bbox_xyxy}"
            xy = floor_xy.get(key)
            if xy is None:
                continue
            by_label.setdefault(_norm(d.label), []).append((d, xy))

        # slip_trip
        cfg = self.hazards.get("slip_trip") or {}
        if cfg.get("enabled"):
            for cls in cfg.get("classes") or []:
                for d, xy in by_label.get(_norm(cls), []):
                    zone = find_walkway(xy[0], xy[1], zones)
                    if zone:
                        candidates.append(
                            self._cand(
                                "slip_trip",
                                "object_in_walkway",
                                d,
                                xy,
                                zone,
                                timestamp,
                                {
                                    "distance_to_heat_m": None,
                                    "distance_to_power_m": None,
                                    "clear_width_m": None,
                                    "in_walkway": True,
                                },
                                severity_base=float((cfg.get("severity") or {}).get("base", 0.55)),
                            )
                        )

        # fire_heat
        cfg = self.hazards.get("fire_heat") or {}
        if cfg.get("enabled"):
            radius = float(cfg.get("fuel_near_heat_radius_m", 1.0))
            fuels = [_norm(c) for c in (cfg.get("fuel_classes") or [])]
            heats = [_norm(c) for c in (cfg.get("heat_classes") or [])]
            fuel_dets = [(d, xy) for c in fuels for d, xy in by_label.get(c, [])]
            heat_dets = [(d, xy) for c in heats for d, xy in by_label.get(c, [])]
            # Also use configured heat anchors
            for fuel_d, fxy in fuel_dets:
                for heat_d, hxy in heat_dets:
                    dist = plan_distance(fxy, hxy)
                    if dist <= radius:
                        candidates.append(
                            self._cand(
                                "fire_heat",
                                "fuel_near_heat",
                                fuel_d,
                                fxy,
                                None,
                                timestamp,
                                {
                                    "distance_to_heat_m": dist,
                                    "paired_label": heat_d.label,
                                    "paired_score": heat_d.score,
                                    "in_walkway": find_walkway(fxy[0], fxy[1], zones) is not None,
                                },
                                severity_base=float((cfg.get("severity") or {}).get("base", 0.65)),
                            )
                        )
                for hs in zones.get("heat_sources") or []:
                    hxy = (float(hs["point"][0]), float(hs["point"][1]))
                    dist = plan_distance(fxy, hxy)
                    if dist <= float(hs.get("radius_m", radius)):
                        candidates.append(
                            self._cand(
                                "fire_heat",
                                "fuel_near_heat_anchor",
                                fuel_d,
                                fxy,
                                None,
                                timestamp,
                                {
                                    "distance_to_heat_m": dist,
                                    "heat_source_id": hs["source_id"],
                                    "in_walkway": False,
                                },
                                severity_base=float((cfg.get("severity") or {}).get("base", 0.65)),
                            )
                        )

        # egress
        cfg = self.hazards.get("egress") or {}
        if cfg.get("enabled"):
            blockers = [_norm(c) for c in (cfg.get("blocking_classes") or [])]
            obs = []
            for c in blockers:
                for d, xy in by_label.get(c, []):
                    ex = find_exit_clearance(xy[0], xy[1], zones)
                    if ex:
                        obs.append((d, xy, ex))
            by_exit: dict[str, list[tuple[Detection, tuple[float, float], dict]]] = {}
            for d, xy, ex in obs:
                by_exit.setdefault(ex["exit_id"], []).append((d, xy, ex))
            for exit_id, items in by_exit.items():
                ex = items[0][2]
                width = estimate_clear_width_m([xy for _, xy, _ in items], ex)
                min_w = float(ex.get("min_clear_width_m", cfg.get("min_clear_width_m", 0.9)))
                if width < min_w:
                    d, xy, _ = items[0]
                    candidates.append(
                        self._cand(
                            "egress",
                            "exit_clear_width",
                            d,
                            xy,
                            exit_id,
                            timestamp,
                            {
                                "clear_width_m": width,
                                "min_clear_width_m": min_w,
                                "in_exit_clearance": True,
                            },
                            severity_base=float((cfg.get("severity") or {}).get("base", 0.7)),
                        )
                    )

        # electrical
        cfg = self.hazards.get("electrical") or {}
        if cfg.get("enabled"):
            radius = float(cfg.get("radius_m", 0.5))
            risk = [_norm(c) for c in (cfg.get("risk_classes") or [])]
            elec = [_norm(c) for c in (cfg.get("electrical_classes") or [])]
            risk_dets = [(d, xy) for c in risk for d, xy in by_label.get(c, [])]
            elec_dets = [(d, xy) for c in elec for d, xy in by_label.get(c, [])]
            for rd, rxy in risk_dets:
                for ed, exy in elec_dets:
                    dist = plan_distance(rxy, exy)
                    if dist <= radius:
                        candidates.append(
                            self._cand(
                                "electrical",
                                "risk_near_electrical",
                                rd,
                                rxy,
                                None,
                                timestamp,
                                {
                                    "distance_to_power_m": dist,
                                    "paired_label": ed.label,
                                },
                                severity_base=float((cfg.get("severity") or {}).get("base", 0.6)),
                            )
                        )
                for es in zones.get("electrical_sources") or []:
                    exy = (float(es["point"][0]), float(es["point"][1]))
                    dist = plan_distance(rxy, exy)
                    if dist <= float(es.get("radius_m", radius)):
                        candidates.append(
                            self._cand(
                                "electrical",
                                "risk_near_electrical_anchor",
                                rd,
                                rxy,
                                None,
                                timestamp,
                                {
                                    "distance_to_power_m": dist,
                                    "electrical_source_id": es["source_id"],
                                },
                                severity_base=float((cfg.get("severity") or {}).get("base", 0.6)),
                            )
                        )

        return candidates

    def _cand(
        self,
        family: str,
        rule_name: str,
        d: Detection,
        xy: tuple[float, float],
        zone_id: str | None,
        timestamp: float,
        features: dict[str, Any],
        *,
        severity_base: float,
    ) -> dict[str, Any]:
        feats = {
            "detector_label": d.label,
            "detector_score": d.score,
            "bbox_xyxy": list(d.bbox_xyxy),
            "floor_x": xy[0],
            "floor_y": xy[1],
            "severity_prior": severity_base,
            "rule_score": float(d.score) * severity_base,
            **features,
        }
        return {
            "candidate_id": f"cand_{uuid.uuid4().hex[:12]}",
            "camera_id": d.camera_id,
            "frame_id": d.frame_id,
            "hazard_family": family,
            "rule_name": rule_name,
            "timestamp": timestamp,
            "detector_label": d.label,
            "detector_score": d.score,
            "zone_id": zone_id,
            "floor_x": xy[0],
            "floor_y": xy[1],
            "features": feats,
            "status": "candidate",
        }
