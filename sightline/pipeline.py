"""End-to-end frame analysis: detect → project → rules → optional VLM → HistGB → hazards."""

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sightline.config import ROOT, Settings, get_settings, load_yaml
from sightline.decision.histgb import HistGBDecision
from sightline.errors import SightlineError
from sightline.geometry import compute_homography, pixel_to_plan
from sightline.hazards.state_machine import transition
from sightline.hazards.store import Store
from sightline.hazards.tracker import SpatialTracker
from sightline.rules.engine import RulesEngine
from sightline.vision.backends import build_detector
from sightline.vision.vlm_gate import VLMGate


class Pipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = Store(self.settings.sightline_db)
        self.detector = build_detector(self.settings)
        self.rules = RulesEngine()
        self.tracker = SpatialTracker()
        self.decide = HistGBDecision()
        self.vlm = VLMGate(self.settings)
        self.cameras = self._load_cameras()
        self.zones = self._load_zones()
        self._H: dict[str, np.ndarray] = {}
        self.latencies: dict[str, list[float]] = {
            "detect_ms": [],
            "vlm_ms": [],
            "decide_ms": [],
        }
        for cam in self.cameras:
            cid = cam["camera_id"]
            self._H[cid] = compute_homography(cam["pixel_points"], cam["plan_points"])

    def _load_cameras(self) -> list[dict[str, Any]]:
        path = ROOT / "config" / "sites" / "demo_workshop" / "cameras.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("cameras") or [])

    def _load_zones(self) -> dict[str, Any]:
        path = ROOT / "config" / "sites" / "demo_workshop" / "zones.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def analyze_frame(
        self,
        frame: np.ndarray,
        *,
        camera_id: str,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        if self.settings.sightline_mode == "live" and self.settings.detector_backend == "replay":
            raise SightlineError(
                error_code="mode_backend_conflict",
                component="pipeline",
                message="SIGHTLINE_MODE=live cannot use DETECTOR_BACKEND=replay",
                hint="Set DETECTOR_BACKEND to llmdet (or another live backend), or set MODE=replay",
            )
        ts = timestamp if timestamp is not None else time.time()
        frame_id = f"frm_{uuid.uuid4().hex[:12]}"
        if camera_id not in self._H:
            raise SightlineError(
                error_code="camera_unknown",
                component="pipeline",
                message=f"Unknown camera_id={camera_id}",
                hint="Add the camera to config/sites/demo_workshop/cameras.json",
            )

        t0 = time.perf_counter()
        detections = self.detector.detect(
            frame, camera_id=camera_id, frame_id=frame_id, timestamp=ts
        )
        self.latencies["detect_ms"].append((time.perf_counter() - t0) * 1000)

        floor_xy: dict[str, tuple[float, float]] = {}
        H = self._H[camera_id]
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            u = (x1 + x2) / 2.0
            v = float(y2)  # footpoint
            xy = pixel_to_plan(H, u, v)
            key = f"{d.frame_id}:{d.label}:{d.bbox_xyxy}"
            floor_xy[key] = xy
            self.store.execute(
                """
                INSERT INTO detections (
                  detection_id, frame_id, camera_id, timestamp, label, score,
                  bbox_x1, bbox_y1, bbox_x2, bbox_y2, floor_x, floor_y, metadata_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    f"det_{uuid.uuid4().hex[:12]}",
                    frame_id,
                    camera_id,
                    ts,
                    d.label,
                    d.score,
                    x1,
                    y1,
                    x2,
                    y2,
                    xy[0],
                    xy[1],
                    json.dumps(d.metadata),
                ),
            )

        candidates = self.rules.propose(
            detections, floor_xy=floor_xy, zones=self.zones, timestamp=ts
        )
        self.tracker.update(candidates, ts)

        # Optional VLM + decide
        decisions = []
        hazards = []
        th = load_yaml("thresholds.yaml")
        min_present = int((th.get("candidate") or {}).get("min_detections_to_confirm", 3))

        for c in candidates:
            self.store.execute(
                """
                INSERT OR REPLACE INTO candidates (
                  candidate_id, camera_id, hazard_family, rule_name, first_seen, last_seen,
                  duration_seconds, detector_label, detector_score, zone_id, floor_x, floor_y,
                  features_json, status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    c["candidate_id"],
                    c.get("camera_id"),
                    c.get("hazard_family"),
                    c.get("rule_name"),
                    ts,
                    ts,
                    float((c.get("features") or {}).get("duration_seconds") or 0),
                    c.get("detector_label"),
                    c.get("detector_score"),
                    c.get("zone_id"),
                    c.get("floor_x"),
                    c.get("floor_y"),
                    json.dumps(c.get("features") or {}),
                    "candidate",
                ),
            )
            # Attach track persistence
            for t in self.tracker.tracks:
                if (
                    t.hazard_family == c["hazard_family"]
                    and plan_close(t.floor_x, t.floor_y, c["floor_x"], c["floor_y"])
                ):
                    c["features"]["duration_seconds"] = t.last_seen - t.first_seen
                    c["features"]["frame_count"] = t.consecutive_present
                    break

            # Pre-score for gating using rule_score as prior P proxy before model
            prior_p = float((c.get("features") or {}).get("rule_score", 0.5))
            if self.vlm.should_gate(c["hazard_family"], prior_p):
                crop = _crop_b64(frame, c["features"]["bbox_xyxy"])
                t1 = time.perf_counter()
                vlm_out = self.vlm.analyze_crop(crop, family=c["hazard_family"])
                self.latencies["vlm_ms"].append((time.perf_counter() - t1) * 1000)
                c["features"]["vlm_called"] = True
                c["features"]["vlm_source_active"] = vlm_out.get("source_active")
                c["features"]["vlm_material_combustible"] = vlm_out.get("material_combustible")
                c["features"]["vlm_liquid_present"] = vlm_out.get("liquid_present")
                c["features"]["vlm_uncertainty"] = vlm_out.get("uncertainty")
                c["features"]["vlm_evidence"] = vlm_out.get("evidence")
            else:
                c["features"]["vlm_called"] = False

            t2 = time.perf_counter()
            try:
                decision = self.decide.predict(c)
            except SightlineError as e:
                if e.error_code == "decide_model_missing":
                    # Fail loud — do not invent probabilities
                    raise
                raise
            self.latencies["decide_ms"].append((time.perf_counter() - t2) * 1000)
            decisions.append(decision)

            status = "candidate"
            if (c["features"].get("frame_count") or 1) >= min_present and not decision["abstain"]:
                status = transition("candidate", "confirmed")
                status = transition(status, "open")
            elif decision["needs_human"]:
                status = transition("candidate", "needs_human")

            hid = f"hz_{uuid.uuid4().hex[:12]}"
            wall = time.time()
            hazard = {
                "hazard_id": hid,
                "candidate_id": c["candidate_id"],
                "hazard_family": c["hazard_family"],
                "status": status,
                "severity": decision["severity"],
                "risk_score": decision["risk_score"],
                "first_seen": wall,
                "confirmed_at": wall if status in {"confirmed", "open"} else None,
                "fixed_at": None,
                "verified_at": None,
                "time_to_fix_seconds": None,
                "location": {"floor_x": c["floor_x"], "floor_y": c["floor_y"], "zone_id": c.get("zone_id")},
                "evidence": {
                    "detector_label": c["detector_label"],
                    "detector_score": c["detector_score"],
                    "features": c["features"],
                    "decision": decision,
                    "frame_id": frame_id,
                },
            }
            self.store.upsert_hazard(hazard)
            hazards.append(hazard)

        return {
            "frame_id": frame_id,
            "camera_id": camera_id,
            "timestamp": ts,
            "detections": [d.__dict__ for d in detections],
            "candidates": candidates,
            "decisions": decisions,
            "hazards": hazards,
        }

    def analyze_video(self, path: str | Path, *, camera_id: str) -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise SightlineError(
                error_code="video_missing",
                component="pipeline",
                message=f"Video not found: {path}",
                hint="Pass an existing MP4 path",
            )
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SightlineError(
                error_code="video_open_failed",
                component="pipeline",
                message=f"Could not open {path}",
                hint="Check codec/path",
            )
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / max(self.settings.sample_fps, 0.1))))
        idx = 0
        results = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                ts = idx / fps
                results.append(self.analyze_frame(frame, camera_id=camera_id, timestamp=ts))
            idx += 1
        cap.release()
        if not results:
            raise SightlineError(
                error_code="video_empty",
                component="pipeline",
                message="No frames sampled from video",
                hint="Check SAMPLE_FPS and video contents",
            )
        return {"frames_analyzed": len(results), "results": results}

    def mark_fixed(self, hazard_id: str) -> dict[str, Any]:
        row = self.store.fetchone("SELECT * FROM hazards WHERE hazard_id=?", (hazard_id,))
        if not row:
            raise SightlineError(
                error_code="hazard_not_found",
                component="pipeline",
                message=f"No hazard {hazard_id}",
                hint="Refresh the fix list",
            )
        status = transition(row["status"], "fix_pending")
        now = time.time()
        self.store.execute(
            "UPDATE hazards SET status=?, fixed_at=? WHERE hazard_id=?",
            (status, now, hazard_id),
        )
        # Auto-verify after configured window is left to a follow-up poll; for MVP verify immediately on second call
        return self.store.fetchone("SELECT * FROM hazards WHERE hazard_id=?", (hazard_id,)) or {}

    def verify_fixed(self, hazard_id: str) -> dict[str, Any]:
        row = self.store.fetchone("SELECT * FROM hazards WHERE hazard_id=?", (hazard_id,))
        if not row:
            raise SightlineError(
                error_code="hazard_not_found",
                component="pipeline",
                message=f"No hazard {hazard_id}",
                hint="Refresh the fix list",
            )
        status = transition(row["status"], "verified")
        now = time.time()
        confirmed = row.get("confirmed_at") or row.get("first_seen") or now
        ttf = float(now - confirmed)
        self.store.execute(
            "UPDATE hazards SET status=?, verified_at=?, time_to_fix_seconds=? WHERE hazard_id=?",
            (status, now, ttf, hazard_id),
        )
        return self.store.fetchone("SELECT * FROM hazards WHERE hazard_id=?", (hazard_id,)) or {}


def plan_close(x1: float, y1: float, x2: float, y2: float, r: float = 0.5) -> bool:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 <= r


def _crop_b64(frame: np.ndarray, bbox: list[int]) -> str:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise SightlineError(
            error_code="vlm_crop_empty",
            component="pipeline",
            message="Bounding box crop is empty",
            hint="Check detector boxes",
        )
    ok, buf = cv2.imencode(".jpg", crop)
    if not ok:
        raise SightlineError(
            error_code="vlm_crop_encode_failed",
            component="pipeline",
            message="JPEG encode failed",
            hint="Check OpenCV build",
        )
    return base64.b64encode(buf.tobytes()).decode("ascii")
