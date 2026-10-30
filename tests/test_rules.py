import json
from pathlib import Path

from sightline.config import ROOT
from sightline.geometry import compute_homography, pixel_to_plan
from sightline.hazards.state_machine import transition
from sightline.hazards.tracker import SpatialTracker
from sightline.rules.engine import RulesEngine
from sightline.vision.backends.base import Detection


def _floor_xy_for(dets: list[Detection]) -> dict[str, tuple[float, float]]:
    cams = json.loads((ROOT / "config/sites/demo_workshop/cameras.json").read_text())
    cam = cams["cameras"][0]
    H = compute_homography(cam["pixel_points"], cam["plan_points"])
    out = {}
    for d in dets:
        x1, y1, x2, y2 = d.bbox_xyxy
        u = (x1 + x2) / 2.0
        v = float(y2)
        out[f"{d.frame_id}:{d.label}:{d.bbox_xyxy}"] = pixel_to_plan(H, u, v)
    return out


def test_rules_produce_metre_candidates():
    zones = json.loads((ROOT / "config/sites/demo_workshop/zones.json").read_text())
    engine = RulesEngine()
    # Object in walkway (plan ~3,2.5 maps near walkway)
    # Use pixel that projects into walkway via demo homography roughly
    dets = [
        Detection(
            camera_id="cam_01",
            frame_id="t0",
            timestamp=0.0,
            label="object_on_floor",
            score=0.9,
            bbox_xyxy=(400, 380, 520, 480),
        ),
        Detection(
            camera_id="cam_01",
            frame_id="t0",
            timestamp=0.0,
            label="heater",
            score=0.85,
            bbox_xyxy=(560, 250, 680, 400),
        ),
        Detection(
            camera_id="cam_01",
            frame_id="t0",
            timestamp=0.0,
            label="cardboard_box",
            score=0.86,
            bbox_xyxy=(570, 260, 700, 420),
        ),
        Detection(
            camera_id="cam_01",
            frame_id="t0",
            timestamp=0.0,
            label="pallet",
            score=0.9,
            bbox_xyxy=(820, 300, 980, 500),
        ),
        Detection(
            camera_id="cam_01",
            frame_id="t0",
            timestamp=0.0,
            label="power_strip",
            score=0.87,
            bbox_xyxy=(300, 420, 380, 480),
        ),
        Detection(
            camera_id="cam_01",
            frame_id="t0",
            timestamp=0.0,
            label="extension_cord",
            score=0.8,
            bbox_xyxy=(320, 400, 450, 470),
        ),
    ]
    floor = _floor_xy_for(dets)
    cands = engine.propose(dets, floor_xy=floor, zones=zones, timestamp=0.0)
    families = {c["hazard_family"] for c in cands}
    assert cands, "expected at least one candidate"
    # At least fire_heat or electrical or egress or slip should fire
    assert families & {"slip_trip", "fire_heat", "egress", "electrical"}
    for c in cands:
        feats = c["features"]
        # Metre fields present when applicable
        for key in ("distance_to_heat_m", "distance_to_power_m", "clear_width_m"):
            if feats.get(key) is not None:
                assert isinstance(feats[key], float)


def test_tracker_debounce_and_state_machine():
    tracker = SpatialTracker()
    c = {
        "candidate_id": "cand_a",
        "hazard_family": "slip_trip",
        "detector_label": "box",
        "floor_x": 3.0,
        "floor_y": 2.5,
        "features": {},
    }
    tracker.update([c], 0.0)
    tracker.update([{**c, "candidate_id": "cand_b"}], 0.5)
    tracker.update([{**c, "candidate_id": "cand_c"}], 1.0)
    assert tracker.tracks[0].consecutive_present >= 3
    s = transition("candidate", "confirmed")
    s = transition(s, "open")
    assert s == "open"
