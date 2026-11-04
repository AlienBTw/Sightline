import json

import numpy as np

from sightline_pipeline.escalation import (EscalationPolicy, MuseVerifier, SamRefiner,
                                           parse_sam_boxes, parse_verdict, pick_box, refine_circle)
from sightline_pipeline.localize import Circle
from sightline_pipeline.storage import ClipWriter, EventStore, RingBuffer, resize_w


def test_event_store_roundtrip(tmp_path):
    st = EventStore(tmp_path)
    eid = st.next_id("cam_01", "20261101T120000")
    assert eid == "evt_cam_01_20261101T120000_01"
    rec = {"event_id": eid, "camera_id": "cam_01", "started_at": "2026-11-01T12:00:00+01:00", "risk_max": 0.8,
           "factors": ["F1", "F9"], "top_factor": "F1", "escalation": {"calls": [{"factor": "F1"}]}}
    st.save(rec)
    assert st.next_id("cam_01", "20261101T120000").endswith("_02")
    rows = st.list("cam_01")
    assert rows[0]["factors"] == "F1,F9" and rows[0]["escalated"] == 1
    assert st.get(eid)["risk_max"] == 0.8
    st.set_status(eid, "acknowledged")
    assert st.list()[0]["status"] == "acknowledged" and st.get(eid)["status"] == "acknowledged"
    assert (tmp_path / "events" / eid / "event.json").exists()


def test_ring_buffer_and_resize():
    rb = RingBuffer(2.0)
    for i in range(100):
        rb.push(i * 0.1, i)
    assert rb.buf[0][0] >= 9.9 - 2.0 - 1e-9 and len(rb.since(9.5)) == 5
    im = np.zeros((1080, 1920, 3), np.uint8)
    assert resize_w(im, 960).shape == (540, 960, 3)


def test_clip_writer_cv2_fallback(tmp_path):
    import cv2
    p = tmp_path / "c.mp4"
    w = ClipWriter(p, 10, (64, 48), prefer_ffmpeg=False)
    for i in range(10):
        w.write(np.full((48, 64, 3), i * 20, np.uint8))
    assert w.close() == "mp4v"
    cap = cv2.VideoCapture(str(p))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10


def test_policy():
    pol = EscalationPolicy(("F1",), (1.0, 1.6), max_per_incident=2, max_per_minute=3)
    assert pol.decide("F1", 0.99, 0.1, 0, 0.0) == (True, "high_severity")
    ok, why = pol.decide("F9", 0.2, 0.16, 0, 1.0)
    assert ok and why.startswith("uncertain")
    ok, why = pol.decide("F9", 0.9, 0.16, 0, 2.0)
    assert not ok and why.startswith("confident")
    assert pol.decide("F1", 0.9, 0.1, 2, 3.0) == (False, "incident_cap")
    assert pol.decide("F1", 0.9, 0.1, 0, 4.0) == (True, "high_severity")
    assert pol.decide("F1", 0.9, 0.1, 0, 5.0) == (False, "minute_cap")
    assert pol.decide("F1", 0.9, 0.1, 0, 70.0)[0]


SAM_LINE = ("<0f>0<|box;x1=100;y1=200;x2=300;y2=400;w=1920;h=1080;c=0.91|><|mask;x=0;y=0;data=10,10,~abc|>,"
            "3<|box;x1=1500;y1=100;x2=1600;y2=200;w=1920;h=1080|><|mask;x=0;y=0;data=5,5,~x|>")


def test_sam_parse_pick_refine():
    b = parse_sam_boxes(SAM_LINE)
    assert len(b) == 2 and b[0]["c"] == 0.91 and b[1]["id"] == 3 and b[1]["c"] is None
    c = Circle(220, 310, 120, 1920, 1080)
    best = pick_box(b, c)
    assert best["id"] == 0
    rc = refine_circle(c, best)
    assert rc.source == "sam3.1" and rc.cx == 200 and rc.cy == 300
    assert pick_box(b, Circle(900, 900, 50, 1920, 1080)) is None
    # boxes reported at another resolution are scaled into the circle's frame
    half = [dict(x, x1=x["x1"] // 2, x2=x["x2"] // 2, y1=x["y1"] // 2, y2=x["y2"] // 2, w=960, h=540) for x in b]
    assert pick_box(half, c)["id"] == 0


def test_verdict_parse():
    v = parse_verdict('Sure: {"confirmed": true, "severity": 4, "reason": "hand at blade", "action": "stop"}')
    assert v["confirmed"] is True and v["severity"] == 4
    assert parse_verdict("no json")["confirmed"] is None


def test_backends_skip_without_key(monkeypatch):
    for k in ("MODEL_API_KEY", "META_API_KEY", "SAM_BACKEND", "MUSE_BACKEND"):
        monkeypatch.delenv(k, raising=False)
    img = np.zeros((10, 10, 3), np.uint8)
    assert SamRefiner().refine(img, "F1", Circle(5, 5, 2, 10, 10)) == {"status": "skipped_no_key"}
    assert MuseVerifier().verify(img, "F1", 0.9, 0.1) == {"status": "skipped_no_key"}
    monkeypatch.setenv("SAM_BACKEND", "off")
    assert SamRefiner().status() == "off"


def test_f1_policy_is_unchanged_and_f5_is_not_auto_escalated():
    pol = EscalationPolicy(("F1",), (1.0, 1.6), max_per_incident=1, max_per_minute=2)
    assert pol.decide("F1", 0.99, 0.1, 0, 0.0) == (True, "high_severity")
    assert pol.decide("F1", 0.99, 0.1, 1, 1.0) == (False, "incident_cap")
    assert pol.decide("F5", 0.99, 0.1, 0, 0.0)[1].startswith("confident")
    assert pol.decide("F2", 0.99, 0.1, 0, 1.0)[1].startswith("confident")
