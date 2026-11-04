"""End-to-end service test with a fake model (no GPU): synthetic video -> incident -> event files."""
import json

import numpy as np

from sightline_pipeline.config import load_config
from sightline_pipeline.escalation import EscalationPolicy, Escalator, MuseVerifier, SamRefiner
from sightline_pipeline.ingest import VideoSource
from sightline_pipeline.service import SightlinePipeline, overlays_at
from sightline_pipeline.storage import EventStore

FACTORS = [f"F{i}" for i in range(1, 11)]


class FakeModel:
    """F1 fires when the frame is bright; its map peaks at the top-right corner."""
    factors = FACTORS
    ckpt_thresholds = {f: 0.5 for f in FACTORS}
    n_frames = 4

    def preprocess(self, bgr):
        return bgr[::4, ::4]

    def step(self, window):
        bright = float(window[-1].mean()) > 100
        probs = {f: 0.01 for f in FACTORS}
        probs["F1"] = 0.95 if bright else 0.02
        m = np.zeros((10, 12, 21), np.float32)
        m[0, 2, 17] = 1.0
        return {"probs": probs, "evidence": m, "attention": m, "latency_ms": {"encoder": 5.0, "head": 1.0}}

    def factor_map(self, out, f, mode):
        return out["evidence"][FACTORS.index(f)]


class Cap:
    def __init__(self, n=250):
        self.i, self.n = 0, n

    def isOpened(self):
        return True

    def get(self, p):
        return {5: 25.0, 3: 320, 4: 180}.get(p, 0)

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        v = 200 if 75 <= self.i < 150 else 20            # bright (hazard) from 3 s to 6 s
        return True, np.full((180, 320, 3), v, np.uint8)

    def release(self):
        pass


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    for k in ("MODEL_API_KEY", "META_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    cfg = load_config(overrides={"storage": {"data_dir": str(tmp_path)},
                                 "sampling": {"stride_s": 0.5, "sample_interval_s": 0.16},
                                 "events": {"min_threshold": 0.0, "merge_gap_s": 1.0, "ema_alpha": 1.0},
                                 "clips": {"pre_s": 1.0, "post_s": 1.0, "width": 320}})
    esc = Escalator(EscalationPolicy(("F1",)), SamRefiner(), MuseVerifier(), enabled=True)
    pipe = SightlinePipeline(cfg, FakeModel(), store=EventStore(tmp_path), escalator=esc)
    got = []
    src = VideoSource("synthetic.mp4", realtime=False, open_fn=lambda u: Cap())
    res = pipe.process(src, camera_id="cam_test", on_event=got.append, session_video=str(tmp_path / "s.mp4"),
                       log=lambda m: None)
    assert len(res["events"]) == 1 and len(got) == 1
    rec = got[0]
    assert rec["factors"] == ["F1"] and rec["top_factor"] == "F1"
    assert "verify_gates" not in rec["escalation"]
    assert 3.0 <= rec["stream_start_s"] <= 4.5 and 5.9 <= rec["stream_end_s"] <= 7.5
    assert rec["escalation"]["calls"][0]["sam3.1"] == {"status": "skipped_no_key"}
    ov = rec["overlays"][0]["circle"]["norm"]
    assert ov["cx"] > 0.7 and ov["cy"] < 0.4                 # top-right, where the fake map peaks
    d = tmp_path / "events" / rec["event_id"]
    for f in ("clip.mp4", "clip_circles.mp4", "thumb.jpg", "event.json"):
        assert (d / f).exists() and (d / f).stat().st_size > 0, f
    assert json.load(open(d / "event.json"))["event_id"] == rec["event_id"]
    assert EventStore(tmp_path).list()[0]["event_id"] == rec["event_id"]
    assert res["summary"]["model_steps"] > 10 and res["summary"]["latency_ms"]["total_p50"] == 6.0


def test_overlays_at_holds_last_step():
    ovs = [{"t_ms": 0, "factor": "F1"}, {"t_ms": 1000, "factor": "F1"}, {"t_ms": 1000, "factor": "F9"}]
    assert len(overlays_at(ovs, 1200)) == 2 and overlays_at(ovs, 500)[0]["t_ms"] == 0
    assert overlays_at(ovs, 4000) == []
