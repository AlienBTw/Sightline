"""The optional on_step hook (live preview for the backend) reports circles without changing results."""
from sightline_pipeline.config import load_config
from sightline_pipeline.escalation import EscalationPolicy, Escalator, MuseVerifier, SamRefiner
from sightline_pipeline.ingest import VideoSource
from sightline_pipeline.service import SightlinePipeline
from sightline_pipeline.storage import EventStore

from .test_service_fake_model import Cap, FakeModel


def test_on_step_reports_active_circles(tmp_path, monkeypatch):
    for k in ("MODEL_API_KEY", "META_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    cfg = load_config(overrides={"storage": {"data_dir": str(tmp_path)},
                                 "sampling": {"stride_s": 0.5, "sample_interval_s": 0.16},
                                 "events": {"min_threshold": 0.0, "merge_gap_s": 1.0, "ema_alpha": 1.0},
                                 "clips": {"pre_s": 1.0, "post_s": 1.0, "width": 320}})
    esc = Escalator(EscalationPolicy(("F1",)), SamRefiner(), MuseVerifier(), enabled=False)
    pipe = SightlinePipeline(cfg, FakeModel(), store=EventStore(tmp_path), escalator=esc)
    steps = []
    res = pipe.process(VideoSource("synthetic.mp4", realtime=False, open_fn=lambda u: Cap()), camera_id="cam_t",
                       log=lambda m: None, on_step=steps.append)
    assert len(steps) == res["summary"]["model_steps"]
    active = [s for s in steps if s["active"]]
    assert active and all(o["factor"] == "F1" for s in active for o in s["active"])
    assert active[0]["incident"] is not None and active[0]["active"][0]["circle"]["norm"]["cx"] > 0.7
    assert len(res["events"]) == 1


def test_on_step_errors_do_not_stop_the_loop(tmp_path):
    cfg = load_config(overrides={"storage": {"data_dir": str(tmp_path)}, "sampling": {"stride_s": 0.5},
                                 "events": {"min_threshold": 0.0}, "clips": {"width": 320}})
    pipe = SightlinePipeline(cfg, FakeModel(), store=EventStore(tmp_path),
                             escalator=Escalator(EscalationPolicy(("F1",)), SamRefiner(), MuseVerifier(), enabled=False))

    def boom(_):
        raise ValueError("preview broke")
    res = pipe.process(VideoSource("synthetic.mp4", realtime=False, open_fn=lambda u: Cap()), log=lambda m: None,
                       on_step=boom)
    assert res["summary"]["model_steps"] > 10
