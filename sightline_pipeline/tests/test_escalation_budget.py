"""Meta API escalation budget: per camera session and per local day. No network."""
import json
import threading

import pytest

from sightline_pipeline.config import _apply_escalation_budget, load_config
from sightline_pipeline.escalation import (EscalationBudget, EscalationPolicy, Escalator, MuseVerifier,
                                           SamRefiner, skip_entry)
from sightline_pipeline.ingest import VideoSource
from sightline_pipeline.service import SightlinePipeline
from sightline_pipeline.storage import EventStore

from .test_service_fake_model import Cap, FakeModel

DAY = "2026-11-04"


def _budget(tmp_path, session=3, day=30, today=DAY):
    return EscalationBudget(tmp_path, session, day, today=lambda: today)


def _ready(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("SAM_BACKEND", "api")
    monkeypatch.setenv("MUSE_BACKEND", "api")


def test_session_cap_of_three_then_skipped_budget(tmp_path, monkeypatch):
    _ready(monkeypatch)
    budget = _budget(tmp_path)
    esc = Escalator(EscalationPolicy(("F1",), (1.0, 1.6), max_per_incident=10, max_per_minute=10),
                    SamRefiner(), MuseVerifier(), budget=budget)
    submitted, skipped = [], []
    for i in range(4):
        go, why = esc.allow("F1", 0.9, 0.1, 0, float(i))
        (submitted if go else skipped).append(why if go else skip_entry("F1", i, why))
    assert submitted == ["high_severity"] * 3
    assert skipped == [{"factor": "F1", "t": 3, "reason": "session_cap", "status": "skipped_budget"}]
    snap = esc.status()["budget"]
    assert snap == {"session_used": 3, "session_max": 3, "day_used": 3, "day_max": 30}
    assert json.loads((tmp_path / "escalation_budget.json").read_text(encoding="utf-8")) == {
        "date": DAY, "count": 3}


def test_reset_session_keeps_the_daily_count(tmp_path):
    budget = _budget(tmp_path)
    assert budget.try_consume() == (True, None)
    assert budget.try_consume() == (True, None)
    budget.reset_session()
    snap = budget.snapshot()
    assert snap["session_used"] == 0 and snap["day_used"] == 2
    assert budget.try_consume() == (True, None)
    snap = budget.snapshot()
    assert snap["session_used"] == 1 and snap["day_used"] == 3


def test_daily_cap_is_shared_by_two_instances(tmp_path):
    a = EscalationBudget(tmp_path, max_per_session=10, max_per_day=2, today=lambda: DAY)
    b = EscalationBudget(tmp_path, max_per_session=10, max_per_day=2, today=lambda: DAY)
    assert a.try_consume()[0] and b.try_consume()[0]
    assert a.try_consume() == (False, "daily_cap")
    assert b.try_consume() == (False, "daily_cap")
    assert a.snapshot()["day_used"] == 2 and b.snapshot()["day_used"] == 2
    assert a.snapshot()["session_used"] == 1 and b.snapshot()["session_used"] == 1
    assert skip_entry("F9", 1.2, "daily_cap")["status"] == "skipped_budget"


def test_date_rollover_resets_the_daily_count(tmp_path):
    clock = {"day": DAY}
    budget = EscalationBudget(tmp_path, 5, 30, today=lambda: clock["day"])
    assert budget.try_consume() == (True, None)
    clock["day"] = "2026-11-05"
    assert budget.snapshot()["day_used"] == 0
    assert budget.snapshot()["session_used"] == 1
    assert budget.try_consume() == (True, None)
    saved = json.loads((tmp_path / "escalation_budget.json").read_text(encoding="utf-8"))
    assert saved == {"date": "2026-11-05", "count": 1}
    assert budget.snapshot()["session_used"] == 2 and budget.snapshot()["day_used"] == 1


def test_corrupt_file_counts_as_zero_and_is_overwritten(tmp_path):
    path = tmp_path / "escalation_budget.json"
    path.write_text("{not json", encoding="utf-8")
    budget = _budget(tmp_path)
    assert budget.snapshot()["day_used"] == 0
    assert json.loads(path.read_text(encoding="utf-8")) == {"date": DAY, "count": 0}
    path.write_text("[]", encoding="utf-8")
    assert budget.try_consume() == (True, None)
    assert json.loads(path.read_text(encoding="utf-8")) == {"date": DAY, "count": 1}


def test_zero_cap_allows_no_calls(tmp_path):
    budget = EscalationBudget(tmp_path, max_per_session=0, max_per_day=30, today=lambda: DAY)
    assert budget.try_consume() == (False, "session_cap")
    assert not (tmp_path / "escalation_budget.json").exists()
    daily = EscalationBudget(tmp_path, max_per_session=3, max_per_day=0, today=lambda: DAY)
    assert daily.try_consume() == (False, "daily_cap")
    assert not (tmp_path / "escalation_budget.json").exists()


def test_env_overrides_and_invalid_values(monkeypatch):
    monkeypatch.setenv("SIGHTLINE_ESC_MAX_PER_SESSION", "1")
    monkeypatch.setenv("SIGHTLINE_ESC_MAX_PER_DAY", "0")
    cfg = load_config()
    assert cfg["escalation"]["max_per_session"] == 1
    assert cfg["escalation"]["max_per_day"] == 0
    monkeypatch.setenv("SIGHTLINE_ESC_MAX_PER_SESSION", "nope")
    monkeypatch.setenv("SIGHTLINE_ESC_MAX_PER_DAY", "-4")
    with pytest.warns(UserWarning) as caught:
        cfg = load_config()
    text = " ".join(str(w.message) for w in caught)
    assert "SIGHTLINE_ESC_MAX_PER_SESSION" in text and "SIGHTLINE_ESC_MAX_PER_DAY" in text
    assert cfg["escalation"]["max_per_session"] == 3
    assert cfg["escalation"]["max_per_day"] == 30


def test_missing_keys_fall_back_to_defaults(monkeypatch):
    monkeypatch.delenv("SIGHTLINE_ESC_MAX_PER_SESSION", raising=False)
    monkeypatch.delenv("SIGHTLINE_ESC_MAX_PER_DAY", raising=False)
    cfg = {"escalation": {"max_per_incident": 1, "max_per_minute": 2}}
    _apply_escalation_budget(cfg)
    assert cfg["escalation"]["max_per_session"] == 3
    assert cfg["escalation"]["max_per_day"] == 30
    assert cfg["escalation"]["max_per_incident"] == 1
    loaded = load_config()
    assert loaded["escalation"]["max_per_session"] == 3
    assert loaded["escalation"]["max_per_day"] == 30


def test_no_consume_without_api_key_or_when_policy_says_no(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("SAM_BACKEND", "api")
    monkeypatch.setenv("MUSE_BACKEND", "api")
    budget = _budget(tmp_path)
    esc = Escalator(EscalationPolicy(("F1",), (1.0, 1.6), max_per_incident=1, max_per_minute=6),
                    SamRefiner(), MuseVerifier(), budget=budget)
    assert esc.allow("F1", 0.9, 0.1, 0, 0.0) == (True, "high_severity")
    assert budget.snapshot()["session_used"] == 0 and budget.snapshot()["day_used"] == 0
    assert not (tmp_path / "escalation_budget.json").exists()

    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    go, why = esc.allow("F9", 0.9, 0.1, 0, 10.0)
    assert go is False and why.startswith("confident")
    assert esc.allow("F1", 0.9, 0.1, 1, 11.0) == (False, "incident_cap")
    assert budget.snapshot()["day_used"] == 0
    assert esc.allow("F1", 0.9, 0.1, 0, 80.0) == (True, "high_severity")
    assert budget.snapshot()["session_used"] == 1 and budget.snapshot()["day_used"] == 1


def test_concurrent_consumes_do_not_overshoot(tmp_path):
    budget = EscalationBudget(tmp_path, max_per_session=100, max_per_day=20, today=lambda: DAY)
    barrier = threading.Barrier(8)

    def run():
        barrier.wait()
        for _ in range(10):
            budget.try_consume()

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert budget.snapshot() == {"session_used": 20, "session_max": 100, "day_used": 20, "day_max": 20}


def test_pipeline_records_skipped_budget_and_does_not_call_api(tmp_path, monkeypatch):
    _ready(monkeypatch)
    monkeypatch.delenv("SIGHTLINE_ESCALATION", raising=False)
    monkeypatch.delenv("SIGHTLINE_VERIFY_GATES", raising=False)
    import requests

    def _no_api(*_a, **_k):
        raise AssertionError("Meta API must not be called")

    monkeypatch.setattr(requests, "post", _no_api)
    cfg = load_config(overrides={
        "storage": {"data_dir": str(tmp_path)},
        "sampling": {"stride_s": 0.5, "sample_interval_s": 0.16},
        "events": {"min_threshold": 0.0, "merge_gap_s": 1.0, "ema_alpha": 1.0},
        "clips": {"pre_s": 1.0, "post_s": 1.0, "width": 320},
        "escalation": {"enabled": True, "max_per_session": 0, "max_per_day": 30},
    })
    pipe = SightlinePipeline(cfg, FakeModel(), store=EventStore(tmp_path))
    got = []
    res = pipe.process(VideoSource("synthetic.mp4", realtime=False, open_fn=lambda u: Cap()),
                       camera_id="cam_budget", on_event=got.append, log=lambda m: None)
    assert len(res["events"]) == 1 and len(got) == 1
    rec = got[0]
    assert rec["factors"] == ["F1"]
    assert rec["escalation"]["calls"] == []
    assert len(rec["escalation"]["skipped"]) == 1
    skipped = rec["escalation"]["skipped"][0]
    assert skipped["factor"] == "F1" and skipped["reason"] == "session_cap"
    assert skipped["status"] == "skipped_budget" and isinstance(skipped["t"], (int, float))
    assert rec["escalation"]["budget"]["session_used"] == 0
    assert rec["escalation"]["budget"]["session_max"] == 0
    assert rec["escalation"]["budget"]["day_used"] == 0
    assert rec["escalation"]["budget"]["day_max"] == 30
    assert res["summary"]["escalation"]["budget"]["session_max"] == 0
    assert not (tmp_path / "escalation_budget.json").exists()


def test_refused_attempts_do_not_fill_the_minute_window(tmp_path, monkeypatch):
    """Session refusals must not occupy max_per_minute, and a later real call still fits."""
    _ready(monkeypatch)
    budget = _budget(tmp_path, session=1, day=30)
    esc = Escalator(EscalationPolicy(("F1",), (1.0, 1.6), max_per_incident=10, max_per_minute=2),
                    SamRefiner(), MuseVerifier(), budget=budget)
    assert esc.allow("F1", 0.9, 0.1, 0, 0.0) == (True, "high_severity")
    for t in (1.0, 2.0, 3.0):
        assert esc.allow("F1", 0.9, 0.1, 0, t) == (False, "session_cap")
        assert esc.begin_verify(0, t) == (False, "session_cap")
    assert budget.snapshot()["session_used"] == 1 and budget.snapshot()["day_used"] == 1
    budget.reset_session()
    # One real submission is in the window (max 2). The refusals above must not have filled it.
    assert esc.allow("F1", 0.9, 0.1, 0, 4.0) == (True, "high_severity")
    assert budget.snapshot()["session_used"] == 1 and budget.snapshot()["day_used"] == 2
    # Session is full again and the minute window holds the two real calls. Session wins.
    assert esc.allow("F1", 0.9, 0.1, 0, 5.0) == (False, "session_cap")
    assert esc.begin_verify(0, 5.0) == (False, "session_cap")
    assert budget.snapshot()["day_used"] == 2


def test_no_api_call_does_not_take_a_minute_slot(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("SAM_BACKEND", "api")
    monkeypatch.setenv("MUSE_BACKEND", "api")
    budget = _budget(tmp_path)
    esc = Escalator(EscalationPolicy(("F1",), (1.0, 1.6), max_per_incident=5, max_per_minute=1),
                    SamRefiner(), MuseVerifier(), budget=budget)
    assert esc.allow("F1", 0.9, 0.1, 0, 0.0) == (True, "high_severity")
    assert budget.snapshot()["session_used"] == 0 and budget.snapshot()["day_used"] == 0
    _ready(monkeypatch)
    assert esc.allow("F1", 0.9, 0.1, 0, 1.0) == (True, "high_severity")
    assert budget.snapshot()["session_used"] == 1 and budget.snapshot()["day_used"] == 1


def test_most_permanent_cap_is_reported(tmp_path, monkeypatch):
    """daily, then session, then incident, then minute. F1 and the gate use the same name."""
    _ready(monkeypatch)

    daily = EscalationBudget(tmp_path / "daily", max_per_session=1, max_per_day=1, today=lambda: DAY)
    esc = Escalator(EscalationPolicy(("F1",), (1.0, 1.6), max_per_incident=1, max_per_minute=1),
                    SamRefiner(), MuseVerifier(), budget=daily)
    assert esc.allow("F1", 0.9, 0.1, 0, 0.0) == (True, "high_severity")
    daily.reset_session()
    assert esc.allow("F1", 0.9, 0.1, 1, 1.0) == (False, "daily_cap")
    assert esc.begin_verify(1, 1.0) == (False, "daily_cap")
    assert daily.snapshot()["day_used"] == 1 and daily.snapshot()["session_used"] == 0

    both = EscalationBudget(tmp_path / "both", max_per_session=0, max_per_day=0, today=lambda: DAY)
    assert both.try_consume() == (False, "daily_cap")
    assert both.blocking_reason() == "daily_cap"
    assert not (tmp_path / "both" / "escalation_budget.json").exists()

    session = EscalationBudget(tmp_path / "session", max_per_session=1, max_per_day=30, today=lambda: DAY)
    esc_s = Escalator(EscalationPolicy(("F1",), (1.0, 1.6), max_per_incident=1, max_per_minute=1),
                      SamRefiner(), MuseVerifier(), budget=session)
    assert esc_s.allow("F1", 0.9, 0.1, 0, 0.0)[0] is True
    assert esc_s.allow("F1", 0.9, 0.1, 1, 1.0) == (False, "session_cap")
    assert esc_s.begin_verify(1, 1.0) == (False, "session_cap")
    assert session.snapshot()["session_used"] == 1 and session.snapshot()["day_used"] == 1

    room = EscalationBudget(tmp_path / "room", max_per_session=5, max_per_day=30, today=lambda: DAY)
    esc_i = Escalator(EscalationPolicy(("F1",), (1.0, 1.6), max_per_incident=1, max_per_minute=1),
                      SamRefiner(), MuseVerifier(), budget=room)
    assert esc_i.allow("F1", 0.9, 0.1, 0, 0.0)[0] is True
    assert esc_i.allow("F1", 0.9, 0.1, 1, 1.0) == (False, "incident_cap")
    assert esc_i.begin_verify(1, 1.0) == (False, "incident_cap")
    assert room.snapshot()["session_used"] == 1

    minute = EscalationBudget(tmp_path / "minute", max_per_session=5, max_per_day=30, today=lambda: DAY)
    esc_m = Escalator(EscalationPolicy(("F1",), (1.0, 1.6), max_per_incident=5, max_per_minute=1),
                      SamRefiner(), MuseVerifier(), budget=minute)
    assert esc_m.allow("F1", 0.9, 0.1, 0, 0.0)[0] is True
    assert esc_m.allow("F1", 0.9, 0.1, 0, 1.0) == (False, "minute_cap")
    assert esc_m.begin_verify(0, 1.0) == (False, "minute_cap")
    assert minute.snapshot()["day_used"] == 1

    stored = skip_entry("F1", 1.2, "rate_limited")
    assert stored == {"factor": "F1", "t": 1.2, "reason": "minute_cap", "status": "skipped_budget"}
    assert skip_entry("F1", 1.2, "minute_cap") == stored
