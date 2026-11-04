"""F2/F5 Muse Spark verify gates. The HTTP client is mocked; nothing here calls the network."""
import base64
import json

import numpy as np

from sightline_pipeline.config import _apply_verify_gates, load_config
from sightline_pipeline.escalation import (EscalationBudget, EscalationPolicy, Escalator, MuseVerifier,
                                           SamRefiner, jpeg_data_url)
from sightline_pipeline.events import IncidentTracker
from sightline_pipeline.ingest import VideoSource
from sightline_pipeline.service import SightlinePipeline
from sightline_pipeline.storage import EventStore
from sightline_pipeline.verify_gate import (VerifyClient, VerifyGates, apply_verify_result, gate_skip_entry,
                                            hazard_votes, parse_answers, prompt_for, sample_evenly,
                                            select_gate_frames)
from sightline_pipeline.verify_gate_eval import main as eval_main

from .test_service_fake_model import Cap, FakeModel

DAY = "2026-11-04"


def _quiet(monkeypatch):
    for key in ("MODEL_API_KEY", "META_API_KEY", "MUSE_SPARK_MODEL", "SIGHTLINE_VERIFY_GATES",
                "SIGHTLINE_VERIFY_GATE_F2", "SIGHTLINE_VERIFY_GATE_F5", "SIGHTLINE_F5_VLM_GATE",
                "SIGHTLINE_F5_MUSE_GATE", "SIGHTLINE_ESCALATION", "SIGHTLINE_ESC_MAX_PER_SESSION",
                "SIGHTLINE_ESC_MAX_PER_DAY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MUSE_BACKEND", "api")
    monkeypatch.setenv("SAM_BACKEND", "api")


def _frame(person, vest=None, head=None):
    row = {"person_visible": person}
    if vest is not None:
        row["wearing_hi_vis_vest"] = vest
    if head is not None:
        row["wearing_head_protection"] = head
    return row


def _payload(frames):
    return json.dumps({"frames": frames})


class Http:
    def __init__(self, content, status=200):
        self.status_code = status
        self.headers = {"content-type": "application/json"}
        if isinstance(content, str):
            self.text = json.dumps({"choices": [{"message": {"content": content}}]})
        else:
            self.text = json.dumps(content)

    def json(self):
        return json.loads(self.text)


def _images(n=3, w=32, h=16):
    return [np.full((h, w, 3), i * 40, np.uint8) for i in range(n)]


def _timed(n=3, t0=0.0, step=1.0, w=32, h=16):
    return [{"image": im, "t": t0 + i * step} for i, im in enumerate(_images(n, w, h))]


def _answers(n, votes, field="wearing_hi_vis_vest"):
    out = []
    for i in range(n):
        row = {"person_visible": True, field: i >= votes}
        out.append(row)
    return out


def test_config_defaults_and_env(monkeypatch):
    _quiet(monkeypatch)
    cfg = load_config()
    gate = cfg["escalation"]["verify_gates"]
    assert gate["enabled"] is True
    assert gate["max_frames"] == 3 and gate["long_side"] == 768 and gate["min_votes"] == 2
    assert gate["max_tokens"] == 4000 and gate["recent_s"] == 3 and gate["sam_hold_s"] == 1.5
    assert gate["factors"]["F5"]["trigger_threshold"] == 0.15
    assert gate["factors"]["F5"]["sam_circle"] is True
    assert gate["factors"]["F5"]["sam_prompts"] == ["person"]
    assert gate["factors"]["F2"]["trigger_threshold"] == 0.5
    assert gate["factors"]["F2"]["sam_circle"] is True
    assert gate["factors"]["F2"]["sam_prompts"] == ["head", "person"]
    assert cfg["model"]["head_checkpoint"].endswith("h3_wide.pt")
    assert "F2" not in cfg["events"]["muted_factors"] and "F5" not in cfg["events"]["muted_factors"]
    assert "F6" in cfg["events"]["muted_factors"]
    assert "f5_muse_gate" not in cfg["escalation"] and "f5_vlm_gate" not in cfg["escalation"]
    assert cfg["escalation"]["muse_model"] == "muse-spark-1.3-contributor"
    blank = load_config(overrides={"escalation": {"muse_model": "  "}})
    assert blank["escalation"]["muse_model"] == "muse-spark-1.3-contributor"
    sparse = {"escalation": {}}
    _apply_verify_gates(sparse)
    filled = sparse["escalation"]["verify_gates"]
    assert filled["min_votes"] == 2 and filled["max_tokens"] == 4000 and filled["recent_s"] == 3
    assert filled["sam_hold_s"] == 1.5
    assert filled["factors"]["F2"]["trigger_threshold"] == 0.5
    assert filled["factors"]["F2"]["sam_prompts"] == ["head", "person"]
    assert filled["factors"]["F5"]["trigger_threshold"] == 0.15
    assert filled["factors"]["F5"]["sam_circle"] is True
    assert filled["factors"]["F5"]["sam_prompts"] == ["person"]
    monkeypatch.setenv("SIGHTLINE_VERIFY_GATES", "off")
    assert load_config()["escalation"]["verify_gates"]["enabled"] is False
    monkeypatch.setenv("SIGHTLINE_VERIFY_GATES", "on")
    monkeypatch.setenv("SIGHTLINE_VERIFY_GATE_F2", "off")
    gated = load_config()["escalation"]["verify_gates"]
    assert gated["enabled"] is True
    assert gated["factors"]["F2"]["enabled"] is False
    assert gated["factors"]["F5"]["enabled"] is True
    monkeypatch.setenv("SIGHTLINE_VERIFY_GATE_F5", "off")
    assert load_config()["escalation"]["verify_gates"]["factors"]["F5"]["enabled"] is False


def test_parse_vote_sample_and_prompts():
    assert sample_evenly([0, 1, 2, 3, 4, 5], 3) == [0, 2, 5]
    assert sample_evenly([0, 1], 3) == [0, 1]
    text = _payload([
        _frame(True, vest=False), _frame(True, vest=True), _frame(False, vest=False),
    ])
    answers = parse_answers(text, n=3, fields=["wearing_hi_vis_vest"])
    assert hazard_votes(answers, "wearing_hi_vis_vest") == 1
    assert hazard_votes([_frame(False, vest=False), _frame(False, head=False)], "wearing_head_protection") == 0
    assert parse_answers("nope", n=1) is None
    assert parse_answers(_payload([{"person_visible": "maybe", "wearing_hi_vis_vest": False}]), n=1) is None
    f5 = prompt_for(["F5"], 2)
    assert "fluorescent yellow" in f5 and "wearing_hi_vis_vest" in f5 and "wearing_head_protection" not in f5
    f2 = prompt_for(["F2"], 2)
    assert "bump cap" in f2 and "hair net" in f2 and "wearing_hi_vis_vest" not in f2
    both = prompt_for(["F2", "F5"], 1)
    assert "wearing_hi_vis_vest" in both and "wearing_head_protection" in both
    wide = np.zeros((400, 1000, 3), np.uint8)
    url = jpeg_data_url(wide, long_side=768)
    raw = base64.b64decode(url.split(",", 1)[1])
    import cv2
    im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert max(im.shape[:2]) == 768


def test_two_of_three_confirms_and_one_rejects(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_SPARK_MODEL", raising=False)
    seen = {}

    def post(url, json=None, timeout=None, headers=None, **_k):
        seen["url"] = url
        seen["json"] = json
        seen["headers"] = headers
        n = sum(1 for p in json["messages"][0]["content"] if p.get("type") == "image_url")
        assert n == 3
        return Http(_payload(_answers(n, 2)))

    gate = VerifyGates(enabled=True, client=VerifyClient(post=post, long_side=768))
    result = gate.evaluate("inc-1", ["F5"], _images(5), lambda: (True, "verify_gate"), t_stream=1.0)
    f5 = result["factors"]["F5"]
    assert f5["status"] == "confirmed" and f5["votes"] == 2 and f5["verified_by"] == "Muse Spark"
    assert result["api_calls"] == {"muse": 1, "sam": 0}
    assert seen["url"].endswith("/chat/completions")
    assert seen["json"]["model"] == "muse-spark-1.3-contributor"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert "wearing_head_protection" not in seen["json"]["messages"][0]["content"][0]["text"]

    def post_one(url, json=None, timeout=None, headers=None, **_k):
        n = sum(1 for p in json["messages"][0]["content"] if p.get("type") == "image_url")
        return Http(_payload(_answers(n, 1)))

    rejected = VerifyGates(enabled=True, client=VerifyClient(post=post_one)).evaluate(
        "inc-2", ["F5"], _images(3), lambda: (True, "verify_gate"))
    assert rejected["factors"]["F5"]["status"] == "rejected"
    assert rejected["factors"]["F5"]["votes"] == 1
    assert rejected["factors"]["F5"]["verified_by"] is None


def test_muse_model_is_contributor_and_env_overrides(monkeypatch, tmp_path):
    """Request body model is Muse Spark 1.3 Contributor unless MUSE_SPARK_MODEL is set."""
    import requests
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_SPARK_MODEL", raising=False)
    seen = {}

    def verdict_post(url, json=None, timeout=None, headers=None, **_k):
        seen["verdict"] = json
        return Http('{"confirmed": true, "severity": 3, "reason": "hand at the blade", "action": "stop"}')

    monkeypatch.setattr(requests, "post", verdict_post)
    img = np.zeros((8, 8, 3), np.uint8)
    verdict = MuseVerifier().verify(img, "F1", 0.9, 0.2)
    assert seen["verdict"]["model"] == "muse-spark-1.3-contributor"
    assert verdict["model"] == "muse-spark-1.3-contributor"

    def gate_post(url, json=None, timeout=None, headers=None, **_k):
        seen["gate"] = json
        n = sum(1 for p in json["messages"][0]["content"] if p.get("type") == "image_url")
        return Http(_payload(_answers(n, n)))

    gate = VerifyGates(enabled=True, client=VerifyClient(post=gate_post))
    result = gate.evaluate("inc", ["F5"], _images(3), lambda: (True, "verify_gate"))
    assert seen["gate"]["model"] == "muse-spark-1.3-contributor"
    assert result["calls"][0]["muse_spark"]["model"] == "muse-spark-1.3-contributor"

    monkeypatch.setenv("MUSE_SPARK_MODEL", "muse-override")
    verdict = MuseVerifier(model="from-config").verify(img, "F1", 0.9, 0.2)
    assert seen["verdict"]["model"] == "muse-override"
    assert verdict["model"] == "muse-override"
    gate = VerifyGates(enabled=True, client=VerifyClient(post=gate_post, model="from-config"))
    result = gate.evaluate("inc-env", ["F5"], _images(3), lambda: (True, "verify_gate"))
    assert seen["gate"]["model"] == "muse-override"
    assert result["calls"][0]["muse_spark"]["model"] == "muse-override"

    monkeypatch.delenv("MUSE_SPARK_MODEL", raising=False)
    configured = MuseVerifier(model="from-config").verify(img, "F1", 0.9, 0.2)
    assert seen["verdict"]["model"] == "from-config"
    assert configured["model"] == "from-config"

    root = tmp_path / "frames"
    root.mkdir()
    import cv2
    cv2.imwrite(str(root / "a.jpg"), np.zeros((8, 8, 3), np.uint8))
    labels = tmp_path / "labels.csv"
    labels.write_text("file,head\na.jpg,0\n", encoding="utf-8")
    monkeypatch.setenv("MUSE_SPARK_MODEL", "from-env")

    def eval_post(url, json=None, timeout=None, headers=None, **_k):
        seen["eval"] = json["model"]
        return Http(_payload([_frame(True, head=False)]))

    monkeypatch.setattr(requests, "post", eval_post)
    assert eval_main(["--frames", str(root), "--labels", str(labels), "--factor", "F2",
                      "--batch-size", "1"]) == 0
    assert seen["eval"] == "from-env"
    assert eval_main(["--frames", str(root), "--labels", str(labels), "--factor", "F2",
                      "--batch-size", "1", "--model", "from-flag"]) == 0
    assert seen["eval"] == "from-flag"


def test_person_not_visible_does_not_count():
    answers = [_frame(False, vest=False), _frame(False, vest=False), _frame(True, vest=False)]
    assert hazard_votes(answers, "wearing_hi_vis_vest") == 1


def _marked(image, scores, t, pre_roll=False):
    row = {"image": image, "scores": scores, "t": t}
    if pre_roll:
        row["pre_roll"] = True
    return row


def test_sampling_ignores_preroll_and_below_trigger():
    imgs = [np.full((16, 32, 3), 20 + i, np.uint8) for i in range(8)]
    samples = [
        _marked(imgs[0], {"F2": 0.95, "F5": 0.9}, 0.0, pre_roll=True),
        _marked(imgs[1], {"F2": 0.2, "F5": 0.05}, 1.0),
        _marked(imgs[2], {"F2": 0.8, "F5": 0.4}, 2.0),
        _marked(imgs[3], {"F2": 0.8, "F5": 0.4}, 8.0),
        _marked(imgs[4], {"F2": 0.85, "F5": 0.5}, 8.5),
        _marked(imgs[5], {"F2": 0.9, "F5": 0.6}, 9.0),
        _marked(imgs[6], {"F2": 0.92, "F5": 0.7}, 9.5),
        _marked(imgs[7], {"F2": 0.99, "F5": 0.8}, 10.0),
    ]
    picked = select_gate_frames(samples, ["F2"], {"F2": 0.5}, 3, recent_s=3.0)
    assert [id(s["image"]) for s in picked] == [id(imgs[3]), id(imgs[5]), id(imgs[7])]
    both = select_gate_frames(samples, ["F2", "F5"], {"F2": 0.5, "F5": 0.15}, 3, recent_s=3.0)
    assert [id(s["image"]) for s in both] == [id(imgs[3]), id(imgs[5]), id(imgs[7])]
    quiet = select_gate_frames(
        [_marked(imgs[0], {"F2": 0.1}, 4.0), _marked(imgs[1], {"F2": 0.9}, 1.0, pre_roll=True)],
        ["F2"], {"F2": 0.5}, 3, recent_s=3.0)
    assert quiet == []

    class Capture:
        model = "muse-spark-1.3-contributor"

        def ask(self, images, factors=("F5",)):
            self.images = list(images)
            n = len(images)
            frames = [_frame(True, head=False, vest=False) for _ in range(n)]
            return {"status": "ok", "raw": _payload(frames), "answers": frames, "model": self.model,
                    "factors": list(factors)}

    cap = Capture()

    def begin():
        raise AssertionError("no eligible frame must not spend the budget")

    empty = VerifyGates(enabled=True, client=cap).evaluate("none", ["F2"], quiet, begin)
    assert empty["factors"]["F2"]["status"] == "failed"
    assert "no frames" in empty["factors"]["F2"]["reason"]

    sent = VerifyGates(enabled=True, client=cap).evaluate(
        "late", ["F2"], samples, lambda: (True, "verify_gate"))
    assert sent["factors"]["F2"]["status"] == "confirmed"
    assert [id(im) for im in cap.images] == [id(imgs[3]), id(imgs[5]), id(imgs[7])]


def test_fail_closed(monkeypatch, caplog):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("META_API_KEY", raising=False)
    caplog.set_level("WARNING", logger="sightline.escalation")

    def boom(*_a, **_k):
        raise AssertionError("HTTP client must not be called")

    gate = VerifyGates(enabled=True, client=VerifyClient(post=boom), min_votes=2)
    missing = gate.evaluate("a", ["F5"], _images(3), lambda: (False, "skipped_no_key"))
    assert missing["factors"]["F5"]["status"] == "skipped_no_key"
    assert missing["api_calls"]["muse"] == 0
    short = gate.evaluate("b", ["F5"], _images(1), lambda: (_ for _ in ()).throw(AssertionError("cap check must not run")))
    assert short["factors"]["F5"]["status"] == "failed"

    def timeout(*_a, **_k):
        import requests
        raise requests.Timeout("slow")

    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    timed = VerifyGates(enabled=True, client=VerifyClient(post=timeout)).evaluate(
        "c", ["F5"], _images(3), lambda: (True, "verify_gate"))
    assert timed["factors"]["F5"]["status"] == "failed" and "timed out" in timed["factors"]["F5"]["reason"]
    parsed = VerifyGates(enabled=True, client=VerifyClient(post=lambda *a, **k: Http("not json"))).evaluate(
        "d", ["F5"], _images(3), lambda: (True, "verify_gate"))
    assert parsed["factors"]["F5"]["status"] == "failed"
    wrong = VerifyGates(enabled=True, client=VerifyClient(post=lambda *a, **k: Http(_payload(_answers(1, 1))))).evaluate(
        "e", ["F5"], _images(3), lambda: (True, "verify_gate"))
    assert wrong["factors"]["F5"]["status"] == "failed" and "wrong number" in wrong["factors"]["F5"]["reason"]
    http = VerifyGates(enabled=True, client=VerifyClient(post=lambda *a, **k: Http("nope", status=500))).evaluate(
        "f", ["F5"], _images(3), lambda: (True, "verify_gate"))
    assert http["factors"]["F5"]["status"] == "failed" and "HTTP 500" in http["factors"]["F5"]["reason"]
    bare = VerifyClient(post=boom)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert bare.ask(_images(1), ("F5",))["status"] == "skipped_no_key"
    assert "fail closed" in caplog.text


def test_combined_request_is_one_call_and_cached(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    calls = {"post": 0, "begin": 0}

    def post(url, json=None, timeout=None, headers=None, **_k):
        calls["post"] += 1
        text = json["messages"][0]["content"][0]["text"]
        assert "wearing_hi_vis_vest" in text and "wearing_head_protection" in text
        n = sum(1 for p in json["messages"][0]["content"] if p.get("type") == "image_url")
        frames = [_frame(True, vest=False, head=False) for _ in range(n)]
        return Http(_payload(frames))

    def begin():
        calls["begin"] += 1
        return True, "verify_gate"

    gate = VerifyGates(enabled=True, client=VerifyClient(post=post))
    first = gate.evaluate("same", ["F2", "F5"], _images(3), begin)
    second = gate.evaluate("same", ["F5", "F2"], _images(3), begin)
    assert first["factors"]["F2"]["status"] == "confirmed"
    assert first["factors"]["F5"]["status"] == "confirmed"
    assert second["cached"] is True
    assert calls == {"post": 1, "begin": 1}
    assert first["api_calls"]["muse"] == 1


def test_caps_budget_and_session_fourth_incident(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setenv("MODEL_API_KEY", "test-key")

    def boom(*_a, **_k):
        raise AssertionError("HTTP client must not be called")

    monkeypatch.setattr("requests.post", boom)
    budget = EscalationBudget(tmp_path, max_per_session=3, max_per_day=30, today=lambda: DAY)
    esc = Escalator(EscalationPolicy(("F1",), (1.0, 1.6), max_per_incident=1, max_per_minute=2),
                    SamRefiner(), MuseVerifier(), budget=budget)
    assert esc.allow("F1", 0.9, 0.1, 0, 0.0) == (True, "high_severity")
    assert esc.begin_verify(1, 1.0) == (False, "incident_cap")
    assert budget.snapshot()["day_used"] == 1

    minute = Escalator(EscalationPolicy(("F1",), (1.0, 1.6), max_per_incident=5, max_per_minute=1),
                       SamRefiner(), MuseVerifier(),
                       budget=EscalationBudget(tmp_path / "m", 5, 30, today=lambda: DAY))
    assert minute.allow("F1", 0.9, 0.1, 0, 0.0)[0] is True
    assert minute.begin_verify(0, 1.0) == (False, "minute_cap")

    blocked = EscalationBudget(tmp_path / "b", max_per_session=0, max_per_day=30, today=lambda: DAY)
    esc_b = Escalator(EscalationPolicy(max_per_incident=3, max_per_minute=6), SamRefiner(), MuseVerifier(), budget=blocked)
    assert esc_b.begin_verify(0, 0.0) == (False, "session_cap")
    assert blocked.snapshot()["day_used"] == 0 and blocked.snapshot()["session_used"] == 0
    daily = EscalationBudget(tmp_path / "d", max_per_session=3, max_per_day=0, today=lambda: DAY)
    esc_d = Escalator(EscalationPolicy(max_per_incident=3, max_per_minute=6), SamRefiner(), MuseVerifier(), budget=daily)
    assert esc_d.begin_verify(0, 0.0) == (False, "daily_cap")
    assert not (tmp_path / "d" / "escalation_budget.json").exists()

    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    bare = Escalator(EscalationPolicy(max_per_incident=3, max_per_minute=6), SamRefiner(), MuseVerifier(),
                     budget=EscalationBudget(tmp_path / "k", 3, 30, today=lambda: DAY))
    assert bare.begin_verify(0, 0.0) == (False, "skipped_no_key")
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    assert bare.begin_verify(0, 1.0)[0] is True

    def post(url, json=None, timeout=None, headers=None, **_k):
        post.n += 1
        n = sum(1 for p in json["messages"][0]["content"] if p.get("type") == "image_url")
        return Http(_payload(_answers(n, n)))
    post.n = 0
    gate = VerifyGates(enabled=True, client=VerifyClient(post=post))
    session = EscalationBudget(tmp_path / "s", max_per_session=3, max_per_day=30, today=lambda: DAY)
    runner = Escalator(EscalationPolicy(max_per_incident=3, max_per_minute=10), SamRefiner(), MuseVerifier(), budget=session)
    for i in range(3):
        got = gate.evaluate(f"inc-{i}", ["F5"], _images(3), lambda t=i * 120.0: runner.begin_verify(0, t))
        assert got["factors"]["F5"]["status"] == "confirmed"
    fourth = gate.evaluate("inc-3", ["F5"], _images(3), lambda: runner.begin_verify(0, 1000.0))
    assert fourth["factors"]["F5"]["status"] == "skipped_budget"
    assert fourth["factors"]["F5"]["reason"] == "session_cap"
    assert fourth["skipped"][0]["status"] == "skipped_budget" and fourth["skipped"][0]["reason"] == "session_cap"
    assert post.n == 3
    refused = gate.evaluate("daily", ["F2"], _images(3), lambda: esc_d.begin_verify(0, 5.0))
    assert refused["factors"]["F2"]["status"] == "skipped_budget" and refused["factors"]["F2"]["reason"] == "daily_cap"
    capped = gate.evaluate("slot", ["F5"], _images(3), lambda: (False, "incident_cap"))
    assert capped["factors"]["F5"]["status"] == "skipped_budget" and capped["factors"]["F5"]["reason"] == "incident_cap"


def test_session_refusals_stay_session_cap_inside_the_minute(tmp_path, monkeypatch):
    """After the session cap, later gate skips stay session_cap and do not fill the minute window."""
    _quiet(monkeypatch)
    monkeypatch.setenv("MODEL_API_KEY", "test-key")

    def post(url, json=None, timeout=None, headers=None, **_k):
        post.n += 1
        n = sum(1 for p in json["messages"][0]["content"] if p.get("type") == "image_url")
        return Http(_payload(_answers(n, n)))

    post.n = 0
    budget = EscalationBudget(tmp_path, max_per_session=1, max_per_day=30, today=lambda: DAY)
    runner = Escalator(EscalationPolicy(max_per_incident=5, max_per_minute=2),
                       SamRefiner(), MuseVerifier(), budget=budget)
    gate = VerifyGates(enabled=True, client=VerifyClient(post=post))
    first = gate.evaluate("inc-0", ["F5"], _images(3), lambda: runner.begin_verify(0, 0.0))
    assert first["factors"]["F5"]["status"] == "confirmed"
    for i, t in enumerate((1.0, 2.0, 3.0), start=1):
        got = gate.evaluate(f"inc-{i}", ["F5"], _images(3), lambda t=t: runner.begin_verify(0, t))
        assert got["factors"]["F5"]["status"] == "skipped_budget"
        assert got["factors"]["F5"]["reason"] == "session_cap"
        assert got["skipped"][0]["reason"] == "session_cap" and got["skipped"][0]["status"] == "skipped_budget"
    assert post.n == 1
    assert budget.snapshot()["session_used"] == 1 and budget.snapshot()["day_used"] == 1
    budget.reset_session()
    again = gate.evaluate("inc-again", ["F5"], _images(3), lambda: runner.begin_verify(0, 4.0))
    assert again["factors"]["F5"]["status"] == "confirmed"
    assert post.n == 2

    old = gate.evaluate("legacy", ["F2"], _images(3), lambda: (False, "rate_limited"))
    assert old["factors"]["F2"]["status"] == "skipped_budget" and old["factors"]["F2"]["reason"] == "minute_cap"
    assert old["skipped"][0]["reason"] == "minute_cap"
    assert gate_skip_entry("F5", 1.0, "rate_limited")["reason"] == "minute_cap"
    assert gate_skip_entry("F5", 1.0, "minute_cap")["reason"] == "minute_cap"


class _BoxSam:
    def __init__(self, boxes_by_prompt):
        self.calls = []
        self.boxes_by_prompt = boxes_by_prompt

    def segment(self, img, prompt):
        self.calls.append(prompt)
        boxes = self.boxes_by_prompt.get(prompt, [])
        return {"status": "ok" if boxes else "no_match", "boxes": boxes, "prompt": prompt, "sent": True}


def _f2_record():
    circle = {"cx": 10, "cy": 40, "r": 8, "w": 100, "h": 80, "source": "head", "score": 0.4,
              "norm": {"cx": 0.1, "cy": 0.5, "rx": 0.08, "ry": 0.1, "angle_deg": 0}}
    return {"factors": ["F2"], "top_factor": "F2", "status": "open", "risk_max": 0.8,
            "segments": [{"factor": "F2", "peak_score": 0.8, "threshold": 0.5, "peak_t": 1.0, "start_s": 0.0}],
            "overlays": [{"factor": "F2", "t_ms": 100, "score": 0.8, "circle": dict(circle)}],
            "localization": {"F2": "weak"}, "escalation": {}, "clip_t0_s": 0.0}


def _head_circle():
    return {"cx": 4, "cy": 4, "r": 3, "w": 100, "h": 80, "source": "head", "score": 0.2,
            "norm": {"cx": 0.04, "cy": 0.05, "rx": 0.03, "ry": 0.04, "angle_deg": 0}}


def _confirm_post(vest=False, head=False):
    def post(url, json=None, timeout=None, headers=None, **_k):
        n = sum(1 for p in json["messages"][0]["content"] if p.get("type") == "image_url")
        return Http(_payload([_frame(True, vest=vest, head=head) for _ in range(n)]))
    return post


def test_sam_head_circle_stays_near_the_sam_frame(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    upper = {"id": 1, "x1": 10, "y1": 2, "x2": 70, "y2": 40, "w": 100, "h": 80, "c": 0.40}
    chosen = {"id": 2, "x1": 40, "y1": 50, "x2": 60, "y2": 70, "w": 100, "h": 80, "c": 0.95}
    sam = _BoxSam({"head": [upper, chosen]})
    gate = VerifyGates(enabled=True, client=VerifyClient(post=_confirm_post(head=False)))
    result = gate.evaluate("sam", ["F2"], _timed(3, t0=10.0, step=1.0), lambda: (True, "verify_gate"), sam=sam)
    assert sam.calls == ["head"]
    assert result["api_calls"] == {"muse": 1, "sam": 1}
    f2 = result["factors"]["F2"]
    assert f2["circle_source"] == "sam3.1" and f2["sam_status"] == "ok" and f2["sam_prompt"] == "head"
    assert f2["sam_boxes"] == {"head": 2}
    assert f2["sam_t_stream"] == 12.0
    assert f2["circle"]["cx"] == 50 and f2["circle"]["cy"] == 60 and f2["circle"]["source"] == "sam3.1"
    wide = dict(_head_circle(), w=200, h=160, cx=8, cy=8)
    rec = _f2_record()
    rec["overlays"] = [
        {"factor": "F2", "t_stream": 12.0, "t_ms": 2000, "score": 0.8, "circle": _head_circle()},
        {"factor": "F2", "t_stream": 11.0, "t_ms": 1000, "score": 0.7, "circle": dict(wide)},
        {"factor": "F2", "t_stream": 9.0, "t_ms": 0, "score": 0.6, "circle": _head_circle()},
        {"factor": "F1", "t_stream": 12.0, "t_ms": 2000, "score": 0.9,
         "circle": {"cx": 1, "cy": 1, "r": 1, "w": 100, "h": 80, "source": "head", "norm": {"cx": 0.01}}},
    ]
    assert apply_verify_result(rec, result) is True
    f2_ovs = [o for o in rec["overlays"] if o["factor"] == "F2"]
    assert len(f2_ovs) == 3
    assert f2_ovs[0]["circle"]["source"] == "sam3.1" and f2_ovs[0]["circle"]["cx"] == 50 and f2_ovs[0]["circle"]["w"] == 100
    assert f2_ovs[1]["circle"]["source"] == "sam3.1" and f2_ovs[1]["circle"]["w"] == 200
    assert abs(f2_ovs[1]["circle"]["norm"]["cx"] - 0.5) < 0.02
    assert f2_ovs[2]["circle"] is None
    assert rec["overlays"][3]["circle"]["source"] == "head"
    stored = rec["escalation"]["verify_gates"]["factors"]["F2"]
    assert stored["circle_source"] == "sam3.1" and stored["sam_prompt"] == "head"
    assert stored["sam_t_stream"] == 12.0
    assert stored["sam_boxes"] == {"head": 2}
    assert rec["escalation"]["verify_gates"]["api_calls"] == {"muse": 1, "sam": 1}

    tied_small = {"id": 3, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "w": 100, "h": 80, "c": 0.5}
    tied_large = {"id": 4, "x1": 20, "y1": 20, "x2": 60, "y2": 70, "w": 100, "h": 80, "c": 0.5}
    tie = _BoxSam({"head": [tied_small, tied_large]})
    tied = VerifyGates(enabled=True, client=VerifyClient(post=_confirm_post(head=False))).evaluate(
        "tie", ["F2"], _images(3), lambda: (True, "verify_gate"), sam=tie)
    assert tied["factors"]["F2"]["circle"]["cx"] == 40 and tied["factors"]["F2"]["circle"]["cy"] == 45


def test_person_box_draws_f2_head_and_f5_torso(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    person = {"id": 1, "x1": 0, "y1": 10, "x2": 100, "y2": 110, "w": 100, "h": 100, "c": 0.88}
    sam = _BoxSam({"head": [], "person": [person]})
    result = VerifyGates(enabled=True, client=VerifyClient(post=_confirm_post(head=False))).evaluate(
        "region", ["F2"], _images(3), lambda: (True, "verify_gate"), sam=sam)
    assert sam.calls == ["head", "person"]
    f2 = result["factors"]["F2"]
    assert f2["circle_source"] == "sam3.1" and f2["sam_prompt"] == "person"
    assert f2["sam_boxes"] == {"head": 0, "person": 1}
    assert f2["circle"]["cx"] == 50 and f2["circle"]["cy"] == 20 and f2["circle"]["r"] == 50
    assert result["api_calls"]["sam"] == 2

    both = _BoxSam({"head": [], "person": [person]})
    shared = VerifyGates(enabled=True, client=VerifyClient(post=_confirm_post(vest=False, head=False))).evaluate(
        "both", ["F2", "F5"], _timed(3, t0=0.0, step=0.5), lambda: (True, "verify_gate"), sam=both)
    assert both.calls.count("person") == 1 and both.calls.count("head") == 1
    assert shared["api_calls"] == {"muse": 1, "sam": 2}
    assert shared["factors"]["F2"]["sam_prompt"] == "person"
    assert shared["factors"]["F2"]["circle"]["cy"] == 20
    f5 = shared["factors"]["F5"]
    assert f5["circle_source"] == "sam3.1" and f5["sam_prompt"] == "person" and f5["sam_status"] == "ok"
    assert f5["sam_boxes"] == {"person": 1}
    assert f5["circle"]["cx"] == 50 and f5["circle"]["cy"] == 50
    assert f5["circle"]["r"] == 49.5
    assert f5["circle"]["source"] == "sam3.1"
    corner = {"cx": 2, "cy": 70, "r": 6, "w": 100, "h": 100, "source": "head", "score": 0.1,
              "norm": {"cx": 0.02, "cy": 0.7, "rx": 0.06, "ry": 0.06, "angle_deg": 0}}
    rec = {"factors": ["F2", "F5"], "top_factor": "F2", "status": "open", "risk_max": 0.8,
           "segments": [{"factor": "F2", "peak_score": 0.8, "threshold": 0.5, "peak_t": 1.0, "start_s": 0.0},
                        {"factor": "F5", "peak_score": 0.4, "threshold": 0.15, "peak_t": 1.2, "start_s": 0.0}],
           "overlays": [
               {"factor": "F2", "t_ms": 0, "score": 0.8, "circle": dict(corner)},
               {"factor": "F2", "t_ms": 1000, "score": 0.7, "circle": dict(corner, cx=3)},
               {"factor": "F5", "t_ms": 0, "score": 0.4, "circle": dict(corner, cy=78)},
               {"factor": "F5", "t_ms": 1000, "score": 0.4, "circle": dict(corner, cy=79)},
           ],
           "localization": {"F2": "weak", "F5": "weak"}, "escalation": {}, "clip_t0_s": 0.0}
    assert apply_verify_result(rec, shared) is True
    assert {o["circle"]["source"] for o in rec["overlays"] if o["factor"] == "F2"} == {"sam3.1"}
    assert {o["circle"]["source"] for o in rec["overlays"] if o["factor"] == "F5"} == {"sam3.1"}
    assert all(o["circle"]["cy"] == 20 for o in rec["overlays"] if o["factor"] == "F2")
    assert all(o["circle"]["cy"] == 50 and o["circle"]["r"] == 49.5 for o in rec["overlays"] if o["factor"] == "F5")
    assert rec["escalation"]["verify_gates"]["factors"]["F5"]["sam_prompt"] == "person"

    missed = _BoxSam({"head": [], "person": []})
    fallback = VerifyGates(enabled=True, client=VerifyClient(post=_confirm_post(head=False))).evaluate(
        "empty", ["F2"], _images(3), lambda: (True, "verify_gate"), sam=missed)
    assert missed.calls == ["head", "person"]
    assert fallback["factors"]["F2"]["circle_source"] == "head_map_fallback"
    assert fallback["factors"]["F2"]["sam_status"] == "no_match"
    assert fallback["factors"]["F2"]["sam_boxes"] == {"head": 0, "person": 0}
    assert fallback["api_calls"]["sam"] == 2
    kept = _f2_record()
    kept["overlays"].append({"factor": "F2", "t_ms": 900, "score": 0.6, "circle": dict(kept["overlays"][0]["circle"])})
    assert apply_verify_result(kept, fallback) is True
    assert [o["circle"]["source"] for o in kept["overlays"]] == ["head_map_fallback", "head_map_fallback"]
    assert kept["overlays"][0]["circle"]["cx"] == 10


def test_sam_circle_only_inside_hold_window(monkeypatch):
    """SAM circles stay within sam_hold_s of the frame SAM ran on. Outside, no circle."""
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    box = {"id": 1, "x1": 40, "y1": 50, "x2": 60, "y2": 70, "w": 100, "h": 80, "c": 0.9}
    sam = _BoxSam({"head": [box]})
    gate = VerifyGates(enabled=True, client=VerifyClient(post=_confirm_post(head=False)))
    result = gate.evaluate("hold", ["F2"], _timed(3, t0=10.0, step=1.0), lambda: (True, "verify_gate"), sam=sam)
    assert result["factors"]["F2"]["sam_t_stream"] == 12.0
    assert result["sam_hold_s"] == 1.5
    head = _head_circle()
    rec = _f2_record()
    rec["overlays"] = [
        {"factor": "F2", "t_stream": 12.0, "t_ms": 0, "score": 0.8, "circle": dict(head)},
        {"factor": "F2", "t_stream": 11.0, "t_ms": 1000, "score": 0.7, "circle": dict(head)},
        {"factor": "F2", "t_stream": 10.4, "t_ms": 1600, "score": 0.6, "circle": dict(head, cx=3)},
        {"factor": "F2", "t_stream": 9.0, "t_ms": 3000, "score": 0.5, "circle": dict(head, cx=9)},
    ]
    assert apply_verify_result(rec, result) is True
    by_t = {o["t_stream"]: o for o in rec["overlays"]}
    assert by_t[12.0]["circle"]["source"] == "sam3.1" and by_t[12.0]["circle"]["cx"] == 50
    assert by_t[11.0]["circle"]["source"] == "sam3.1"
    assert by_t[10.4]["circle"] is None
    assert by_t[9.0]["circle"] is None
    assert by_t[10.4]["factor"] == "F2" and by_t[10.4]["score"] == 0.6

    tight = VerifyGates(enabled=True, sam_hold_s=0.4, client=VerifyClient(post=_confirm_post(head=False)))
    tight_sam = _BoxSam({"head": [box]})
    narrow = tight.evaluate("narrow", ["F2"], _timed(3, t0=10.0, step=1.0), lambda: (True, "verify_gate"), sam=tight_sam)
    assert narrow["sam_hold_s"] == 0.4 and narrow["factors"]["F2"]["sam_t_stream"] == 12.0
    near = _f2_record()
    near["overlays"] = [
        {"factor": "F2", "t_stream": 12.0, "t_ms": 0, "score": 0.8, "circle": dict(head)},
        {"factor": "F2", "t_stream": 11.0, "t_ms": 1000, "score": 0.7, "circle": dict(head)},
    ]
    assert apply_verify_result(near, narrow) is True
    assert near["overlays"][0]["circle"]["source"] == "sam3.1"
    assert near["overlays"][1]["circle"] is None


def test_suppressed_gated_factor_is_not_published():
    result = {"cached": False, "raw": "raw-answer", "answers": [_frame(True, vest=True)],
              "api_calls": {"muse": 1, "sam": 0}, "skipped": [], "calls": [],
              "factors": {"F5": {"status": "rejected", "votes": 1, "n_frames": 3,
                                 "reason": "1 of 3 frames show a person with no hi-vis vest",
                                 "verified_by": None}}}
    rec = {"factors": ["F5"], "top_factor": "F5", "status": "open", "risk_max": 0.4,
           "segments": [{"factor": "F5", "peak_score": 0.4, "threshold": 0.15}],
           "overlays": [{"factor": "F5", "circle": {"source": "head"}}],
           "localization": {"F5": "weak"}, "escalation": {}}
    assert apply_verify_result(rec, result) is False
    assert rec["status"] == "suppressed" and rec["factors"] == []
    assert rec["escalation"]["verify_gates"]["suppressed"] == ["F5"]
    assert "verified_by" not in rec


def test_muted_factor_opens_only_through_the_gate():
    tr = IncidentTracker({"F5": 0.5, "F2": 0.5, "F1": 0.5}, ema_alpha=1.0, min_on_steps=2, muted=("F2", "F5"))
    assert tr.update(0.0, {"F5": 0.9, "F2": 0.8, "F1": 0.0}) == []
    assert tr.update(1.0, {"F5": 0.9, "F2": 0.8, "F1": 0.0}) == []
    opened = tr.note_gate_factor(1.0, "F5", 0.9, 0.15, True)
    assert opened[0][0] == "open"
    joined = tr.note_gate_factor(1.0, "F2", 0.8, 0.5, True)
    assert joined[0][0] == "gate_on" and tr.current.factors == ["F5", "F2"]


class _ScoreModel(FakeModel):
    def __init__(self, factors):
        self._hot = factors

    def step(self, window):
        out = super().step(window)
        bright = float(window[-1].mean()) > 100
        probs = {f: 0.01 for f in self.factors}
        if bright:
            for fid, score in self._hot.items():
                probs[fid] = score
        out["probs"] = probs
        if "F2" in self._hot:
            out["evidence"] = out["evidence"].copy()
            out["evidence"][1, 2, 8] = 1.0
        return out


class _LongCap(Cap):
    """Bright for long enough that some F2 overlays fall outside the 1.5 s SAM hold."""

    def __init__(self):
        super().__init__(n=400)

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        v = 200 if 40 <= self.i < 280 else 20
        return True, np.full((180, 320, 3), v, np.uint8)


def _run(tmp_path, monkeypatch, model, escalation, post, env=None, cap=None):
    _quiet(monkeypatch)
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("requests.post", post)
    cfg = load_config(overrides={
        "storage": {"data_dir": str(tmp_path)},
        "sampling": {"stride_s": 0.5, "sample_interval_s": 0.16},
        "events": {"min_threshold": 0.0, "merge_gap_s": 1.0, "ema_alpha": 1.0},
        "clips": {"pre_s": 1.0, "post_s": 1.0, "width": 320},
        "escalation": escalation,
    })
    pipe = SightlinePipeline(cfg, model, store=EventStore(tmp_path))
    got = []
    opener = cap if cap is not None else Cap
    res = pipe.process(VideoSource("synthetic.mp4", realtime=False, open_fn=lambda u: opener()),
                       camera_id="cam_gate", on_event=got.append, log=lambda m: None)
    return res, got


def _muse(votes, field="wearing_hi_vis_vest"):
    def post(url, json=None, timeout=None, headers=None, **_k):
        post.calls.append(json)
        if not json or "messages" not in json:
            return Http('{"confirmed": false, "severity": 1, "reason": "no", "action": "none"}')
        text = json["messages"][0]["content"][0]["text"]
        if field not in text and "workshop safety reviewer" not in text:
            raise AssertionError(text[:120])
        n = sum(1 for p in json["messages"][0]["content"] if p.get("type") == "image_url")
        if "workshop safety reviewer" in text:
            return Http('{"confirmed": false, "severity": 1, "reason": "no", "action": "none"}')
        return Http(_payload(_answers(n, votes, field)))
    post.calls = []
    return post


def test_gate_off_never_alerts_and_f1_still_publishes(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("a disabled verify gate must not call Muse")

    res, got = _run(tmp_path, monkeypatch, _ScoreModel({"F5": 0.9}),
                    {"verify_gates": {"enabled": False}}, boom)
    assert res["events"] == [] and got == []

    def no_gate_prompt(url, json=None, timeout=None, headers=None, **_k):
        if json and "messages" in json:
            text = json["messages"][0]["content"][0]["text"]
            if "wearing_hi_vis_vest" in text or "wearing_head_protection" in text:
                raise AssertionError("F1 incident must not ask the verify gate")
        return Http('{"confirmed": true, "severity": 3, "reason": "hand", "action": "stop"}')

    res, got = _run(tmp_path / "f1", monkeypatch, FakeModel(), {"verify_gates": {"enabled": True}}, no_gate_prompt)
    assert len(got) == 1 and got[0]["factors"] == ["F1"] and got[0]["status"] == "open"
    assert "verify_gates" not in got[0]["escalation"]
    assert got[0]["peak_t_stream"] == round(got[0]["segments"][0]["peak_t"], 2)
    f1_thumb = tmp_path / "f1" / "events" / got[0]["event_id"] / "thumb.jpg"
    assert f1_thumb.exists() and f1_thumb.stat().st_size > 0


def test_pipeline_confirms_f5_and_suppresses_a_rejected_gate(tmp_path, monkeypatch):
    post = _muse(3)
    res, got = _run(tmp_path, monkeypatch, _ScoreModel({"F5": 0.9}),
                    {"verify_gates": {"enabled": True, "max_tokens": 1750,
                                      "factors": {"F5": {"trigger_threshold": 0.15}}}}, post)
    assert len(got) == 1 and got[0]["factors"] == ["F5"] and got[0]["status"] == "open"
    assert got[0]["verified_by"] == "Muse Spark"
    gate = got[0]["escalation"]["verify_gates"]
    assert gate["factors"]["F5"]["status"] == "confirmed" and gate["factors"]["F5"]["votes"] >= 2
    assert gate["api_calls"]["muse"] == 1 and gate["suppressed"] == []
    assert got[0]["escalation"]["calls"][0]["reason"] == "verify_gate"
    muse = [c for c in post.calls if isinstance(c, dict) and "messages" in c]
    assert len(muse) == 1 and muse[0]["max_tokens"] == 1750
    assert got[0]["escalation"]["budget"]["session_used"] == 1

    post_no = _muse(1)
    res, got = _run(tmp_path / "no", monkeypatch, _ScoreModel({"F5": 0.9}),
                    {"verify_gates": {"enabled": True}}, post_no)
    assert got == [] and len(res["events"]) == 1
    rec = res["events"][0]
    assert rec["status"] == "suppressed" and rec["factors"] == []
    assert rec["escalation"]["verify_gates"]["factors"]["F5"]["status"] == "rejected"
    assert rec["escalation"]["verify_gates"]["suppressed"] == ["F5"]


def test_pipeline_incident_cap_and_session_env(tmp_path, monkeypatch):
    def post(url, json=None, timeout=None, headers=None, **_k):
        post.n += 1
        if json and "messages" in json:
            text = json["messages"][0]["content"][0]["text"]
            if "wearing_hi_vis_vest" in text or "wearing_head_protection" in text:
                raise AssertionError("verify gate must not run after the incident cap")
            return Http('{"confirmed": false, "severity": 1, "reason": "no", "action": "none"}')
        return Http("")
    post.n = 0
    res, got = _run(tmp_path, monkeypatch, _ScoreModel({"F1": 0.95, "F5": 0.9}),
                    {"verify_gates": {"enabled": True}}, post)
    assert len(got) == 1 and got[0]["factors"] == ["F1"]
    assert got[0]["escalation"]["verify_gates"]["factors"]["F5"]["status"] == "skipped_budget"
    assert got[0]["escalation"]["verify_gates"]["factors"]["F5"]["reason"] == "incident_cap"
    assert any(s.get("status") == "skipped_budget" and s.get("reason") == "incident_cap" and s.get("factor") == "F5"
               for s in got[0]["escalation"]["skipped"])

    def boom(*_a, **_k):
        raise AssertionError("session cap 0 must not call Muse")

    res, got = _run(tmp_path / "env", monkeypatch, _ScoreModel({"F5": 0.9}),
                    {"verify_gates": {"enabled": True}}, boom,
                    env={"SIGHTLINE_ESC_MAX_PER_SESSION": "0"})
    assert got == [] and res["events"][0]["status"] == "suppressed"
    skipped = res["events"][0]["escalation"]["skipped"]
    assert skipped and skipped[0]["reason"] == "session_cap" and skipped[0]["status"] == "skipped_budget"
    assert res["events"][0]["escalation"]["verify_gates"]["factors"]["F5"]["status"] == "skipped_budget"


def test_pipeline_f2_sam_circle(tmp_path, monkeypatch):
    box = {"id": 0, "x1": 10, "y1": 4, "x2": 30, "y2": 20, "w": 64, "h": 48, "c": 0.8}

    def segment(self, img, prompt):
        segment.calls.append(prompt)
        if prompt == "head":
            return {"status": "ok", "boxes": [dict(box, w=img.shape[1], h=img.shape[0])], "prompt": prompt, "sent": True}
        return {"status": "no_match", "boxes": [], "prompt": prompt, "sent": True}
    segment.calls = []
    monkeypatch.setattr(SamRefiner, "segment", segment)
    post = _muse(3, "wearing_head_protection")
    res, got = _run(tmp_path, monkeypatch, _ScoreModel({"F2": 0.8}),
                    {"verify_gates": {"enabled": True, "factors": {"F2": {"trigger_threshold": 0.5, "sam_circle": True}}}},
                    post, cap=_LongCap)
    assert len(got) == 1 and got[0]["factors"] == ["F2"]
    assert got[0]["escalation"]["verify_gates"]["factors"]["F2"]["circle_source"] == "sam3.1"
    assert got[0]["escalation"]["verify_gates"]["api_calls"]["sam"] == 1
    assert segment.calls == ["head"]
    f2 = got[0]["escalation"]["verify_gates"]["factors"]["F2"]
    sam_t = f2["sam_t_stream"]
    assert got[0]["peak_t_stream"] == sam_t
    thumb = tmp_path / "events" / got[0]["event_id"] / "thumb.jpg"
    assert thumb.exists() and thumb.stat().st_size > 0
    f2_ovs = [o for o in got[0]["overlays"] if o["factor"] == "F2"]
    inside = [o for o in f2_ovs if abs(o["t_stream"] - sam_t) <= 1.5 + 1e-6]
    outside = [o for o in f2_ovs if abs(o["t_stream"] - sam_t) > 1.5 + 1e-6]
    assert inside and outside
    assert {o["circle"]["source"] for o in inside} == {"sam3.1"}
    assert len({(o["circle"]["norm"]["cx"], o["circle"]["norm"]["cy"]) for o in inside}) == 1
    assert all(o["circle"] is None for o in outside)

    def empty(self, img, prompt):
        empty.calls.append(prompt)
        return {"status": "no_match", "boxes": [], "prompt": prompt, "sent": True}
    empty.calls = []
    monkeypatch.setattr(SamRefiner, "segment", empty)
    res, got = _run(tmp_path / "fb", monkeypatch, _ScoreModel({"F2": 0.8}),
                    {"verify_gates": {"enabled": True}}, _muse(3, "wearing_head_protection"))
    assert got[0]["escalation"]["verify_gates"]["factors"]["F2"]["circle_source"] == "head_map_fallback"
    assert got[0]["escalation"]["verify_gates"]["api_calls"]["sam"] == 2
    assert empty.calls == ["head", "person"]
    sources = [o["circle"]["source"] for o in got[0]["overlays"] if o["factor"] == "F2"]
    assert sources and set(sources) == {"head_map_fallback"}
    assert got[0]["escalation"]["verify_gates"]["factors"]["F2"]["sam_t_stream"] is not None
    assert got[0]["peak_t_stream"] == round(got[0]["segments"][0]["peak_t"], 2)
    assert got[0]["peak_t_stream"] != got[0]["escalation"]["verify_gates"]["factors"]["F2"]["sam_t_stream"]


def test_parse_fences_indexes_yes_no_and_reasoning(monkeypatch):
    inner = '{"frames":[{"person_visible":true,"wearing_hi_vis_vest":false}]}'
    assert parse_answers(f"```json\n{inner}\n```", n=1) == [
        {"person_visible": True, "wearing_hi_vis_vest": False}]
    text = json.dumps({
        "image_10": {"person_visible": False, "wearing_hi_vis_vest": True},
        "image_2": {"person_visible": True, "wearing_hi_vis_vest": True},
        "image_1": {"person_visible": True, "wearing_hi_vis_vest": False},
        "usage": {"tokens": 3},
    })
    assert parse_answers(text, n=3)[0]["wearing_hi_vis_vest"] is False
    yes = ('{"frames":[{"person_visible":"yes","wearing_hi_vis_vest":"no"},'
           '{"person_visible":"TRUE","wearing_hi_vis_vest":"False"}]}')
    assert parse_answers(yes, n=2) == [
        {"person_visible": True, "wearing_hi_vis_vest": False},
        {"person_visible": True, "wearing_hi_vis_vest": False},
    ]
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    answer = '{"frames":[{"person_visible":"yes","wearing_head_protection":"no"}]}'

    def post(url, json=None, timeout=None, headers=None, **_k):
        return _message({"content": "", "reasoning_content": answer})

    out = VerifyClient(post=post).ask(_images(1), ("F2",))
    assert out["status"] == "ok" and out["answers"][0]["wearing_head_protection"] is False


def _message(message, status=200):
    class R:
        status_code = status
        headers = {"content-type": "application/json"}
        text = json.dumps({"choices": [{"message": message}]})

        def json(self):
            return json.loads(self.text)
    return R()


def test_muse_failure_stores_diagnostic_without_retry(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    reasoning = ("R" * 1800) + " not json"
    calls = {"n": 0}

    def post(url, json=None, timeout=None, headers=None, **_k):
        calls["n"] += 1
        assert json["max_tokens"] == 4000
        assert headers["Authorization"] == "Bearer test-key"
        body = {
            "usage": {"prompt_tokens": 120, "completion_tokens": 4000,
                      "completion_tokens_details": {"reasoning_tokens": 4000, "note": "ignore-me"}},
            "choices": [{
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "", "reasoning_content": reasoning},
            }],
        }

        class Response:
            status_code = 200
            headers = {"Authorization": "Bearer test-key", "content-type": "application/json"}
            text = __import__("json").dumps(body)

            def json(self):
                return body

        return Response()

    result = VerifyGates(enabled=True, client=VerifyClient(post=post)).evaluate(
        "bad", ["F5", "F2"], _images(3), lambda: (True, "verify_gate"))
    assert calls["n"] == 1
    assert result["factors"]["F2"]["status"] == "failed"
    assert "expected JSON" in result["factors"]["F5"]["reason"]
    diag = result["diagnostic"]
    assert diag["http_status"] == 200 and diag["finish_reason"] == "length"
    assert diag["content"] == ""
    assert diag["reasoning"] == reasoning[:1500] and len(diag["reasoning"]) == 1500
    assert diag["usage"]["completion_tokens"] == 4000
    assert diag["usage"]["completion_tokens_details"] == {"reasoning_tokens": 4000}
    blob = json.dumps(diag)
    assert "Bearer" not in blob and "test-key" not in blob and "Authorization" not in blob
    rec = {"factors": ["F5", "F2"], "top_factor": "F5", "status": "open", "risk_max": 0.4,
           "segments": [{"factor": "F5", "peak_score": 0.4, "threshold": 0.15},
                        {"factor": "F2", "peak_score": 0.8, "threshold": 0.5}],
           "overlays": [], "localization": {}, "escalation": {}}
    assert apply_verify_result(rec, result) is False
    stored = rec["escalation"]["verify_gates"]["diagnostic"]
    assert stored["content"] == "" and stored["finish_reason"] == "length"
    assert stored["reasoning"].startswith("R") and "not json" not in stored["reasoning"]
    call = result["calls"][0]
    assert call["diagnostic"]["content"] == "" and call["diagnostic"]["usage"]["prompt_tokens"] == 120
    assert call["muse_spark"]["diagnostic"]["reasoning"] == reasoning[:1500]
    assert "headers" not in call["diagnostic"] and "headers" not in call["muse_spark"]


def test_cli_dry_run_show_raw_and_factor(tmp_path, monkeypatch, capsys):
    import cv2
    root = tmp_path / "frames"
    root.mkdir()
    cv2.imwrite(str(root / "a.jpg"), np.zeros((8, 8, 3), np.uint8))
    cv2.imwrite(str(root / "b.jpg"), np.full((8, 8, 3), 255, np.uint8))
    labels = tmp_path / "labels.csv"
    labels.write_text("file,vest\na.jpg,0\nb.jpg,1\n", encoding="utf-8")

    def boom(*_a, **_k):
        raise AssertionError("dry-run must not call the API")

    monkeypatch.setattr("requests.post", boom)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert eval_main(["--frames", str(root), "--labels", str(labels), "--dry-run", "--factor", "F5"]) == 0
    dry = capsys.readouterr().out
    assert "api_calls: 0" in dry and "would_call: 1" in dry and "dry_run: true" in dry
    assert "factor: F5" in dry

    monkeypatch.setenv("MODEL_API_KEY", "test-key")

    def post(url, json=None, timeout=None, headers=None, **_k):
        post.n += 1
        assert json["max_tokens"] == 640
        text = json["messages"][0]["content"][0]["text"]
        assert "wearing_hi_vis_vest" in text and "wearing_head_protection" not in text
        n = sum(1 for p in json["messages"][0]["content"] if p.get("type") == "image_url")
        vest = post.n == 2
        return Http(_payload(_answers(n, 0 if vest else 1)))
    post.n = 0
    monkeypatch.setattr("requests.post", post)
    assert eval_main(["--frames", str(root), "--labels", str(labels), "--batch-size", "1",
                      "--max-tokens", "640", "--show-raw"]) == 0
    out = capsys.readouterr().out
    assert "api_calls: 2" in out and "accuracy: 1.000" in out
    assert "file: a.jpg label: 0 person_visible: true wearing_hi_vis_vest: false correct: 1" in out
    assert "status: ok" in out
    assert "eval-show-raw-secret" not in out
