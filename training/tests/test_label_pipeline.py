"""Offline end-to-end test of label_glimmer.py with a fake model client (no Ollama)."""
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
yaml = pytest.importorskip("yaml")

import label_glimmer as lg  # noqa: E402
from glimmer_schema import GLIMMER_FACTORS, LabelValidationError  # noqa: E402


def good_reply(present=()):
    return json.dumps({"factors": {f: {"state": "present" if f in present else "absent",
                                       "confidence": 0.9, "evidence": ""} for f in GLIMMER_FACTORS}})


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def check(self):
        pass

    def chat(self, messages):
        self.calls += 1
        return self.replies.pop(0) if self.replies else good_reply(present=("F8",))


def test_label_frame_retries_on_bad_json():
    c = FakeClient(["not json", '{"factors": {"F1": 1}}', good_reply()])
    parsed, raw, attempts = lg.label_frame(c, "sys", "user", b"jpg", retries=2)
    assert attempts == 3 and c.calls == 3 and parsed["factors"]["F1"]["state"] == "absent"


def test_label_frame_gives_up():
    c = FakeClient(["bad"] * 5)
    with pytest.raises(LabelValidationError, match="gave up after 2"):
        lg.label_frame(c, "sys", "user", b"jpg", retries=1)


def test_prompt_renders_all_factors():
    system, user, pid = lg.load_prompt(lg.DEFAULT_PROMPT)
    assert pid.startswith("v1-")
    for f in GLIMMER_FACTORS:
        assert f"- {f} (" in system
    assert "F11" not in system.split("Hazard factors:")[1]
    assert "{factor_block}" not in system


def test_keyframe_indices():
    assert lg.keyframe_indices(100, 4) == [12, 37, 62, 87]
    assert lg.keyframe_indices(2, 4) == [0, 1]
    assert lg.keyframe_indices(0, 4) == []


def make_video(path, n=24):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i * 10 % 255, dtype=np.uint8))
    w.release()


def test_end_to_end_with_cache(tmp_path, monkeypatch):
    vid = tmp_path / "clip.mp4"
    make_video(vid)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "test/clip", "source": "unit", "kind": "video",
                                    "path": str(vid), "split": "train", "sha256": "x",
                                    "factor_hints": {"F8": "present"}}) + "\n", encoding="utf-8")
    fake = FakeClient(["oops"])  # first frame needs one retry
    monkeypatch.setattr(lg, "GlimmerClient", lambda *a, **k: fake)
    out = tmp_path / "out.jsonl"
    args = ["--manifest", str(manifest), "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
            "--keyframes", "3"]
    assert lg.main(args) == 0
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["factors"]["F8"]["label"] == "present"
    assert rec["hint_agreement"]["F8"]["agree"] is True
    assert len(rec["keyframes"]) == 3
    assert fake.calls == 4
    # rerun: everything comes from the cache, no new calls
    assert lg.main(args) == 0
    assert fake.calls == 4


def test_client_sends_reasoning_effort_and_json_mode(monkeypatch):
    sent = {}
    c = lg.GlimmerClient("http://x/v1", "m", reasoning_effort="none")

    def fake_post(path, payload):
        sent.update(payload)
        return {"choices": [{"message": {"content": good_reply()}, "finish_reason": "stop"}], "usage": {"completion_tokens": 9}}
    monkeypatch.setattr(c, "_post", fake_post)
    assert json.loads(c.chat([]))["factors"]["F1"]["state"] == "absent"
    assert sent["reasoning_effort"] == "none" and sent["response_format"] == {"type": "json_object"}
    assert sent["max_tokens"] == 2048 and c.last_finish_reason == "stop"
    c2 = lg.GlimmerClient("http://x/v1", "m", json_mode=False, reasoning_effort="default")
    sent.clear()
    monkeypatch.setattr(c2, "_post", fake_post)
    c2.chat([])
    assert "reasoning_effort" not in sent and "response_format" not in sent


def test_truncated_reply_error_mentions_max_tokens():
    class Cut(FakeClient):
        def chat(self, messages):
            self.last_finish_reason = "length"
            return '{"factors": {"F1": {"state": "absent", "conf'
    with pytest.raises(LabelValidationError, match="max_tokens"):
        lg.label_frame(Cut([]), "s", "u", b"j", retries=0)


class SlowFake(FakeClient):
    """Thread-safe fake that records how many requests overlap."""
    def __init__(self):
        super().__init__([])
        import threading
        self.lock = threading.Lock()
        self.inflight = self.peak = 0

    def chat(self, messages):
        import time
        with self.lock:
            self.calls += 1
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        time.sleep(0.05)
        with self.lock:
            self.inflight -= 1
        return good_reply(present=("F8",))


def test_concurrency_matches_serial_and_overlaps(tmp_path, monkeypatch):
    lines = []
    for i in range(6):
        vid = tmp_path / f"c{i}.mp4"
        make_video(vid)
        lines.append(json.dumps({"id": f"t/c{i}", "source": "unit", "kind": "video", "path": str(vid),
                                 "split": "train", "sha256": f"h{i}", "factor_hints": {}}))
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outs = {}
    for conc in (1, 4):
        fake = SlowFake()
        monkeypatch.setattr(lg, "GlimmerClient", lambda *a, **k: fake)
        out = tmp_path / f"out{conc}.jsonl"
        assert lg.main(["--manifest", str(manifest), "--cache-dir", str(tmp_path / f"cache{conc}"),
                        "--out", str(out), "--keyframes", "2", "--concurrency", str(conc)]) == 0
        assert fake.calls == 12 and lg.LAST_RUN["calls"] == 12 and lg.LAST_RUN["concurrency"] == conc
        assert (fake.peak > 1) == (conc > 1)
        outs[conc] = [{k: r[k] for k in ("clip_id", "factors", "keyframes")}
                      for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())]
    assert outs[1] == outs[4] and len(outs[4]) == 6


def test_backend_defaults_and_sglang_payload(monkeypatch):
    assert lg.BACKENDS["sglang"][0].endswith(":30000/v1")
    sent = {}
    c = lg.GlimmerClient("http://x/v1", "m", reasoning_effort="none", backend="sglang")

    def fake_post(path, payload):
        sent.update(payload)
        return {"choices": [{"message": {"content": good_reply()}, "finish_reason": "stop"}], "usage": {}}
    monkeypatch.setattr(c, "_post", fake_post)
    c.chat([])
    assert sent["chat_template_kwargs"] == {"reasoning_strength": "none"} and "reasoning_effort" not in sent
    assert c.last_finish_reason == "stop"


def test_sglang_cache_key_differs_from_ollama(tmp_path):
    a = lg.cache_path(tmp_path, "v2-x", lg.BACKENDS["ollama"][1], "voxel51/0_tr1", 5, 768, "none")
    b = lg.cache_path(tmp_path, "v2-x", lg.BACKENDS["sglang"][1], "voxel51/0_tr1", 5, 768, "none")
    assert a != b
