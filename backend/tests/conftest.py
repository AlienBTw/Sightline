"""Backend API tests with a fake pipeline (no GPU, no torch, no network)."""
import json
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.security import hash_password
from backend.settings import Settings
from sightline_pipeline.storage import EventStore

PASSWORDS = {"demo": "demo-pass-123", "admin": "admin-pass-456"}
_HASHES = {u: hash_password(p, n=2 ** 12) for u, p in PASSWORDS.items()}   # cheap hashes for tests


def make_video(path: Path, seconds: float = 3.0, fps: int = 10, size=(160, 96)):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(int(seconds * fps)):
        img = np.full((size[1], size[0], 3), (i * 7) % 255, np.uint8)
        w.write(img)
    w.release()
    return path


class FakePipeline:
    """Reads the real source (so the live preview has frames), reports one step with a circle and
    writes one event with real files after `event_after` frames."""

    def __init__(self, store: EventStore, event_after: int = 5):
        self.store, self.event_after = store, event_after

    def process(self, source, camera_id="cam", on_event=None, stop=None, max_seconds=None, log=print, on_step=None):
        n, emitted = 0, False
        circle = {"factor": "F1", "score": 0.9, "raw": 0.9,
                  "circle": {"cx": 80, "cy": 40, "r": 10, "norm": {"cx": 0.5, "cy": 0.4, "rx": 0.1, "ry": 0.15, "angle_deg": 0}}}
        for fr in source:
            if stop is not None and stop.is_set():
                break
            n += 1
            if on_step and n % 3 == 0:
                on_step({"camera_id": camera_id, "t": fr.t, "wall": fr.wall, "probs": {"F1": 0.9}, "ema": {"F1": 0.9},
                         "active": [circle], "incident": None})
            if n == self.event_after and not emitted:
                emitted = True
                eid = self.store.next_id(camera_id, time.strftime("%Y%m%dT%H%M%S"))
                d = self.store.event_dir(eid)
                (d / "clip.mp4").write_bytes(bytes(range(256)) * 400)          # 102400 bytes
                (d / "clip_circles.mp4").write_bytes(b"\x00" * 5000)
                (d / "thumb.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
                rec = {"event_id": eid, "camera_id": camera_id, "started_at": "2026-11-01T10:00:00+01:00",
                       "ended_at": "2026-11-01T10:00:10+01:00", "risk_max": 0.91, "top_factor": "F1",
                       "factors": ["F1"], "status": "open", "source": "/secret/path.mp4", "overlays": [circle],
                       "escalation": {"calls": []}}
                self.store.save(rec)
                if on_event:
                    on_event(rec)
            if max_seconds and fr.t >= max_seconds:
                break
        source.stop()
        return {"summary": {"frames": n, "stream_seconds": n / 10, "error": None}, "events": []}


class FakeFactory:
    stride_s = 1.0

    def __init__(self, root: Path):
        self.store = EventStore(root / "live")
        self.loaded = 0

    def load_model(self, device):
        self.loaded += 1
        return type("M", (), {"history": []})()

    def make_pipeline(self, model):
        return FakePipeline(self.store)


@pytest.fixture
def pushed():
    return []


@pytest.fixture
def app_env(tmp_path, pushed):
    def handler(req: httpx.Request):
        msgs = json.loads(req.content)
        pushed.extend(msgs)
        return httpx.Response(200, json={"data": [{"status": "ok", "id": f"t{i}"} for i, _ in enumerate(msgs)]})

    s = Settings(jwt_secret="t" * 48,
                 users={u: {"hash": _HASHES[u], "role": u} for u in PASSWORDS},
                 backend_dir=tmp_path / "backend", uploads_dir=tmp_path / "uploads", gpus=["cuda:0"],
                 cameras_per_gpu=2, seed_demo_camera=False, autostart=False, upload_max_mb=2,
                 upload_min_seconds=1.0, upload_max_seconds=30, push_cooldown_s=0.0, uploads_per_hour=50,
                 mjpeg_max_seconds=1)
    factory = FakeFactory(tmp_path)
    app = create_app(s, factory=factory, push_transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        yield client, s, factory, tmp_path
    

@pytest.fixture
def client(app_env):
    return app_env[0]


def login(client, user="demo"):
    r = client.post("/api/auth/login", json={"username": user, "password": PASSWORDS[user]})
    assert r.status_code == 200, r.text
    return r.json()


def auth(tokens):
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def seed_file_camera(client, path: Path, name="Sample video", start=True, tokens=None) -> dict:
    """Insert a virtual camera that points at a file on disk, then start it through the API."""
    import secrets
    ctx = client.app.state.ctx
    cid = "cam_" + secrets.token_hex(3)
    ctx.db.add_camera({"id": cid, "name": name, "kind": "virtual", "uri": str(path),
                       "loop": True, "enabled": False, "created_by": "test",
                       "meta": {"video": {"duration_s": 3.0, "fps": 10.0}}})
    headers = auth(tokens)
    if not start:
        body = client.get(f"/api/cameras/{cid}", headers=headers).json()
        body["started"], body["start_error"] = False, None
        return body
    r = client.post(f"/api/cameras/{cid}/start", headers=headers)
    if r.status_code == 409:
        body = client.get(f"/api/cameras/{cid}", headers=headers).json()
        body["started"], body["start_error"] = False, r.json().get("detail")
        return body
    assert r.status_code == 200, r.text
    body = r.json()
    body["started"], body["start_error"] = True, None
    return body


def wait_for(fn, timeout=10.0, every=0.05):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(every)
    raise AssertionError("timed out")
