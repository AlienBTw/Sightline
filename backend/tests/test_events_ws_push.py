import time

from .conftest import auth, login, make_video, seed_file_camera, wait_for


def _start_and_get_event(client, t, tmp_path):
    v = make_video(tmp_path / "v.mp4")
    cam = seed_file_camera(client, v, tokens=t)
    ev = wait_for(lambda: client.get("/api/events", headers=auth(t)).json()["items"])
    return cam, ev[0]


def test_events_list_detail_media_range_and_status(app_env, tmp_path, pushed):
    client, s, factory, _ = app_env
    t = login(client)
    cam, e = _start_and_get_event(client, t, tmp_path)
    assert e["camera_id"] == cam["id"] and e["factors"] == ["F1"] and e["top_factor_label"] == "Hand in danger zone"
    eid = e["event_id"]
    time.sleep(2.1)                                          # clips "settle" before they are served
    d = client.get(f"/api/events/{eid}", headers=auth(t)).json()
    assert "source" not in d and d["overlays"] and d["media"]["clip"]["ready"]
    clip_url = d["media"]["clip"]["url"]
    # full file
    r = client.get(clip_url)
    assert r.status_code == 200 and len(r.content) == 102400 and r.headers["accept-ranges"] == "bytes"
    # range request (what Android ExoPlayer sends)
    r = client.get(clip_url, headers={"Range": "bytes=100-199"})
    assert r.status_code == 206 and r.headers["content-range"] == "bytes 100-199/102400"
    assert r.content == (bytes(range(256)) * 400)[100:200]
    r = client.get(clip_url, headers={"Range": "bytes=-16"})
    assert r.status_code == 206 and len(r.content) == 16
    r = client.get(clip_url, headers={"Range": "bytes=999999-"})
    assert r.status_code == 416 and r.headers["content-range"] == "bytes */102400"
    r = client.head(clip_url)
    assert r.status_code == 200 and r.headers["content-length"] == "102400" and r.content == b""
    etag = client.get(clip_url).headers["etag"]
    assert client.get(clip_url, headers={"If-None-Match": etag}).status_code == 304
    # bearer works too, unsigned does not; path traversal is rejected by the id pattern
    assert client.get(f"/api/events/{eid}/clip_circles.mp4", headers=auth(t)).status_code == 200
    assert client.get(f"/api/events/{eid}/clip.mp4").status_code == 401
    assert client.get(f"/api/events/{eid}/event.json", headers=auth(t)).status_code in (404, 422)
    assert client.get("/api/events/..%2F..%2Fsecret/clip.mp4", headers=auth(t)).status_code in (404, 422)
    assert client.get(d["thumb_url"]).headers["content-type"] == "image/jpeg"
    # filters + pagination
    assert client.get("/api/events?factor=F1", headers=auth(t)).json()["total"] == 1
    assert client.get("/api/events?factor=F9", headers=auth(t)).json()["total"] == 0
    assert client.get(f"/api/events?camera_id={cam['id']}&min_risk=0.9", headers=auth(t)).json()["total"] == 1
    assert client.get("/api/events?status=dismissed", headers=auth(t)).json()["total"] == 0
    assert client.get("/api/events?status=bogus", headers=auth(t)).status_code == 422
    p = client.get("/api/events?limit=1&offset=0", headers=auth(t)).json()
    assert p["limit"] == 1 and p["next_offset"] is None
    # acknowledge / dismiss
    assert client.post(f"/api/events/{eid}/ack", headers=auth(t)).json()["status"] == "acknowledged"
    assert client.get("/api/events?status=acknowledged", headers=auth(t)).json()["total"] == 1
    assert client.post(f"/api/events/{eid}/dismiss", headers=auth(t)).json()["status"] == "dismissed"
    d = client.get(f"/api/events/{eid}", headers=auth(t)).json()
    assert d["status"] == "dismissed" and [a["status"] for a in d["actions"]] == ["acknowledged", "dismissed"]
    assert client.post(f"/api/events/{eid}/status", json={"status": "deleted"}, headers=auth(t)).status_code == 422
    assert client.post("/api/events/evt_nope/ack", headers=auth(t)).status_code == 404


def test_websocket_gets_events_and_push_is_sent(app_env, tmp_path, pushed):
    client, s, factory, _ = app_env
    t = login(client)
    assert client.post("/api/devices", json={"token": "ExponentPushToken[abcdefghijklmnop]"}, headers=auth(t)).status_code == 201
    assert client.post("/api/devices", json={"token": "not-a-token-123"}, headers=auth(t)).status_code == 422
    tk = client.post("/api/ws/ticket", headers=auth(t)).json()
    with client.websocket_connect(tk["url"]) as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["user"] == "demo"
        v = make_video(tmp_path / "v.mp4")
        seed_file_camera(client, v, tokens=t)
        got = None
        for _ in range(50):
            m = ws.receive_json()
            if m["type"] == "event":
                got = m["event"]
                break
        assert got and got["factors"] == ["F1"] and "source" not in got
    wait_for(lambda: pushed)
    msg = pushed[0]
    assert msg["to"] == "ExponentPushToken[abcdefghijklmnop]"
    assert set(msg["data"]) == {"event_id", "factor", "risk"}          # no images / camera names in the payload
    assert msg["data"]["event_id"] == got["event_id"] and msg["channelId"] == "alerts"


def test_websocket_ticket_required_and_single_use(client):
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/ws?ticket=bad") as ws:
            ws.receive_json()
    t = login(client)
    tk = client.post("/api/ws/ticket", headers=auth(t)).json()
    with client.websocket_connect(tk["url"]) as ws:
        assert ws.receive_json()["type"] == "hello"
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(tk["url"]) as ws:
            ws.receive_json()


def test_push_device_not_registered_is_pruned(tmp_path):
    import httpx
    from backend.db import DB
    from backend.push import ExpoPush
    db = DB(tmp_path / "b.db")
    db.add_device("ExponentPushToken[aaaaaaaaaaaaaaaa]", "demo", "android")
    tr = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [
        {"status": "error", "message": "gone", "details": {"error": "DeviceNotRegistered"}}]}))
    p = ExpoPush(db, "https://exp.host/--/api/v2/push/send", transport=tr, cooldown_s=60)
    rec = {"event_id": "evt_a", "camera_id": "c", "top_factor": "F1", "risk_max": 0.5}
    p.notify(rec).result(timeout=5)
    assert db.devices() == [] and p.stats["errors"] == 1
    assert p.notify(rec) is None and p.stats["suppressed"] == 1          # cooldown per camera+factor


def test_health_detail_authed(client):
    t = login(client)
    r = client.get("/api/health", headers=auth(t))
    assert r.status_code == 200 and r.json()["db"] is True and r.json()["capacity"]["max"] == 2
