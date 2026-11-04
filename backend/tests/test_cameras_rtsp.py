"""Backend handling of real (RTSP) cameras: transport option, live connection status, stopping a dead camera."""
import socket

from .conftest import auth, login, wait_for


def _closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_rtsp_transport_option(client):
    h = auth(login(client, "admin"))
    r = client.post("/api/cameras", json={"name": "NVR ch1", "uri": "rtsp://admin:pw@10.0.0.8:554/Streams/101",
                                          "start": False, "rtsp_transport": "udp"}, headers=h)
    assert r.status_code == 201 and r.json()["rtsp_transport"] == "udp"
    r = client.post("/api/cameras", json={"name": "x", "uri": "rtsp://10.0.0.8/x", "start": False,
                                          "rtsp_transport": "quic"}, headers=h)
    assert r.status_code == 422


def test_dead_rtsp_camera_reports_reason_and_stops(client, app_env):
    _, _, factory, _ = app_env
    h = auth(login(client, "admin"))
    port = _closed_port()
    r = client.post("/api/cameras", json={"name": "Dead cam", "uri": f"rtsp://admin:secret@127.0.0.1:{port}/Streams/101"},
                    headers=h)
    assert r.status_code == 201, r.text
    cid = r.json()["id"]
    src = wait_for(lambda: (client.get(f"/api/cameras/{cid}", headers=h).json()["status"].get("source") or {})
                   .get("last_error"), timeout=15)
    assert "refused" in src and "secret" not in src
    assert client.post(f"/api/cameras/{cid}/stop", headers=h).status_code == 200
    mgr = client.app.state.ctx.manager
    wait_for(lambda: mgr.worker(cid) is None, timeout=5)        # used to hang forever in queue.get()
    assert client.get(f"/api/cameras/{cid}", headers=h).json()["status"]["state"] == "stopped"
