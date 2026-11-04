from backend.media import parse_range
from .conftest import auth, login, make_video, seed_file_camera, wait_for


def test_file_camera_runs(app_env, tmp_path):
    client, s, factory, _ = app_env
    t = login(client)
    v = make_video(tmp_path / "v.mp4")
    cam = seed_file_camera(client, v, name="Bench saw", tokens=t)
    assert cam["kind"] == "virtual" and cam["loop"] and cam["started"] and cam["name"] == "Bench saw"
    assert "source" not in cam                                   # file path never leaves the server
    wait_for(lambda: client.get(f"/api/cameras/{cam['id']}", headers=auth(t)).json()["status"]["state"] == "running")
    # live preview: snapshot with Bearer, then the signed URL without it
    jpg = wait_for(lambda: (lambda r: r if r.status_code == 200 else None)(
        client.get(f"/api/cameras/{cam['id']}/snapshot.jpg?w=320", headers=auth(t))))
    assert jpg.headers["content-type"] == "image/jpeg" and jpg.content[:2] == b"\xff\xd8"
    signed = cam["preview"]["snapshot_url"]
    assert client.get(signed).status_code == 200
    assert client.get(f"/api/cameras/{cam['id']}/snapshot.jpg").status_code == 401
    # a signed link for one camera does not open another path
    other = signed.replace(f"/cameras/{cam['id']}/", "/cameras/cam_zzz/")
    assert client.get(other).status_code in (403, 404)
    r = client.post(f"/api/cameras/{cam['id']}/stop", headers=auth(t))
    assert r.status_code == 200
    wait_for(lambda: client.get(f"/api/cameras/{cam['id']}", headers=auth(t)).json()["status"]["state"] == "stopped")


def test_capacity_cap(app_env, tmp_path):
    client, s, _, _ = app_env            # 1 GPU x 2 cameras
    t = login(client)
    v = make_video(tmp_path / "v.mp4", seconds=4)
    cams = [seed_file_camera(client, v, name=f"Sample {i}", tokens=t) for i in range(3)]
    assert [c["started"] for c in cams] == [True, True, False]
    assert "slots" in cams[2]["start_error"]
    r = client.post(f"/api/cameras/{cams[2]['id']}/start", headers=auth(t))
    assert r.status_code == 409
    lst = client.get("/api/cameras", headers=auth(t)).json()
    assert lst["capacity"]["max"] == 2


def test_admin_rtsp_crud_redacts_credentials(app_env):
    client, s, _, _ = app_env
    ta, td = login(client, "admin"), login(client, "demo")
    r = client.post("/api/cameras", json={"name": "Phone", "uri": "rtsp://user:pw@192.168.1.20:8554/live", "start": False},
                    headers=auth(ta))
    assert r.status_code == 201, r.text
    cam = r.json()
    assert cam["kind"] == "rtsp" and cam["source"] == "rtsp://***@192.168.1.20:8554/live"
    demo_view = client.get(f"/api/cameras/{cam['id']}", headers=auth(td)).json()
    assert "source" not in demo_view
    assert client.post(f"/api/cameras/{cam['id']}/start", headers=auth(td)).status_code == 403
    assert client.post("/api/cameras", json={"name": "bad", "uri": "/etc/passwd"}, headers=auth(ta)).status_code == 422
    assert client.patch(f"/api/cameras/{cam['id']}", json={"name": "Phone 2"}, headers=auth(ta)).json()["name"] == "Phone 2"
    assert client.delete(f"/api/cameras/{cam['id']}", headers=auth(td)).status_code == 403
    assert client.delete(f"/api/cameras/{cam['id']}", headers=auth(ta)).status_code == 200
    assert client.get(f"/api/cameras/{cam['id']}", headers=auth(ta)).status_code == 404


def test_parse_range():
    assert parse_range(None, 100) is None
    assert parse_range("bytes=0-9", 100) == (0, 9)
    assert parse_range("bytes=90-", 100) == (90, 99)
    assert parse_range("bytes=-10", 100) == (90, 99)
    assert parse_range("bytes=50-500", 100) == (50, 99)
    assert parse_range("bytes=100-", 100) == "unsatisfiable"
    assert parse_range("bytes=0-1,5-6", 100) is None
    assert parse_range("items=0-1", 100) is None


def test_mjpeg_live_stream_is_bounded(app_env, tmp_path):
    client, s, _, _ = app_env
    t = login(client)
    cam = seed_file_camera(client, make_video(tmp_path / "v.mp4", seconds=4), tokens=t)
    wait_for(lambda: client.get(f"/api/cameras/{cam['id']}/snapshot.jpg", headers=auth(t)).status_code == 200)
    r = client.get(cam["preview"]["mjpeg_url"] + "&fps=5&w=200")        # ends after mjpeg_max_seconds (1 s here)
    assert r.status_code == 200 and r.headers["content-type"].startswith("multipart/x-mixed-replace")
    parts = r.content.count(b"--frame\r\nContent-Type: image/jpeg")
    assert 1 <= parts <= 6 and b"\xff\xd8" in r.content
    stopped = client.post(f"/api/cameras/{cam['id']}/stop", headers=auth(t))
    assert stopped.status_code == 200
    wait_for(lambda: client.get(f"/api/cameras/{cam['id']}/snapshot.jpg", headers=auth(t)).status_code == 503)
