"""Backend path with real RTSP cameras: add cameras over the API as admin, check live MJPEG/snapshot,
camera status and events, measure load, then delete the cameras.

  SIGHTLINE_URL=http://127.0.0.1:8000 CAMTEST_USER=... CAMTEST_PASS=... \
  python tools/camera_sim/backend_check.py --seconds 120
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import httpx

CAMS = [("CAM01 main (H.264 1080p25, TCP)", "Streams/101", "tcp"),
        ("CAM03 HEVC (H.265 720p20, TCP)", "Streams/301", "tcp"),
        ("CAM01 sub (H.264 360p15, UDP)", "Streams/102", "udp")]


def gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        return repr(e)


def cpu_top():
    out = subprocess.run(["top", "-bn1", "-w", "200"], capture_output=True, text=True).stdout.splitlines()
    return [l for l in out[7:] if "python" in l or "ffmpeg" in l or "mediamtx" in l][:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--host", default=os.environ.get("CAMSIM_HOST", "127.0.0.1"))
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    base = os.environ.get("SIGHTLINE_URL", "http://127.0.0.1:8000")
    c = httpx.Client(base_url=base, timeout=30)
    tok = c.post("/api/auth/login", json={"username": os.environ["CAMTEST_USER"],
                                          "password": os.environ["CAMTEST_PASS"]}).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    rep = {"cameras": {}, "gpu_before": gpu()}
    ids = []
    for name, path, tr in CAMS:
        r = c.post("/api/cameras", headers=h, json={"name": name, "rtsp_transport": tr,
                   "uri": f"rtsp://admin:Sightline2026@{a.host}:8554/{path}"})
        j = r.json()
        ids.append(j["id"])
        rep["cameras"][j["id"]] = {"name": name, "http": r.status_code, "started": j.get("started"),
                                   "source_shown": j.get("source"), "start_error": j.get("start_error")}
    t0 = time.time()
    while time.time() - t0 < 120:
        sts = [c.get(f"/api/cameras/{i}", headers=h).json()["status"] for i in ids]
        if all(s.get("state") == "running" and s.get("frames", 0) > 50 for s in sts):
            break
        time.sleep(3)
    rep["seconds_to_all_running"] = round(time.time() - t0, 1)
    for i in ids:
        cam = c.get(f"/api/cameras/{i}", headers=h).json()
        snap = c.get(cam["preview"]["snapshot_url"])
        t = time.time()
        n, size = 0, 0
        with c.stream("GET", cam["preview"]["mjpeg_url"] + "&fps=2") as s:
            for chunk in s.iter_bytes():
                size += len(chunk)
                n += chunk.count(b"--frame")
                if time.time() - t > 5:
                    break
        rep["cameras"][i].update({"snapshot_http": snap.status_code, "snapshot_bytes": len(snap.content),
                                  "snapshot_frame_age_s": snap.headers.get("x-frame-age"),
                                  "mjpeg_parts_5s": n, "mjpeg_kb_5s": round(size / 1024, 1)})
    samples = {i: [] for i in ids}
    while time.time() - t0 < a.seconds:
        time.sleep(10)
        for i in ids:
            st = c.get(f"/api/cameras/{i}", headers=h).json()["status"]
            samples[i].append((st.get("fps") or 0, st.get("last_frame_age_s") or 0, st.get("device")))
    for i in ids:
        sm = samples[i][1:] or samples[i]          # skip the warm-up sample
        if sm:
            rep["cameras"][i]["steady"] = {"device": sm[-1][2], "fps_mean": round(sum(x[0] for x in sm) / len(sm), 1),
                                           "fps_min": min(x[0] for x in sm),
                                           "frame_age_mean_s": round(sum(x[1] for x in sm) / len(sm), 2),
                                           "frame_age_max_s": max(x[1] for x in sm), "samples": len(sm)}
    rep["gpu_during"] = gpu()
    rep["cpu_during"] = cpu_top()
    ev = c.get("/api/events", headers=h, params={"limit": 100}).json()
    items = ev.get("items", ev if isinstance(ev, list) else [])
    for i in ids:
        st = c.get(f"/api/cameras/{i}", headers=h).json()["status"]
        rep["cameras"][i].update({"state": st.get("state"), "fps": st.get("fps"), "frames": st.get("frames"),
                                  "events": st.get("events"), "restarts": st.get("restarts"),
                                  "source": st.get("source"), "last_frame_age_s": st.get("last_frame_age_s"),
                                  "events_in_api": sum(1 for e in items if e.get("camera_id") == i)})
    if not a.keep:
        for i in ids:
            rep["cameras"][i]["delete_http"] = c.delete(f"/api/cameras/{i}", headers=h).status_code
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
