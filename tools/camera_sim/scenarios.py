"""Robustness scenarios for Sightline's RTSP ingest against the simulator (start_sim.sh must be running).

  ~/vjepa-venv/bin/python tools/camera_sim/scenarios.py [--only name,name] [--out results.json]

No GPU needed: this drives sightline_pipeline.ingest.VideoSource (the exact code path the pipeline and
the backend camera workers use) and the simulator's start/stop scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from sightline_pipeline.ingest import VideoSource  # noqa: E402

HERE = Path(__file__).resolve().parent
HOST = os.environ.get("CAMSIM_HOST", "127.0.0.1")
CRED = "admin:Sightline2026"
LOG = Path(os.environ.get("CAMSIM_LOG", "/mnt/f/models/logs/camsim"))
BIN = Path(os.environ.get("CAMSIM_BIN", "/mnt/f/models/camsim/bin"))
PUB = "rtsp://publisher:pub-local-only@127.0.0.1:8554"


def url(path, cred=CRED):
    return f"rtsp://{cred}@{HOST}:8554/{path}"


def sim(*args):
    subprocess.run(["bash", str(HERE / "start_sim.sh"), *args], check=True, capture_output=True)


def unsim(*args):
    subprocess.run(["bash", str(HERE / "stop_sim.sh"), *args], check=True, capture_output=True)


class Reader:
    """Read a source in a thread and keep a timeline of (wall, shape)."""

    def __init__(self, uri, **kw):
        self.src = VideoSource(uri, **kw)
        self.frames, self.t0 = [], time.time()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        for fr in self.src:
            self.frames.append((fr.wall, fr.image.shape))

    def stop(self):
        self.src.stop()
        self.th.join(10)

    def fps_between(self, a, b):
        n = sum(1 for w, _ in self.frames if self.t0 + a <= w < self.t0 + b)
        return round(n / (b - a), 1)

    def steady_fps(self):
        """fps after the first frame + 1 s (excludes connect / first-keyframe time)."""
        ws = [w for w, _ in self.frames]
        sel = [w for w in ws if ws and w >= ws[0] + 1.0]
        return round((len(sel) - 1) / (sel[-1] - sel[0]), 1) if len(sel) > 2 and sel[-1] > sel[0] else 0.0

    def first_frame_s(self):
        return round(self.frames[0][0] - self.t0, 2) if self.frames else None

    def max_gap(self):
        ws = [w for w, _ in self.frames]
        return round(max((b - a for a, b in zip(ws, ws[1:])), default=0), 2)


def custom_pub(path, vf, extra=(), seconds=None):
    args = [str(BIN / "ffmpeg"), "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i",
            str(REPO / "data/demo/sightline_demo.mp4"), "-an", "-vf", vf, "-c:v", "libx264", "-profile:v", "main",
            "-preset", "veryfast", "-tune", "zerolatency", "-g", "25", "-bf", "0", *extra, "-pix_fmt", "yuv420p",
            "-f", "rtsp", "-rtsp_transport", "tcp", f"{PUB}/{path}"]
    return subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_proc(p):
    p.terminate()
    try:
        p.wait(5)
    except subprocess.TimeoutExpired:
        p.kill()


# ------------------------------------------------------------------ scenarios
NOMINAL = {"Streams/101": 25, "Streams/102": 15, "Streams/201": 15, "Streams/301": 20}


def s_streams():
    out = {}
    for path in ("Streams/101", "Streams/102", "Streams/201", "Streams/301"):
        r = Reader(url(path))
        time.sleep(8)
        r.stop()
        sh = r.frames[-1][1] if r.frames else None
        out[path] = {"steady_fps": r.steady_fps(), "nominal_fps": NOMINAL[path], "first_frame_s": r.first_frame_s(),
                     "size": r.src.size, "frame_shape": sh, "error": r.src.last_error}
    ok = all(v["steady_fps"] >= 0.85 * v["nominal_fps"] and v["frame_shape"] for v in out.values())
    return ok, out


def s_udp():
    r = Reader(url("Streams/101"), rtsp_transport="udp")
    time.sleep(8)
    r.stop()
    fps = r.steady_fps()
    return fps >= 0.85 * 25, {"steady_fps": fps, "nominal_fps": 25, "first_frame_s": r.first_frame_s(), "max_gap_s": r.max_gap(), "size": r.src.size, "error": r.src.last_error}


def s_wrong_credentials():
    t = time.time()
    src = VideoSource(url("Streams/101", "admin:WRONG"), max_reconnects=0)
    n = len(list(src))
    dt = round(time.time() - t, 2)
    # with retries: how many connection attempts in 20 s (must not hammer the camera)
    attempts = []
    src2 = VideoSource(url("Streams/101", "admin:WRONG"), diagnose_fn=lambda u: (attempts.append(time.time()) or
                       __import__("sightline_pipeline.rtsp_probe", fromlist=["diagnose"]).diagnose(u)))
    th = threading.Thread(target=lambda: list(src2), daemon=True)
    th.start()
    time.sleep(20)
    src2.stop()
    th.join(5)
    ok = n == 0 and src.error and "authentication failed" in src.error and "WRONG" not in src.error and len(attempts) <= 2
    return bool(ok), {"error": src.error, "seconds_to_error": dt, "attempts_in_20s": len(attempts),
                      "state_while_retrying": src2.state}


def s_wrong_path():
    src = VideoSource(url("Streams/999"), max_reconnects=0)
    list(src)
    return bool(src.error and "not_found" in src.error), {"error": src.error}


def s_unreachable():
    out = {}
    for name, u in (("blackhole_ip", "rtsp://admin:x@10.255.255.1:554/Streams/101"),
                    ("closed_port", f"rtsp://admin:x@{HOST}:8559/Streams/101"),
                    ("bad_dns", "rtsp://admin:x@no-such-camera.invalid:554/Streams/101")):
        t = time.time()
        src = VideoSource(u, max_reconnects=0)
        list(src)
        out[name] = {"error": src.error, "seconds_to_error": round(time.time() - t, 2)}
    # cancel while reconnecting forever (used to hang the camera worker)
    ev = threading.Event()
    src = VideoSource("rtsp://admin:x@10.255.255.1:554/Streams/101", cancel=ev)
    th = threading.Thread(target=lambda: list(src), daemon=True)
    th.start()
    time.sleep(3)
    t = time.time()
    ev.set()
    th.join(15)
    out["cancel_while_reconnecting_s"] = round(time.time() - t, 2)
    ok = all(v["seconds_to_error"] < 20 and v["error"] for k, v in out.items() if isinstance(v, dict)) and not th.is_alive()
    return ok, out


def _disconnect(stop_args, start_args, path="Streams/201", down_s=6):
    r = Reader(url(path))
    time.sleep(5)
    before = r.steady_fps()
    unsim(*stop_args)
    t_down = time.time()
    time.sleep(down_s)
    state_down = r.src.state
    err_down = r.src.last_error
    sim(*start_args)
    t_up = time.time()
    while time.time() - t_up < 40 and not any(w > t_up for w, _ in r.frames):
        time.sleep(0.2)
    back = [w for w, _ in r.frames if w > t_up]
    recover = round(back[0] - t_up, 2) if back else None
    time.sleep(4)
    after_n = sum(1 for w, _ in r.frames if w > time.time() - 3) / 3
    r.stop()
    ok = recover is not None and after_n >= 10
    return ok, {"fps_before": before, "state_while_down": state_down, "error_while_down": err_down,
                "seconds_to_recover_after_restart": recover, "fps_after": round(after_n, 1),
                "reconnects": r.src.stats["reconnects"], "max_gap_s": r.max_gap(), "down_s": round(t_up - t_down, 1)}


def s_camera_drop():
    return _disconnect(("pub", "201"), ("pub", "201"))


def s_server_restart():
    ok, d = _disconnect(("server",), ("all",), path="Streams/101", down_s=8)
    return ok, d


def s_stall():
    pid = int((LOG / "pub_201.pid").read_text())
    r = Reader(url("Streams/201"), read_timeout_s=5)
    time.sleep(4)
    os.kill(pid, signal.SIGSTOP)          # camera freezes: TCP stays open, no packets
    t = time.time()
    while time.time() - t < 30 and r.src.state == "streaming":
        time.sleep(0.2)
    detect = round(time.time() - t, 2)
    os.kill(pid, signal.SIGCONT)
    t2 = time.time()
    while time.time() - t2 < 40 and not any(w > t2 for w, _ in r.frames):
        time.sleep(0.2)
    back = [w for w, _ in r.frames if w > t2]
    r.stop()
    rec = round(back[0] - t2, 2) if back else None
    return rec is not None and detect < 15, {"seconds_to_detect_stall": detect, "state": "reconnecting",
                                             "seconds_to_recover_after_resume": rec, "error": r.src.last_error}


def s_resolution_change():
    p = custom_pub("Streams/401", "scale=1280:720,fps=20")
    time.sleep(3)
    r = Reader(url("Streams/401"))
    time.sleep(5)
    stop_proc(p)                         # camera reconfigured: comes back at another resolution
    p = custom_pub("Streams/401", "scale=640:360,fps=20")
    t = time.time()
    while time.time() - t < 40 and not r.src.stats["resized_frames"]:
        time.sleep(0.2)
    time.sleep(3)
    r.stop()
    stop_proc(p)
    shapes = sorted({s for _, s in r.frames})
    ok = shapes == [(720, 1280, 3)] and r.src.stats["resolution_changes"] >= 1
    return ok, {"delivered_shapes": shapes, "camera_size_now": r.src.size, "resolution_changes": r.src.stats["resolution_changes"],
                "resized_frames": r.src.stats["resized_frames"], "reconnects": r.src.stats["reconnects"]}


def s_packet_loss():
    # drop ~3% of H.264 packets (and corrupt some bytes) at the publisher: what a lossy Wi-Fi camera looks like
    p = custom_pub("Streams/501", "scale=1280:720,fps=20", ("-bsf:v", "noise=drop='lt(random(0)\\,0.03)':amount=0"))
    time.sleep(3)
    r = Reader(url("Streams/501"))
    time.sleep(12)
    r.stop()
    stop_proc(p)
    fps = r.fps_between(2, 12)
    return fps >= 8, {"fps_with_3pct_loss": fps, "max_gap_s": r.max_gap(), "reconnects": r.src.stats["reconnects"],
                      "error": r.src.last_error}


SCEN = {"streams": s_streams, "udp": s_udp, "wrong_credentials": s_wrong_credentials, "wrong_path": s_wrong_path,
        "unreachable": s_unreachable, "camera_drop": s_camera_drop, "server_restart": s_server_restart,
        "stall": s_stall, "resolution_change": s_resolution_change, "packet_loss": s_packet_loss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(LOG / "scenarios.json"))
    a = ap.parse_args()
    names = [n for n in a.only.split(",") if n] or list(SCEN)
    res = {}
    for n in names:
        t = time.time()
        try:
            ok, d = SCEN[n]()
        except Exception as e:
            ok, d = False, {"exception": repr(e)}
        res[n] = {"pass": bool(ok), "seconds": round(time.time() - t, 1), **d}
        print(json.dumps({n: res[n]}, default=str), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    return 0 if all(v["pass"] for v in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
