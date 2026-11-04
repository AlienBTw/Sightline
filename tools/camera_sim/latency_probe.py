"""End-to-end latency for the camera rig: a stream whose every frame carries its capture time as a
barcode, read back through Sightline's ingest (and optionally the full model pipeline).

  python tools/camera_sim/latency_probe.py publish --seconds 120             # -> rtsp://.../Streams/901
  python tools/camera_sim/latency_probe.py ingest --seconds 30 [--udp]       # capture -> frame delivered
  python tools/camera_sim/latency_probe.py pipeline --seconds 60 --device cuda:1   # capture -> model step/alert

The barcode is 44 cells of 16x16 px (bit = black/white) along the bottom-left of a 1280x720 frame:
the capture time in ms (mod 2^42) at the moment the frame is handed to the encoder. Publisher and
reader run on the same machine (same clock), so the difference is the real glass-to-ingest delay
(x264 encode + RTSP + MediaMTX + FFmpeg demux/decode + Sightline's queue).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
BITS, CELL = 44, 16
URL_PUB = "rtsp://publisher:pub-local-only@127.0.0.1:8554/Streams/901"
URL_VIEW = "rtsp://admin:Sightline2026@127.0.0.1:8554/Streams/901"


def stamp(img: np.ndarray, ms: int):
    h = img.shape[0]
    y0 = h - CELL - 4
    img[y0 - 4:h, 0:BITS * CELL + 8] = 128
    for i in range(BITS):
        v = 255 if (ms >> (BITS - 1 - i)) & 1 else 0
        img[y0:y0 + CELL, 4 + i * CELL:4 + (i + 1) * CELL] = v


def read_stamp(img: np.ndarray) -> int | None:
    h = img.shape[0]
    y = h - CELL - 4 + CELL // 2
    if img.shape[1] < BITS * CELL + 8:
        return None
    row = img[y - 3:y + 4, :, :].mean(axis=(0, 2))
    ms = 0
    for i in range(BITS):
        c = row[4 + i * CELL + 3:4 + (i + 1) * CELL - 3].mean()
        if 80 < c < 175:                     # not a barcode frame
            return None
        ms = (ms << 1) | int(c >= 128)
    return ms


def now_ms() -> int:
    return int(time.time() * 1000) % (1 << BITS)


def publish(a):
    import cv2
    ff = os.environ.get("CAMSIM_FFMPEG", "/mnt/f/models/camsim/bin/ffmpeg")
    cap = cv2.VideoCapture(str(REPO / "data/demo/sightline_demo.mp4"))
    fps = 25.0
    p = subprocess.Popen([ff, "-hide_banner", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", "1280x720", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-profile:v", "main",
                          "-preset", "veryfast", "-tune", "zerolatency", "-g", "50", "-bf", "0", "-b:v", "3M",
                          "-pix_fmt", "yuv420p", "-f", "rtsp", "-rtsp_transport", "tcp", URL_PUB], stdin=subprocess.PIPE)
    t0, n = time.monotonic(), 0
    try:
        while a.seconds <= 0 or n < a.seconds * fps:
            ok, img = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            img = cv2.resize(img, (1280, 720))
            due = t0 + n / fps
            d = due - time.monotonic()
            if d > 0:
                time.sleep(d)
            stamp(img, now_ms())
            p.stdin.write(img.tobytes())
            n += 1
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        try:
            p.stdin.close()
        except Exception:
            pass
        p.wait(timeout=10)


def pct(v, q):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(len(v) * q / 100))], 1) if v else None


def ingest(a):
    from sightline_pipeline.ingest import VideoSource
    src = VideoSource(a.uri, rtsp_transport="udp" if a.udp else "tcp")
    lat, bad, t_end = [], 0, time.time() + a.seconds
    for fr in src:
        ms = read_stamp(fr.image)
        if ms is None:
            bad += 1
        else:
            lat.append((int(fr.wall * 1000) % (1 << BITS)) - ms)
        if time.time() > t_end:
            src.stop()
            break
    lat = [x for x in lat if 0 <= x < 60000]
    out = {"mode": "ingest", "transport": "udp" if a.udp else "tcp", "frames": len(lat), "unreadable": bad,
           "fps": src.fps, "size": src.size, "dropped": src.stats["dropped"],
           "capture_to_ingest_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "max": max(lat) if lat else None,
                                    "mean": round(statistics.mean(lat), 1) if lat else None}}
    print(json.dumps(out))
    return out


def pipeline(a):
    from sightline_pipeline.config import load_config
    from sightline_pipeline.ingest import VideoSource
    from sightline_pipeline.model import HazardModel
    from sightline_pipeline.service import SightlinePipeline
    cfg = load_config(None, {"escalation": {"enabled": False}, "model": {"device": a.device},
                             "storage": {"data_dir": a.data_dir}})
    model = HazardModel(cfg)
    pipe = SightlinePipeline(cfg, model)
    stamps = {}

    class Stamped(VideoSource):
        def __iter__(self):
            for fr in super().__iter__():
                ms = read_stamp(fr.image)
                if ms is not None:
                    stamps[fr.wall] = ms
                    if len(stamps) > 2000:
                        stamps.pop(next(iter(stamps)))
                yield fr

    step_lat, ingest_lat = [], []

    def on_step(info):
        ms = stamps.get(info["wall"])
        if ms is not None:
            done = int(time.time() * 1000) % (1 << BITS)
            step_lat.append(done - ms)
            ingest_lat.append(int(info["wall"] * 1000) % (1 << BITS) - ms)

    res = pipe.process(Stamped(a.uri), camera_id="cam_latency", max_seconds=a.seconds, log=lambda m: None,
                       on_step=on_step)
    s = res["summary"]
    out = {"mode": "pipeline", "steps": len(step_lat), "frames": s["frames"], "dropped": s["dropped_frames"],
           "capture_to_ingest_ms_at_step": {"p50": pct(ingest_lat, 50), "p95": pct(ingest_lat, 95)},
           "capture_to_alert_decision_ms": {"p50": pct(step_lat, 50), "p95": pct(step_lat, 95),
                                            "max": max(step_lat) if step_lat else None},
           "model_step_ms": s["latency_ms"], "events": len(s["events"])}
    print(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["publish", "ingest", "pipeline"])
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--uri", default=URL_VIEW)
    ap.add_argument("--udp", action="store_true")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--data-dir", default="/mnt/f/models/camsim/pipeline_out")
    a = ap.parse_args()
    {"publish": publish, "ingest": ingest, "pipeline": pipeline}[a.mode](a)


if __name__ == "__main__":
    main()
