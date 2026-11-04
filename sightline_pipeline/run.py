"""CLI: run the live pipeline on one source end to end.

  python -m sightline_pipeline.run --source demo.mp4 --camera-id cam_demo           # paced like a live camera
  python -m sightline_pipeline.run --source demo.mp4 --fast                          # as fast as possible
  python -m sightline_pipeline.run --source rtsp://<phone-ip>:8554/live --camera-id phone_cam
Run in WSL with ~/vjepa-venv (needs torch + ~/vjepa2). Secrets come from the environment / .env
(MODEL_API_KEY for SAM 3.1 + Muse Spark escalation); without them escalation is recorded as skipped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from .config import REPO, load_config, resolve_path
from .ingest import VideoSource


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sightline_pipeline.run")
    ap.add_argument("--source", required=True, help="video file / uploaded video path, or rtsp:// / http:// URL")
    ap.add_argument("--camera-id", default="cam_01")
    ap.add_argument("--config", default=None, help="YAML overriding sightline_pipeline/config.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--fast", action="store_true", help="do not pace files to real time")
    ap.add_argument("--loop", action="store_true", help="loop a file forever (virtual camera)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--no-escalation", action="store_true")
    ap.add_argument("--device", default=None, help="e.g. cuda:1")
    ap.add_argument("--session-video", default=None, help="also write the whole stream with live circles (mp4)")
    ap.add_argument("--steps-out", default=None, help="write per-step scores (jsonl)")
    a = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env", override=False)     # secrets by env var name only; never printed
    except ImportError:
        pass
    ov = {}
    if a.data_dir:
        ov.setdefault("storage", {})["data_dir"] = a.data_dir
    if a.no_escalation:
        ov.setdefault("escalation", {})["enabled"] = False
    if a.device:
        ov.setdefault("model", {})["device"] = a.device
    cfg = load_config(a.config, ov)
    from .model import HazardModel
    from .service import SightlinePipeline
    model = HazardModel(cfg)
    print(json.dumps({"msg": "model_loaded", "seconds": round(model.load_s, 1), "device": str(model.device)}), flush=True)
    pipe = SightlinePipeline(cfg, model)
    src = VideoSource(a.source, realtime=not a.fast, loop=a.loop)
    res = pipe.process(src, camera_id=a.camera_id, max_seconds=a.max_seconds, session_video=a.session_video,
                       log=lambda m: print(m, flush=True))
    runs = resolve_path(cfg["storage"]["data_dir"]) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    (runs / f"{stamp}_{a.camera_id}_summary.json").write_text(json.dumps(res["summary"], indent=1), encoding="utf-8")
    steps_out = Path(a.steps_out) if a.steps_out else runs / f"{stamp}_{a.camera_id}_steps.jsonl"
    with open(steps_out, "w", encoding="utf-8") as f:
        for s in res["steps"]:
            f.write(json.dumps(s) + "\n")
    print(json.dumps({"msg": "done", **res["summary"]}, indent=1), flush=True)
    return 0 if not res["summary"].get("error") or res["summary"]["error"] == "stream ended" else 1


if __name__ == "__main__":
    sys.exit(main())
