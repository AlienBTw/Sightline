"""Sightline live inference pipeline (built during the Meta hackathon event window).

camera / RTSP / uploaded video -> ingest -> V-JEPA 2.1 + concept-bottleneck head (sliding
16-frame windows) -> temporal smoothing + per-factor thresholds -> incidents with circles
-> optional escalation (SAM 3.1 refine, Muse Spark verdict) -> clips + JSON + SQLite.

Pure modules (ingest pacing, events, localize, storage, render, escalation policy) import
without torch so they run in the Windows test suite; `model` needs torch + ~/vjepa2 (WSL).
"""
__version__ = "0.1.0"
