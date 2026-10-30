"""Video/webcam sampling at configured FPS."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from sightline.errors import SightlineError


def iter_video_frames(path: str | Path, sample_fps: float) -> Iterator[tuple[int, float, np.ndarray]]:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SightlineError(
            error_code="ingest_open_failed",
            component="ingest.sampler",
            message=f"Cannot open {path}",
            hint="Check path and codecs",
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / max(sample_fps, 0.1))))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield idx, idx / fps, frame
        idx += 1
    cap.release()


def open_webcam(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SightlineError(
            error_code="webcam_open_failed",
            component="ingest.sampler",
            message=f"Cannot open webcam index {index}",
            hint="Check device permissions / index",
        )
    return cap
