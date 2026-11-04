"""Live preview frames: latest camera frame, downscaled, with the current step's circles drawn."""
from __future__ import annotations

import time

from sightline_pipeline.render import draw_overlays
from sightline_pipeline.storage import resize_w


def render_preview(worker, name: str, width: int, overlay: bool, quality: int) -> tuple[bytes, float]:
    import cv2
    idx, wall, img = worker.latest
    key = (idx, width, overlay)
    cache = getattr(worker, "_pcache", None)
    if cache and cache[0] == key:
        return cache[1], time.time() - wall
    small = resize_w(img, width)
    ov = []
    st = worker.last_step
    hold = max(2.0, 2.5 * float(getattr(worker, "stride_s", 1.0)))
    if overlay and st and time.time() - st.get("wall_rx", 0) <= hold:
        ov = st.get("active") or []
    facs = sorted({o["factor"] for o in ov})
    hdr = f"{name}  LIVE  " + ("ALERT " + ",".join(facs) if facs else "ok")
    out = draw_overlays(small, ov, header=hdr) if overlay else small
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    jpg = buf.tobytes()
    worker._pcache = (key, jpg)
    return jpg, time.time() - wall
