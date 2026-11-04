"""Draw hazard circles and labels on frames (BGR, OpenCV)."""
from __future__ import annotations

import numpy as np

from .config import FACTOR_LABELS

COLORS = {  # BGR
    "F1": (40, 40, 255), "F2": (0, 200, 255), "F5": (0, 255, 255), "F6": (255, 0, 255),
    "F7": (0, 140, 255), "F8": (255, 120, 0), "F9": (255, 255, 0), "F10": (0, 0, 200), "F11": (200, 200, 200),
}


def draw_overlays(img: np.ndarray, overlays: list[dict], header: str | None = None) -> np.ndarray:
    """overlays: [{"factor", "score", "circle": Circle-like dict with norm cx/cy/rx}]"""
    import cv2
    out = img.copy()
    h, w = out.shape[:2]
    th = max(2, int(round(w / 320)))
    fs = max(0.5, w / 1600)
    for ov in overlays:
        f = ov["factor"]
        n = ov["circle"]["norm"]
        cx, cy, r = int(n["cx"] * w), int(n["cy"] * h), max(4, int(n["rx"] * w))
        col = COLORS.get(f, (0, 0, 255))
        cv2.circle(out, (cx, cy), r, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), r, col, th, cv2.LINE_AA)
        label = f"{FACTOR_LABELS.get(f, f)} {ov.get('score', 0):.2f}"
        if ov["circle"].get("source", "head") != "head":
            label += f" [{ov['circle']['source']}]"
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        x = min(max(0, cx - tw // 2), max(0, w - tw - 4))
        y = cy - r - 8 if cy - r - 8 > tht + 4 else min(h - 4, cy + r + tht + 8)
        cv2.rectangle(out, (x - 3, y - tht - 5), (x + tw + 3, y + 4), (0, 0, 0), -1)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
    if header:
        cv2.rectangle(out, (0, 0), (w, int(28 * fs * 1.6)), (0, 0, 0), -1)
        cv2.putText(out, header, (8, int(20 * fs * 1.6)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out
