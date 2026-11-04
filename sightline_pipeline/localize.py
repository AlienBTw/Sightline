"""Patch map -> circle(s), plus circle geometry used for evaluation (pure numpy + cv2).

The head's per-factor patch map lives on the pooled 12 x 21 token grid of a 384 x 672 input.
The encoder input is the full frame squashed (no crop), so normalised map coordinates are
normalised frame coordinates: upsample to the frame aspect, keep the connected region around
the peak that is >= rel_threshold * peak, and fit a circle (mass-weighted centre, radius from
the region area, clamped).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Circle:
    cx: float           # pixels in the frame the circle was computed for
    cy: float
    r: float
    w: int              # frame size, so coordinates can be normalised for the app
    h: int
    score: float = 0.0  # map peak value
    source: str = "head"   # head | sam3.1

    def norm(self) -> dict:
        """Normalised for the app: centre in [0,1], radius as an ellipse (rx by width, ry by height)."""
        return {"cx": round(self.cx / self.w, 4), "cy": round(self.cy / self.h, 4),
                "rx": round(self.r / self.w, 4), "ry": round(self.r / self.h, 4), "angle_deg": 0}

    def scaled(self, w: int, h: int) -> "Circle":
        sx, sy = w / self.w, h / self.h
        return Circle(self.cx * sx, self.cy * sy, self.r * math.sqrt(sx * sy), w, h, self.score, self.source)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({k: round(v, 2) for k, v in d.items() if isinstance(v, float)})
        d["norm"] = self.norm()
        return d


def upsample(m: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    import cv2
    return cv2.resize(np.asarray(m, np.float32), (int(out_w), int(out_h)), interpolation=cv2.INTER_CUBIC)


def circles_from_map(m: np.ndarray, frame_w: int, frame_h: int, rel_threshold: float = 0.55,
                     min_radius_frac: float = 0.04, max_radius_frac: float = 0.35,
                     radius_scale: float = 1.25, max_circles: int = 1, work_w: int = 336) -> list[Circle]:
    """Circles for one factor map (any grid size). Works at a reduced resolution, then scales."""
    import cv2
    m = np.asarray(m, np.float32)
    if m.size == 0 or not np.isfinite(m).any() or float(np.nanmax(m) - np.nanmin(m)) < 1e-9:
        return []
    work_h = max(1, int(round(work_w * frame_h / frame_w)))
    u = upsample(m, work_w, work_h)
    lo, hi = float(u.min()), float(u.max())
    if hi - lo < 1e-9:
        return []
    un = (u - lo) / (hi - lo)
    mask = (un >= rel_threshold).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for k in range(1, n):
        sel = lab == k
        comps.append((float(un[sel].max()), k, sel))
    comps.sort(key=lambda c: -c[0])
    diag = math.hypot(frame_w, frame_h)
    s = frame_w / work_w   # pixels per work pixel (aspect kept)
    out = []
    for peak, k, sel in comps[:max_circles]:
        if out and peak < 0.8:      # secondary circles only for near-equal peaks
            break
        ys, xs = np.nonzero(sel)
        wts = un[ys, xs]
        cx, cy = float((xs * wts).sum() / wts.sum()), float((ys * wts).sum() / wts.sum())
        r = math.sqrt(sel.sum() / math.pi) * radius_scale * s
        r = min(max(r, min_radius_frac * diag), max_radius_frac * diag)
        out.append(Circle((cx + 0.5) * frame_w / work_w, (cy + 0.5) * frame_h / work_h,
                          r, int(frame_w), int(frame_h), score=float(lo + peak * (hi - lo))))
    return out


def box_to_circle(x1: float, y1: float, x2: float, y2: float, w: int, h: int, source="box") -> Circle:
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return Circle(cx, cy, 0.5 * math.hypot(x2 - x1, y2 - y1), w, h, source=source)


def contains(c: Circle, x: float, y: float) -> bool:
    return (x - c.cx) ** 2 + (y - c.cy) ** 2 <= c.r ** 2


def circle_box_iou(c: Circle, box: tuple, grid: int = 256) -> float:
    """IoU between a circle and an axis-aligned box (x1, y1, x2, y2), rasterised on the union bbox."""
    x1, y1, x2, y2 = box
    ux0, uy0 = min(x1, c.cx - c.r), min(y1, c.cy - c.r)
    ux1, uy1 = max(x2, c.cx + c.r), max(y2, c.cy + c.r)
    if ux1 <= ux0 or uy1 <= uy0:
        return 0.0
    xs = np.linspace(ux0, ux1, grid)
    ys = np.linspace(uy0, uy1, grid)
    X, Y = np.meshgrid(xs, ys)
    inc = (X - c.cx) ** 2 + (Y - c.cy) ** 2 <= c.r ** 2
    inb = (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)
    u = (inc | inb).sum()
    return float((inc & inb).sum() / u) if u else 0.0


def yolo_to_xyxy(b: tuple, w: int, h: int) -> tuple:
    cx, cy, bw, bh = b
    return ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


def circle_iou(a: Circle, b: Circle) -> float:
    d = math.hypot(a.cx - b.cx, a.cy - b.cy)
    r1, r2 = a.r, b.r
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        inter = math.pi * min(r1, r2) ** 2
    else:
        a1 = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
        a2 = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
        a3 = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        inter = a1 + a2 - a3
    return inter / (math.pi * r1 * r1 + math.pi * r2 * r2 - inter)
