"""Per-factor confidence relative to its threshold, and an incident risk score (pure)."""
from __future__ import annotations


def confidence(p: float, thr: float) -> float:
    """Map a factor probability to [0, 1] so that its alert threshold sits at 0.5.
    Thresholds differ a lot per factor (clip-level calibration), so raw sigmoids are not comparable."""
    thr = min(max(thr, 1e-6), 1 - 1e-6)
    if p >= thr:
        return 0.5 + 0.5 * (p - thr) / (1 - thr)
    return 0.5 * p / thr


def noisy_or(values) -> float:
    q = 1.0
    for v in values:
        q *= 1.0 - min(max(v, 0.0), 1.0)
    return 1.0 - q
