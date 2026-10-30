"""Risk = calibrated_probability × severity (exposure fixed at 1 for MVP)."""

from __future__ import annotations


def risk_score(probability: float, severity: float) -> float:
    return float(max(0.0, min(1.0, probability)) * max(0.0, min(1.0, severity)))
