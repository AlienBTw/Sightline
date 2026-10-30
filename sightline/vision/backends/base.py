"""Detector backend protocol — one backend loaded from config, never cascaded."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Detection:
    camera_id: str
    frame_id: str
    timestamp: float
    label: str
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


class DetectorBackend(Protocol):
    name: str

    def detect(self, frame: np.ndarray, *, camera_id: str, frame_id: str, timestamp: float) -> list[Detection]:
        ...
