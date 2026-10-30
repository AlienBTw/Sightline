"""D-FINE closed-set backend — requires an explicit fine-tuned checkpoint path."""

from __future__ import annotations

from typing import Any

import numpy as np

from sightline.errors import SightlineError
from sightline.vision.backends.base import Detection


class DFineBackend:
    name = "dfine"

    def __init__(self, device: str = "cuda", checkpoint: str = "") -> None:
        self.device = device
        self.checkpoint = checkpoint
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        raise SightlineError(
            error_code="detector_not_ready",
            component="vision.dfine",
            message="D-FINE loader not configured with a verified checkpoint in this MVP build",
            hint="Fine-tune D-FINE on site clips, then wire the checkpoint — do not silently skip",
        )

    def detect(
        self,
        frame: np.ndarray,
        *,
        camera_id: str,
        frame_id: str,
        timestamp: float,
    ) -> list[Detection]:
        self._ensure_loaded()
        return []
