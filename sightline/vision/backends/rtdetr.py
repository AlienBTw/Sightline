"""RT-DETRv2 closed-set backend — requires an explicit fine-tuned checkpoint path."""

from __future__ import annotations

from typing import Any

import numpy as np

from sightline.errors import SightlineError
from sightline.vision.backends.base import Detection


class RTDETRBackend:
    name = "rtdetr"

    def __init__(self, device: str = "cuda", checkpoint: str = "") -> None:
        self.device = device
        self.checkpoint = checkpoint
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint:
            raise SightlineError(
                error_code="detector_checkpoint_missing",
                component="vision.rtdetr",
                message="RT-DETR requires a checkpoint path",
                hint="Set RTDETR_CHECKPOINT or use DETECTOR_BACKEND=llmdet until a fine-tuned weight exists",
            )
        raise SightlineError(
            error_code="detector_not_ready",
            component="vision.rtdetr",
            message="RT-DETR loader not configured with a verified checkpoint in this MVP build",
            hint="Train/export a RT-DETRv2 weight, then wire it here — do not silently skip",
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
