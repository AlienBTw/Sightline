"""Load exactly one detector backend from DETECTOR_BACKEND."""

from __future__ import annotations

from sightline.config import Settings
from sightline.errors import SightlineError
from sightline.vision.backends.base import DetectorBackend


def build_detector(settings: Settings) -> DetectorBackend:
    name = settings.detector_backend
    if name == "llmdet":
        from sightline.vision.backends.llmdet import LLMDetBackend

        return LLMDetBackend(device=settings.torch_device)
    if name == "owlv2":
        from sightline.vision.backends.owlv2 import OWLv2Backend

        return OWLv2Backend(device=settings.torch_device)
    if name == "rtdetr":
        from sightline.vision.backends.rtdetr import RTDETRBackend

        return RTDETRBackend(device=settings.torch_device)
    if name == "dfine":
        from sightline.vision.backends.dfine import DFineBackend

        return DFineBackend(device=settings.torch_device)
    if name == "replay":
        from sightline.vision.backends.replay import ReplayBackend

        return ReplayBackend()
    raise SightlineError(
        error_code="detector_backend_unknown",
        component="vision.factory",
        message=f"Unknown DETECTOR_BACKEND={name!r}",
        hint="Set DETECTOR_BACKEND to llmdet|owlv2|rtdetr|dfine|replay",
    )
