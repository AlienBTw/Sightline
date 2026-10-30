import os

import numpy as np
import pytest

from sightline.config import Settings, get_settings
from sightline.errors import SightlineError
from sightline.vision.backends import build_detector
from sightline.vision.backends.replay import ReplayBackend


def test_replay_backend_loads_pack():
    det = ReplayBackend()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = det.detect(frame, camera_id="cam_01", frame_id="frm_000", timestamp=0.0)
    assert len(dets) >= 1
    assert dets[0].label


def test_build_detector_replay(monkeypatch):
    monkeypatch.setenv("SIGHTLINE_MODE", "replay")
    monkeypatch.setenv("DETECTOR_BACKEND", "replay")
    monkeypatch.setenv("REPORT_BACKEND", "ollama")
    get_settings.cache_clear()
    settings = get_settings()
    det = build_detector(settings)
    assert det.name == "replay"


def test_unknown_backend():
    settings = Settings.model_construct(  # type: ignore[attr-defined]
        sightline_mode="live",
        detector_backend="nope",  # type: ignore[arg-type]
        report_backend="ollama",
        torch_device="cpu",
        sightline_db="./data/sightline.db",
        sample_fps=2.0,
        api_url="http://127.0.0.1:8000",
        vlm_base_url="",
        vlm_model="",
        vlm_api_key="",
        ollama_base_url="http://localhost:11434/v1",
        ollama_model="muse-glimmer:30b",
        meta_api_base_url="",
        meta_api_key="",
        meta_api_model="",
    )
    # bypass literal by poking attribute
    object.__setattr__(settings, "detector_backend", "nope")
    with pytest.raises(SightlineError) as ei:
        build_detector(settings)
    assert ei.value.error_code == "detector_backend_unknown"
