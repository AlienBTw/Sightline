"""Probe VLM gate against VLM_BASE_URL. Fail loud on misconfig/JSON errors."""

from __future__ import annotations

import base64
import sys

import cv2
import numpy as np

from sightline.config import get_settings
from sightline.errors import SightlineError
from sightline.vision.vlm_gate import VLMGate


def main() -> int:
    try:
        settings = get_settings()
        if not settings.vlm_base_url:
            raise SightlineError(
                error_code="vlm_url_missing",
                component="scripts.smoke_vlm",
                message="VLM_BASE_URL is empty",
                hint="Set VLM_BASE_URL to an OpenAI-compatible endpoint, e.g. Ollama at http://localhost:11434/v1",
            )
        gate = VLMGate(settings)
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            raise RuntimeError("encode failed")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        out = gate.analyze_crop(b64, family="fire_heat")
        print(f"OK: vlm response keys={sorted(out.keys())}")
        return 0
    except SightlineError as e:
        print(f"FAIL: {e.to_dict()}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
