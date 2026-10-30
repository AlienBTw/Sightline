"""FastAPI entry — fail-loud JSON errors."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sightline import __version__
from sightline.api.schemas import AnalyzeVideoRequest, ErrorBody, HazardAction, ReportRequest
from sightline.config import get_settings
from sightline.errors import SightlineError
from sightline.pipeline import Pipeline
from sightline.reporting.report import generate_shift_report

app = FastAPI(title="Sightline API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Pipeline | None = None
_boot_error: dict[str, Any] | None = None


def get_pipeline() -> Pipeline:
    global _pipeline, _boot_error
    if _boot_error:
        raise SightlineError(**_boot_error)
    if _pipeline is None:
        try:
            _pipeline = Pipeline()
        except SightlineError as e:
            _boot_error = e.to_dict()
            raise
        except Exception as e:
            _boot_error = {
                "error_code": "pipeline_boot_failed",
                "component": "api",
                "message": str(e),
                "hint": "Check .env and dependencies",
                "details": {},
            }
            raise SightlineError(**_boot_error) from e
    return _pipeline


@app.exception_handler(SightlineError)
async def sightline_error_handler(_, exc: SightlineError) -> JSONResponse:
    return JSONResponse(status_code=500, content=exc.to_dict())


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        settings = get_settings()
    except SightlineError as e:
        return {"ok": False, "error": e.to_dict()}
    out: dict[str, Any] = {
        "ok": _boot_error is None,
        "version": __version__,
        "mode": settings.sightline_mode,
        "detector_backend": settings.detector_backend,
        "report_backend": settings.report_backend,
    }
    if _boot_error:
        out["error"] = _boot_error
    return out


@app.get("/system")
def system() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": None,
        "cuda_available": False,
        "cuda": None,
        "device_name": None,
        "detector_backend": None,
        "mode": None,
    }
    try:
        settings = get_settings()
        info["detector_backend"] = settings.detector_backend
        info["mode"] = settings.sightline_mode
        info["vlm_base_url_set"] = bool(settings.vlm_base_url)
    except SightlineError as e:
        raise e
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        raise SightlineError(
            error_code="system_probe_failed",
            component="api.system",
            message=str(e),
            hint="Install torch (CUDA build) or run CPU-only after setting TORCH_DEVICE=cpu",
        ) from e
    try:
        pipe = get_pipeline()
        info["latencies_ms"] = {
            k: {
                "count": len(v),
                "p50": float(np.percentile(v, 50)) if v else None,
                "p95": float(np.percentile(v, 95)) if v else None,
            }
            for k, v in pipe.latencies.items()
        }
    except SightlineError:
        info["latencies_ms"] = None
    return info


@app.post("/analyze/frame")
async def analyze_frame(
    camera_id: str = Form("cam_01"),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise SightlineError(
            error_code="frame_decode_failed",
            component="api.analyze_frame",
            message="Could not decode uploaded image",
            hint="Upload a JPEG/PNG frame",
        )
    return get_pipeline().analyze_frame(frame, camera_id=camera_id, timestamp=time.time())


@app.post("/analyze/video")
def analyze_video(body: AnalyzeVideoRequest) -> dict[str, Any]:
    return get_pipeline().analyze_video(body.path, camera_id=body.camera_id)


@app.get("/hazards")
def list_hazards(status: str | None = None) -> list[dict[str, Any]]:
    return get_pipeline().store.list_hazards(status=status)


@app.post("/hazards/fix")
def mark_fixed(body: HazardAction) -> dict[str, Any]:
    return get_pipeline().mark_fixed(body.hazard_id)


@app.post("/hazards/verify")
def verify_fixed(body: HazardAction) -> dict[str, Any]:
    return get_pipeline().verify_fixed(body.hazard_id)


@app.get("/cameras")
def cameras() -> dict[str, Any]:
    return {"cameras": get_pipeline().cameras, "zones": get_pipeline().zones}


@app.get("/calibration")
def calibration() -> dict[str, Any]:
    from pathlib import Path

    from sightline.decision.histgb import MODEL_DIR

    meta = MODEL_DIR / "histgb_isotonic.meta.json"
    if not meta.exists():
        raise SightlineError(
            error_code="calibration_missing",
            component="api.calibration",
            message="No HistGB metrics file found",
            hint="Train with scripts/train_histgb.py after labeling",
        )
    import json

    return json.loads(meta.read_text(encoding="utf-8"))


@app.post("/report/shift")
def report_shift(body: ReportRequest) -> dict[str, Any]:
    settings = get_settings()
    pipe = get_pipeline()
    hazards = pipe.store.list_hazards()
    payload = {
        "shift_id": body.shift_id,
        "site": body.site,
        "summary": {
            "total_hazards": len(hazards),
            "open": sum(1 for h in hazards if h.get("status") == "open"),
            "verified": sum(1 for h in hazards if h.get("status") == "verified"),
            "needs_human": sum(1 for h in hazards if h.get("status") == "needs_human"),
        },
        "top_hazards": sorted(hazards, key=lambda h: h.get("risk_score") or 0, reverse=True)[:10],
    }
    text = generate_shift_report(settings, payload)
    return {"backend": settings.report_backend, "report": text, "payload": payload}
