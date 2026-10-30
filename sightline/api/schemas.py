from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    error_code: str
    component: str
    message: str
    hint: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class AnalyzeVideoRequest(BaseModel):
    path: str
    camera_id: str = "cam_01"


class HazardAction(BaseModel):
    hazard_id: str


class ReportRequest(BaseModel):
    shift_id: str = "shift_local"
    site: str = "demo_workshop"
