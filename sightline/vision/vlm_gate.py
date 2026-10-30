"""Sparse VLM evidence gate — attributes only, never the final hazard verdict.

Any OpenAI-compatible endpoint; local Muse Glimmer via Ollama by default.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from sightline.config import Settings, load_yaml
from sightline.errors import SightlineError

VLM_SCHEMA_KEYS = {
    "source_active",
    "source_type",
    "material",
    "material_combustible",
    "liquid_present",
    "uncertainty",
    "evidence",
}

PROMPT = """Describe hazard attributes only. Do not say whether this is a hazard or assign probability.
Return JSON only with keys:
source_active (bool|null), source_type (string|null), material (string|null),
material_combustible (bool|null), liquid_present (bool|null),
uncertainty ("low"|"medium"|"high"), evidence (short string).
"""


class VLMGate:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.thresholds = load_yaml("thresholds.yaml").get("vlm") or {}
        self.timeout = float(self.thresholds.get("timeout_s", 60))

    def should_gate(self, family: str, probability: float | None) -> bool:
        # Without a configured endpoint, do not enter the gate path.
        # analyze_crop still fails loud if called with an empty VLM_BASE_URL.
        if not self.settings.vlm_base_url:
            return False
        always = set(self.thresholds.get("always_gate") or [])
        if family in always:
            return True
        if probability is None:
            return False
        lo = float(self.thresholds.get("gate_probability_low", 0.35))
        hi = float(self.thresholds.get("gate_probability_high", 0.75))
        return lo <= probability <= hi

    def analyze_crop(self, image_b64: str, *, family: str) -> dict[str, Any]:
        if not self.settings.vlm_base_url:
            raise SightlineError(
                error_code="vlm_not_configured",
                component="vision.vlm_gate",
                message="VLM_BASE_URL is empty but a gated candidate requires VLM evidence",
                hint="Set VLM_BASE_URL (e.g. http://localhost:11434/v1 for Ollama), or avoid gated families until configured",
            )
        url = self.settings.vlm_base_url.rstrip("/") + "/chat/completions"
        headers = (
            {"Authorization": f"Bearer {self.settings.vlm_api_key}"} if self.settings.vlm_api_key else {}
        )
        model = self.settings.vlm_model
        if not model:
            raise SightlineError(
                error_code="vlm_model_missing",
                component="vision.vlm_gate",
                message="VLM_MODEL is empty",
                hint="Set VLM_MODEL (e.g. muse-glimmer:30b)",
            )
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{PROMPT}\nFamily context: {family}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        }
        retries = int(self.thresholds.get("max_retries", 1))
        last_err: Exception | None = None
        for _ in range(retries + 1):
            t0 = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    r = client.post(url, headers=headers, json=payload)
                if r.status_code >= 400:
                    raise SightlineError(
                        error_code="vlm_http_error",
                        component="vision.vlm_gate",
                        message=f"VLM HTTP {r.status_code}: {r.text[:500]}",
                        hint="Check the VLM server (Ollama running?) and model name",
                    )
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                parsed = self._parse_json(text)
                parsed["_latency_ms"] = (time.perf_counter() - t0) * 1000
                return parsed
            except SightlineError as e:
                last_err = e
            except Exception as e:
                last_err = e
        raise SightlineError(
            error_code="vlm_failed",
            component="vision.vlm_gate",
            message=str(last_err),
            hint="Fix VLM endpoint/JSON; gated analyze will not invent empty evidence",
        )

    def _parse_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise SightlineError(
                error_code="vlm_invalid_json",
                component="vision.vlm_gate",
                message=f"VLM did not return valid JSON: {e}",
                hint="Tighten the prompt or lower temperature",
            ) from e
        if not isinstance(obj, dict):
            raise SightlineError(
                error_code="vlm_invalid_json",
                component="vision.vlm_gate",
                message="VLM JSON root must be an object",
                hint="Require object schema in the prompt",
            )
        missing = VLM_SCHEMA_KEYS - set(obj.keys())
        if missing:
            raise SightlineError(
                error_code="vlm_schema_incomplete",
                component="vision.vlm_gate",
                message=f"Missing keys: {sorted(missing)}",
                hint="Update the VLM prompt to require all schema keys",
            )
        return obj
