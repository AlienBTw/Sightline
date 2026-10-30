"""Shift report — exactly one configured backend (ollama | meta_api), both OpenAI-compatible."""

from __future__ import annotations

import json
from typing import Any

import httpx

from sightline.config import Settings
from sightline.errors import SightlineError


def generate_shift_report(settings: Settings, payload: dict[str, Any]) -> str:
    backend = settings.report_backend
    system = (
        "Write a concise shift supervisor safety report. "
        "Do not invent hazards. Do not change risk scores. "
        "Use only the provided structured evidence."
    )
    user = json.dumps(payload, indent=2)
    if backend == "ollama":
        if not settings.ollama_base_url or not settings.ollama_model:
            raise SightlineError(
                error_code="report_ollama_not_configured",
                component="reporting.ollama",
                message="REPORT_BACKEND=ollama requires OLLAMA_BASE_URL and OLLAMA_MODEL",
                hint="Set OLLAMA_BASE_URL=http://localhost:11434/v1 and OLLAMA_MODEL=muse-glimmer:30b",
            )
        return _chat_completions(
            base_url=settings.ollama_base_url,
            api_key="",
            model=settings.ollama_model,
            system=system,
            user=user,
            component="reporting.ollama",
        )
    if backend == "meta_api":
        if not settings.meta_api_base_url or not settings.meta_api_key or not settings.meta_api_model:
            raise SightlineError(
                error_code="report_meta_api_not_configured",
                component="reporting.meta_api",
                message="REPORT_BACKEND=meta_api requires META_API_BASE_URL, META_API_KEY and META_API_MODEL",
                hint="Set the Meta Model API env vars or switch REPORT_BACKEND to ollama",
            )
        return _chat_completions(
            base_url=settings.meta_api_base_url,
            api_key=settings.meta_api_key,
            model=settings.meta_api_model,
            system=system,
            user=user,
            component="reporting.meta_api",
        )
    raise SightlineError(
        error_code="report_backend_unknown",
        component="reporting",
        message=f"Unknown REPORT_BACKEND={backend!r}",
        hint="Set REPORT_BACKEND to ollama or meta_api",
    )


def _chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    component: str,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        with httpx.Client(timeout=120) as client:
            r = client.post(base_url.rstrip("/") + "/chat/completions", headers=headers, json=payload)
        if r.status_code >= 400:
            raise SightlineError(
                error_code="report_http_error",
                component=component,
                message=f"HTTP {r.status_code}: {r.text[:400]}",
                hint="Check the report backend base URL (must include /v1) and model id",
            )
        data = r.json()
        return str(data["choices"][0]["message"]["content"])
    except SightlineError:
        raise
    except Exception as e:
        raise SightlineError(
            error_code="report_failed",
            component=component,
            message=str(e),
            hint="Fix the configured report backend; no silent template fallback",
        ) from e
