"""Shared Streamlit chrome: CSS, top bar, error callout, API client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def inject_css() -> None:
    css = (ROOT / "app" / "styles" / "sightline.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def api_base() -> str:
    import os

    return os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")


def topbar(mode: str | None = None, error: bool = False) -> None:
    mode_cls = "sl-mode sl-mode-error" if error else "sl-mode"
    label = mode or "unknown"
    st.markdown(
        f"""
        <div class="sl-topbar">
          <div class="sl-brand">Sightline</div>
          <div class="{mode_cls}">MODE: {label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def error_callout(err: dict[str, Any]) -> None:
    code = err.get("error_code", "error")
    msg = err.get("message", "Unknown error")
    hint = err.get("hint", "")
    component = err.get("component", "")
    st.markdown(
        f"""
        <div class="sl-error">
          <div><code>{code}</code> · {component}</div>
          <div style="margin-top:6px">{msg}</div>
          <div class="hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(text: str) -> None:
    st.markdown(f'<div class="sl-empty">{text}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        '<div class="sl-footer">Decision-support demo · not a certified safety system</div>',
        unsafe_allow_html=True,
    )


def api_get(path: str) -> tuple[dict[str, Any] | list[Any] | None, dict[str, Any] | None]:
    try:
        with httpx.Client(timeout=60) as client:
            r = client.get(api_base() + path)
        data = r.json()
        if r.status_code >= 400:
            if isinstance(data, dict) and "error_code" in data:
                return None, data
            return None, {
                "error_code": "api_http_error",
                "component": "ui",
                "message": f"HTTP {r.status_code}: {json.dumps(data)[:400]}",
                "hint": "Check API_URL and that uvicorn is running",
            }
        return data, None
    except Exception as e:
        return None, {
            "error_code": "api_unreachable",
            "component": "ui",
            "message": str(e),
            "hint": f"Start the API at {api_base()} or fix API_URL",
        }


def api_post(path: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        with httpx.Client(timeout=120) as client:
            r = client.post(api_base() + path, json=payload)
        data = r.json()
        if r.status_code >= 400:
            if isinstance(data, dict) and "error_code" in data:
                return None, data
            return None, {
                "error_code": "api_http_error",
                "component": "ui",
                "message": f"HTTP {r.status_code}",
                "hint": "See API logs",
            }
        return data, None
    except Exception as e:
        return None, {
            "error_code": "api_unreachable",
            "component": "ui",
            "message": str(e),
            "hint": f"Start the API at {api_base()}",
        }
