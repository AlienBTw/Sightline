"""Sightline Streamlit entry."""

from __future__ import annotations

import streamlit as st

from app.ui_common import api_get, error_callout, footer, inject_css, topbar

st.set_page_config(
    page_title="Sightline",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

health, err = api_get("/health")
mode = (health or {}).get("mode") if isinstance(health, dict) else None
topbar(mode=mode, error=err is not None or (isinstance(health, dict) and health.get("ok") is False))

st.sidebar.markdown("### Sightline")
st.sidebar.caption("Clear line of sight over every latent hazard.")
page = st.sidebar.radio(
    "Pages",
    ["Live", "Floor plan", "Fix list", "Calibration", "Report"],
    label_visibility="collapsed",
)

if err:
    error_callout(err)
elif isinstance(health, dict) and health.get("ok") is False and health.get("error"):
    error_callout(health["error"])

st.title(page)
st.caption("Shift-supervisor workplace hazard inspector")

if page == "Live":
    from app.pages import live

    live.render()
elif page == "Floor plan":
    from app.pages import floorplan

    floorplan.render()
elif page == "Fix list":
    from app.pages import fixlist

    fixlist.render()
elif page == "Calibration":
    from app.pages import calibration

    calibration.render()
elif page == "Report":
    from app.pages import report

    report.render()

footer()
