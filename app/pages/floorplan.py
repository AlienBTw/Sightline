from __future__ import annotations

import streamlit as st

from app.ui_common import api_get, empty_state, error_callout


def render() -> None:
    data, err = api_get("/cameras")
    if err:
        error_callout(err)
        return
    assert data is not None
    cams = data.get("cameras") or []
    zones = data.get("zones") or {}
    if not cams:
        empty_state("No cameras configured. Edit config/sites/demo_workshop/cameras.json")
        return
    st.subheader("Cameras")
    st.json(cams)
    st.caption(
        "Calibration: four pixel ↔ plan points + known_length_m in "
        "`config/sites/demo_workshop/cameras.json`. Run "
        "`python scripts/validate_calibration.py` after edits."
    )
    st.subheader("Zones")
    st.json(zones)
    hazards, herr = api_get("/hazards")
    if herr:
        error_callout(herr)
        return
    st.subheader("Hazard pins (plan metres)")
    if not hazards:
        empty_state("No hazards yet. Calibrate and analyze a clip.")
        return
    rows = []
    for h in hazards:
        loc = h.get("location") or {}
        rows.append(
            {
                "id": h.get("hazard_id"),
                "family": h.get("hazard_family"),
                "status": h.get("status"),
                "risk": round(float(h.get("risk_score") or 0), 3),
                "x_m": loc.get("floor_x"),
                "y_m": loc.get("floor_y"),
                "zone": loc.get("zone_id"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
