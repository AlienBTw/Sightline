from __future__ import annotations

import streamlit as st

from app.ui_common import api_get, api_post, empty_state, error_callout


def render() -> None:
    hazards, err = api_get("/hazards")
    if err:
        error_callout(err)
        return
    assert isinstance(hazards, list)
    if not hazards:
        empty_state("No hazards yet. Analyze a clip from Live or run scripts/demo_e2e_replay.py")
        return
    status_filter = st.selectbox(
        "Status filter",
        ["all", "open", "needs_human", "fix_pending", "verified", "candidate", "confirmed"],
    )
    rows = hazards if status_filter == "all" else [h for h in hazards if h.get("status") == status_filter]
    if not rows:
        empty_state("No hazards for this filter.")
        return

    hdr = st.columns([0.9, 1.2, 1.1, 0.9, 1.2, 1.6, 1.8])
    for col, label in zip(
        hdr,
        ["Risk", "Family", "Status", "P", "Zone", "ID", "Actions"],
        strict=True,
    ):
        col.caption(label)

    for h in sorted(rows, key=lambda x: float(x.get("risk_score") or 0), reverse=True):
        cols = st.columns([0.9, 1.2, 1.1, 0.9, 1.2, 1.6, 1.8])
        risk = float(h.get("risk_score") or 0)
        cols[0].markdown(f"`{risk:.2f}`")
        cols[1].write(h.get("hazard_family"))
        cols[2].write(h.get("status"))
        p = float((h.get("evidence") or {}).get("decision", {}).get("calibrated_probability") or 0)
        cols[3].markdown(f"`{p:.2f}`")
        loc = h.get("location") or {}
        cols[4].write(loc.get("zone_id") or "—")
        cols[5].caption(h.get("hazard_id"))
        with cols[6]:
            c1, c2 = st.columns(2)
            if c1.button("Fix", key=f"fix_{h['hazard_id']}"):
                _, e = api_post("/hazards/fix", {"hazard_id": h["hazard_id"]})
                if e:
                    error_callout(e)
                else:
                    st.rerun()
            if c2.button("Verify", key=f"ver_{h['hazard_id']}"):
                _, e = api_post("/hazards/verify", {"hazard_id": h["hazard_id"]})
                if e:
                    error_callout(e)
                else:
                    st.rerun()
        ttf = h.get("time_to_fix_seconds")
        if ttf is not None:
            st.caption(f"time_to_fix = {float(ttf):.1f}s for `{h.get('hazard_id')}`")
