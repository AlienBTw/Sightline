from __future__ import annotations

import streamlit as st

from app.ui_common import api_post, empty_state, error_callout


def render() -> None:
    st.caption("Uses exactly one configured REPORT_BACKEND (ollama = local Muse Glimmer, or meta_api = Meta Model API). Failures surface as errors — no template fallback.")
    if st.button("Generate shift report", type="primary"):
        data, err = api_post("/report/shift", {"shift_id": "shift_local", "site": "demo_workshop"})
        if err:
            error_callout(err)
            return
        assert data is not None
        st.caption(f"Backend: `{data.get('backend')}`")
        st.markdown(data.get("report") or "")
        st.session_state["last_report"] = data
        st.download_button(
            "Download report.md",
            data=data.get("report") or "",
            file_name="sightline_shift_report.md",
            mime="text/markdown",
        )
    last = st.session_state.get("last_report")
    if not last:
        empty_state("No report yet. Configure REPORT_BACKEND credentials, then generate.")
