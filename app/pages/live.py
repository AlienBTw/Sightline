from __future__ import annotations

import streamlit as st

from app.ui_common import api_post, empty_state, error_callout


def render() -> None:
    st.markdown("Analyze a recorded clip or uploaded frame via the API.")
    path = st.text_input("Video path (server-local)", value="")
    camera_id = st.text_input("Camera ID", value="cam_01")
    if st.button("Analyze video", type="primary"):
        if not path.strip():
            error_callout(
                {
                    "error_code": "ui_path_required",
                    "component": "ui.live",
                    "message": "Video path is empty",
                    "hint": "Provide a path the API process can read",
                }
            )
            return
        data, err = api_post("/analyze/video", {"path": path.strip(), "camera_id": camera_id})
        if err:
            error_callout(err)
            return
        st.session_state["last_analyze"] = data
        st.success(f"Analyzed {data.get('frames_analyzed')} frames")
    last = st.session_state.get("last_analyze")
    if not last:
        empty_state("No analysis yet. Run a video or wait for detections.")
        return
    frames = last.get("results") or []
    if not frames:
        empty_state("Analysis returned no frames.")
        return
    last_frame = frames[-1]
    dets = last_frame.get("detections") or []
    st.subheader("Latest frame detections")
    if not dets:
        empty_state("No detections on the last sampled frame.")
        return
    st.dataframe(
        [
            {
                "label": d.get("label"),
                "score": round(float(d.get("score", 0)), 3),
                "bbox": d.get("bbox_xyxy"),
            }
            for d in dets
        ],
        use_container_width=True,
        hide_index=True,
    )
