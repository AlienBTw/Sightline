from __future__ import annotations

import streamlit as st

from app.ui_common import api_get, empty_state, error_callout


def render() -> None:
    data, err = api_get("/calibration")
    if err:
        error_callout(err)
        empty_state("Train HistGB after labeling: python scripts/bootstrap_labels.py && python scripts/train_histgb.py --bootstrap data/models/bootstrap_labels.csv")
        return
    assert isinstance(data, dict)
    st.markdown("**Shipped model:** HistGradientBoosting + isotonic")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ECE", f"{float(data.get('ece', float('nan'))):.3f}")
    c2.metric("Brier", f"{float(data.get('brier', float('nan'))):.3f}")
    c3.metric("n_train", data.get("n_train"))
    c4.metric("n_val", data.get("n_val"))
    st.caption(
        "Holdout metrics on labeled candidates. "
        f"Label source: `{data.get('label_source', 'unknown')}`. "
        "MVP sample sizes are small — do not treat ECE as certification. "
        "Not a certified safety system."
    )
    # Simple reliability-style bars if present
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Bar(
                    x=["ECE", "Brier", "log_loss"],
                    y=[
                        float(data.get("ece") or 0),
                        float(data.get("brier") or 0),
                        float(data.get("log_loss") or 0),
                    ],
                    marker_color="#c45c26",
                )
            ]
        )
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f4f6f8",
            font=dict(family="IBM Plex Sans", color="#1a2332"),
            yaxis=dict(gridcolor="#d8dee6"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.json(data)
    with st.expander("Raw metrics JSON"):
        st.json(data)
