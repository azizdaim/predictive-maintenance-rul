"""Landing page: what the project does, the 3D engine, and headline results.

The hero renders first and live numbers fill in afterwards, so visitors never
stare at a spinner while all 100 engines are scored.
"""

import httpx
import streamlit as st

from src.dashboard.components.engine3d import engine_3d
from src.dashboard.data import get_fleet, get_model_info

hero_text, hero_engine = st.columns([1, 1.15], gap="large", vertical_alignment="center")

with hero_text:
    st.badge("NASA C-MAPSS turbofan data", icon=":material/flight:", color="blue")
    st.title("Know how long an engine has left, before it fails")
    st.markdown(
        "A machine learning system that reads an engine's sensor history and "
        "predicts its **remaining useful life**, in flight cycles. It runs end to "
        "end: a tested model, a FastAPI prediction service, and this live dashboard."
    )
    kpi_row = st.container(horizontal=True)

    with st.container(horizontal=True):
        if st.button(
            "Open fleet overview", icon=":material/dashboard:", type="primary"
        ):
            st.switch_page("app_pages/fleet.py")
        if st.button("Inspect an engine", icon=":material/monitoring:"):
            st.switch_page("app_pages/engine.py")
    api_notice = st.container()

with hero_engine:
    engine_3d(
        label="Interactive cutaway · hover a section",
        fan_speed=0.7,
        glow=0.55,
        height=460,
        key="hero_engine",
    )

st.space("medium")
st.subheader("How it works")

steps = [
    (
        ":material/sensors:",
        "1 · Sensor data",
        "21 sensors and 3 settings per flight cycle, from NASA's engine simulator.",
    ),
    (
        ":material/insights:",
        "2 · Features",
        "Rolling averages smooth noise. 30-cycle trends capture slow wear.",
    ),
    (
        ":material/model_training:",
        "3 · Model",
        "LightGBM, scored with cross-validation grouped by engine to avoid leakage.",
    ),
    (
        ":material/api:",
        "4 · Serving",
        "A tested FastAPI service answers every prediction on this dashboard.",
    ),
]
for column, (icon, title, text) in zip(
    st.columns(len(steps), border=True), steps, strict=True
):
    with column:
        st.markdown(f"#### {icon} {title}")
        st.caption(text)

results = st.container()

# Slow work last: the page above is already visible while this loads
try:
    info = get_model_info()
    fleet = get_fleet()
except httpx.HTTPError:
    with api_notice:
        st.info(
            "Start the prediction API with `uvicorn src.api.main:app` to see live "
            "fleet numbers.",
            icon=":material/cloud_off:",
        )
    st.stop()

with kpi_row:
    st.metric(
        "Test error (RMSE)",
        f"{info['metrics']['test_rmse']:.1f} cycles",
        border=True,
        help="Average error on NASA's 100 unseen test engines, with labels capped "
        "at 125 cycles.",
    )
    st.metric("Engines monitored", f"{len(fleet)}", border=True)
    st.metric(
        "Need attention",
        f"{int((fleet['status'] != 'Healthy').sum())}",
        border=True,
        help="Engines predicted to have fewer than 60 cycles left.",
    )

with results:
    st.space("medium")
    st.subheader("Model results")
    m = info["metrics"]
    with st.container(horizontal=True):
        st.metric(
            "Cross-validated RMSE",
            f"{m['cv_rmse_mean']:.1f} ± {m['cv_rmse_std']:.1f}",
            border=True,
            help="Mean and spread across 5 folds of training engines.",
        )
        st.metric(
            "Near-failure RMSE",
            f"{m['cv_near_failure_rmse_mean']:.1f} cycles",
            border=True,
            help="Error on cycles with under 50 cycles left, where accuracy "
            "matters most.",
        )
        st.metric(
            "Test RMSE, raw labels",
            f"{m['test_rmse_raw']:.1f} cycles",
            border=True,
            help="Test error without capping the true RUL at 125.",
        )
        st.metric("Features", f"{info['n_features']}", border=True)
