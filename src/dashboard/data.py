"""Cached data access shared by the dashboard pages."""

import pandas as pd
import streamlit as st

from src.dashboard.client import (
    API_URL,
    fetch_history_predictions,
    fetch_model_info,
    fleet_table,
    load_test_engines,
    true_rul_by_cycle,
)


@st.cache_data(ttl="10m", show_spinner=False)
def get_model_info() -> dict:
    """Fetch the served model's saved evaluation results."""
    return fetch_model_info()


@st.cache_data
def get_test_engines() -> pd.DataFrame:
    """Load the test engines once and share them across sessions."""
    return load_test_engines()


@st.cache_data
def get_true_rul() -> pd.DataFrame:
    """Compute true RUL for every test engine once."""
    return true_rul_by_cycle(get_test_engines())


@st.cache_data(ttl="10m", show_spinner=False)
def get_predictions(engine_id: int) -> pd.DataFrame:
    """Fetch one engine's predictions, reused while the user explores it."""
    engines = get_test_engines()
    return fetch_history_predictions(engines[engines["engine_id"] == engine_id])


@st.cache_data(ttl="10m", show_spinner="Scoring all 100 engines...")
def get_fleet() -> pd.DataFrame:
    """Score every test engine through the API and summarize the fleet."""
    engine_ids = get_test_engines()["engine_id"].unique()
    return fleet_table({int(e): get_predictions(int(e)) for e in engine_ids})


def show_api_error() -> None:
    """Explain how to start the API, then stop the page."""
    st.error(
        f"The dashboard gets its predictions from the FastAPI service at {API_URL}. "
        "Start it in another terminal with `uvicorn src.api.main:app`, then refresh.",
        title="Can't reach the prediction API",
        icon=":material/cloud_off:",
    )
    st.stop()
