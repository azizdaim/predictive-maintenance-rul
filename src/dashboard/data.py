"""Cached data access shared by the dashboard pages."""

import pandas as pd
import streamlit as st

from src.dashboard.client import (
    API_URL,
    fetch_drift,
    fetch_history_predictions,
    fetch_model_info,
    fleet_table,
    load_test_engines,
    true_rul_by_cycle,
)
from src.dashboard.embedded_api import embedded_api_enabled
from src.data.loader import PROJECT_ROOT, load_data
from src.monitoring.drift import RESULTS_PATH

# Written by python -m src.models.ablation and committed with the code
ABLATION_REPORT = PROJECT_ROOT / "reports" / "ablation_results.csv"


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


def api_unavailable_message() -> str:
    """Explain a missing prediction service to whoever is looking.

    On the hosted site, where the API runs inside the app, visitors cannot fix
    anything, so they get a plain message. Locally, the developer gets the
    command that starts the API.
    """
    if embedded_api_enabled():
        return "Live predictions are temporarily unavailable. Try again in a minute."
    return (
        f"The dashboard gets its predictions from the FastAPI service at {API_URL}. "
        "Start it in another terminal with `uvicorn src.api.main:app`, then refresh."
    )


def show_api_error() -> None:
    """Explain that predictions are unavailable, then stop the page."""
    st.error(
        api_unavailable_message(),
        title="Predictions unavailable",
        icon=":material/cloud_off:",
    )
    st.stop()


@st.cache_data
def get_ablation_results() -> pd.DataFrame:
    """Load the saved ablation study results."""
    return pd.read_csv(ABLATION_REPORT)


@st.cache_data(ttl="10m", show_spinner="Comparing sensors with the training engines...")
def get_drift(fleet: str) -> dict:
    """Check one NASA fleet's test engines for drift through the API."""
    return fetch_drift(load_data(f"test_{fleet}.txt"))


@st.cache_data
def get_drift_results() -> pd.DataFrame:
    """Load the model's saved error on each fleet, indexed by fleet."""
    return pd.read_csv(RESULTS_PATH, index_col="fleet")
