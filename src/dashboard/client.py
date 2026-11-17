"""Data and API helpers for the dashboard, kept separate so they can be tested."""

import os

import httpx
import pandas as pd

from src.dashboard.palette import AMBER, GREEN, RED
from src.data.loader import DATASET_PATH, load_data

# Where the FastAPI service runs; override with the RUL_API_URL environment variable
API_URL = os.environ.get("RUL_API_URL", "http://127.0.0.1:8000")

# Same cap as the training target, so true and predicted RUL are comparable
RUL_CAP = 125

# Health bands in cycles. Late predictions are the costly mistake, so the
# bands are deliberately generous: an engine is flagged well before failure.
REPLACE_SOON_BELOW = 30
WATCH_BELOW = 60

# Label colors, most urgent first, so tables and badges stay consistent
STATUS_COLORS = {"Replace soon": "red", "Watch": "orange", "Healthy": "green"}

# The same colors as hex codes, for custom visuals: cockpit red, amber, and green
STATUS_HEX = {"Replace soon": RED, "Watch": AMBER, "Healthy": GREEN}


def load_test_engines() -> pd.DataFrame:
    """Load raw telemetry for the 100 FD001 test engines."""
    return load_data("test_FD001.txt")


def true_rul_by_cycle(engines: pd.DataFrame) -> pd.DataFrame:
    """Compute the true RUL at every cycle of every test engine, capped at 125.

    RUL_FD001.txt gives the true RUL at each engine's last recorded cycle, so
    the RUL at an earlier cycle is that value plus the cycles still to come.
    """
    final = pd.read_csv(DATASET_PATH / "RUL_FD001.txt", header=None, names=["final"])
    final["engine_id"] = range(1, len(final) + 1)
    last_cycle = engines.groupby("engine_id")["cycle"].max().rename("last_cycle")

    out = engines[["engine_id", "cycle"]].merge(final, on="engine_id")
    out = out.merge(last_cycle, on="engine_id")
    out["true_rul"] = (out["final"] + out["last_cycle"] - out["cycle"]).clip(
        upper=RUL_CAP
    )
    return out[["engine_id", "cycle", "true_rul"]]


def fetch_history_predictions(
    engine_rows: pd.DataFrame, api_url: str = API_URL
) -> pd.DataFrame:
    """Ask the API for a predicted RUL at every cycle of one engine.

    Args:
        engine_rows: Raw telemetry for a single engine, oldest cycle first.
        api_url: Base URL of the FastAPI service.

    Returns:
        pd.DataFrame: Columns 'cycle' and 'predicted_rul'.

    Raises:
        httpx.HTTPError: If the API is unreachable or rejects the request.
    """
    payload = {
        "cycles": engine_rows.drop(columns="engine_id").to_dict(orient="records")
    }
    response = httpx.post(f"{api_url}/predict/history", json=payload, timeout=30.0)
    response.raise_for_status()
    return pd.DataFrame(response.json()["predictions"])


def fetch_model_info(api_url: str = API_URL) -> dict:
    """Ask the API how the served model scored when it was trained.

    Raises:
        httpx.HTTPError: If the API is unreachable or has no saved metrics.
    """
    response = httpx.get(f"{api_url}/model/info", timeout=10.0)
    response.raise_for_status()
    return response.json()


def health_status(predicted_rul: float) -> tuple[str, str]:
    """Map a predicted RUL to a maintenance label and a badge color."""
    if predicted_rul < REPLACE_SOON_BELOW:
        label = "Replace soon"
    elif predicted_rul < WATCH_BELOW:
        label = "Watch"
    else:
        label = "Healthy"
    return label, STATUS_COLORS[label]


def fleet_table(
    predictions: dict[int, pd.DataFrame], trend_cycles: int = 50
) -> pd.DataFrame:
    """Summarize every engine's latest prediction, most urgent first.

    Args:
        predictions: Per-engine frames with 'cycle' and 'predicted_rul' columns.
        trend_cycles: How many recent predictions to keep for the trend line.

    Returns:
        pd.DataFrame: One row per engine, sorted by predicted RUL ascending.
    """
    rows = []
    for engine_id, engine in predictions.items():
        latest = float(engine["predicted_rul"].iloc[-1])
        rows.append(
            {
                "engine_id": engine_id,
                "status": health_status(latest)[0],
                "predicted_rul": latest,
                "cycles": int(engine["cycle"].iloc[-1]),
                "trend": engine["predicted_rul"].tail(trend_cycles).tolist(),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("predicted_rul", ignore_index=True)
