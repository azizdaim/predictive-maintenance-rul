"""Data drift: is new engine data still like the data the model learned from?

Each sensor is scored with the Population Stability Index (PSI). The training
readings are split into 10 buckets holding 10% each, and PSI measures how far
the new readings' shares of those buckets have moved:

    PSI = sum over buckets of (new share - training share) * ln(new / training)

Only the first 30 cycles of each engine are compared. Engines wear as they age,
and the test engines stop earlier in life than the training engines, which all
run to failure, so comparing every cycle mistakes wear for drift. When engines
are new, any difference comes from the fleet itself.

Run `python -m src.monitoring.drift` to measure the model's error on each
drifted NASA fleet and save it to reports/drift_results.csv.
"""

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.data.loader import DATASET_PATH, PROJECT_ROOT, load_data
from src.evaluation.metrics import evaluate_predictions
from src.features.build_features import prepare_features
from src.features.sensors import ENGINE_SECTIONS
from src.models.paths import MODEL_PATH

EARLY_LIFE_CYCLES = 30
BUCKETS = 10

# Common PSI rule of thumb: below 0.1 stable, 0.1 to 0.25 worth watching
WATCH_FROM = 0.1
DRIFTED_FROM = 0.25

# Every sensor that changes in FD001, with the engine section it measures.
# Sensor 6 is left out: 96% of its training readings are one value, so it
# cannot be split into buckets and its PSI would always be zero.
SENSOR_SECTIONS = {
    f"sensor_{number}": name
    for name, numbers in ENGINE_SECTIONS.values()
    for number in numbers
    if number != 6
}

# The other NASA scenarios, used as drifted fleets on the Drift check page
FLEETS = {
    "FD001": "Same fleet as training",
    "FD003": "A second fault type",
    "FD002": "Six operating conditions",
}

RESULTS_PATH = PROJECT_ROOT / "reports" / "drift_results.csv"


def early_life(readings: pd.DataFrame) -> pd.DataFrame:
    """Keep only the cycles from each engine's early life."""
    return readings[readings["cycle"] <= EARLY_LIFE_CYCLES]


def bucket_edges(reference: pd.Series) -> np.ndarray:
    """Split reference values into buckets of equal share, open at both ends."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, BUCKETS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def shares(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Return the share of values falling into each bucket."""
    return np.histogram(values, edges)[0] / len(values)


def psi(reference_shares: np.ndarray, new_shares: np.ndarray) -> float:
    """Compute the Population Stability Index between two bucket shares.

    Empty buckets get a tiny floor share, since ln(0) is undefined.
    """
    ref = np.clip(reference_shares, 1e-4, None)
    new = np.clip(new_shares, 1e-4, None)
    return float(np.sum((new - ref) * np.log(new / ref)))


def drift_status(score: float) -> str:
    """Label a PSI score as Stable, Watch, or Drifted."""
    if score >= DRIFTED_FROM:
        return "Drifted"
    if score >= WATCH_FROM:
        return "Watch"
    return "Stable"


@lru_cache
def reference_readings() -> pd.DataFrame:
    """Load the training engines' early-life readings once."""
    return early_life(load_data("train_FD001.txt"))


def compare_sensors(reference: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Score every sensor's drift between reference and new early-life readings.

    Args:
        reference: Early-life training readings.
        new: Early-life readings from the fleet being checked.

    Returns:
        pd.DataFrame: One row per sensor with its section, PSI, status, and
            the reference and new bucket shares, most drifted first.
    """
    rows = []
    for sensor, section in SENSOR_SECTIONS.items():
        edges = bucket_edges(reference[sensor])
        ref_shares = shares(reference[sensor], edges)
        new_shares = shares(new[sensor], edges)
        score = psi(ref_shares, new_shares)
        rows.append(
            {
                "sensor": sensor,
                "section": section,
                "psi": score,
                "status": drift_status(score),
                "reference_shares": ref_shares.tolist(),
                "new_shares": new_shares.tolist(),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def overall_status(sensor_statuses: pd.Series) -> str:
    """Report the worst status of any sensor."""
    for status in ["Drifted", "Watch"]:
        if (sensor_statuses == status).any():
            return status
    return "Stable"


def fleet_error(fleet: str) -> dict[str, float]:
    """Measure the served model's error on one NASA fleet's test engines.

    Uses each engine's last cycle and the fleet's answer file, the same way
    the FD001 test score is measured.
    """
    model = joblib.load(MODEL_PATH)
    featured = prepare_features(load_data(f"test_{fleet}.txt"), is_train=False)
    last = featured.groupby("engine_id").last().reset_index()
    truth = pd.read_csv(DATASET_PATH / f"RUL_{fleet}.txt", header=None)[0]
    preds = np.clip(model.predict(last[model.feature_name_]), 0, 125)
    rmse = evaluate_predictions(truth.clip(upper=125), preds)["rmse"]
    return {"fleet": fleet, "engines": len(truth), "rmse": round(rmse, 2)}


def main() -> None:
    """Save the model's error on each fleet for the dashboard to show."""
    results = pd.DataFrame([fleet_error(fleet) for fleet in FLEETS])
    results.to_csv(RESULTS_PATH, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
