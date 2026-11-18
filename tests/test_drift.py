"""Tests for drift detection and the /drift endpoint."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.api.main as api
from src.monitoring.drift import (
    bucket_edges,
    compare_sensors,
    drift_status,
    early_life,
    psi,
    reference_readings,
    shares,
)


def fleet_payload(engines: pd.DataFrame, n_cycles: int = 30) -> dict:
    """Build a /drift request from the first cycles of every engine."""
    return {
        "engines": [
            {"cycles": rows.drop(columns="engine_id").head(n_cycles).to_dict("records")}
            for _, rows in engines.groupby("engine_id")
        ]
    }


def test_psi_is_zero_for_identical_shares() -> None:
    """Verify unchanged data scores no drift."""
    same = np.full(10, 0.1)

    assert psi(same, same) == pytest.approx(0.0)


def test_psi_grows_as_data_shifts() -> None:
    """Verify a bigger shift in the readings gives a bigger PSI."""
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(size=5000))
    edges = bucket_edges(reference)
    ref_shares = shares(reference, edges)

    small = psi(ref_shares, shares(reference + 0.2, edges))
    large = psi(ref_shares, shares(reference + 1.0, edges))

    assert ref_shares == pytest.approx(np.full(10, 0.1), abs=0.01)
    assert 0 < small < large


@pytest.mark.parametrize(
    ("score", "status"),
    [(0.05, "Stable"), (0.1, "Watch"), (0.2, "Watch"), (0.25, "Drifted")],
)
def test_drift_status_thresholds(score: float, status: str) -> None:
    """Verify the PSI rule of thumb: 0.1 to watch, 0.25 to call drift."""
    assert drift_status(score) == status


def test_early_life_keeps_first_30_cycles() -> None:
    """Verify only each engine's first 30 cycles are compared."""
    readings = pd.DataFrame({"cycle": [1, 30, 31, 200]})

    assert early_life(readings)["cycle"].tolist() == [1, 30]


def test_same_fleet_is_stable(test_engines: pd.DataFrame) -> None:
    """Verify FD001 test engines raise no false alarm against FD001 training."""
    report = compare_sensors(reference_readings(), early_life(test_engines))

    assert (report["status"] == "Stable").all()


def test_drift_endpoint_flags_shifted_sensors(test_engines: pd.DataFrame) -> None:
    """Verify the API calls the same fleet stable and a shifted one drifted."""
    client = TestClient(api.app)

    same = client.post("/drift", json=fleet_payload(test_engines))
    shifted_engines = test_engines.assign(sensor_11=test_engines["sensor_11"] + 0.5)
    shifted = client.post("/drift", json=fleet_payload(shifted_engines))

    assert same.status_code == 200
    assert same.json()["status"] == "Stable"
    assert same.json()["readings_compared"] == 100 * 30
    body = shifted.json()
    assert body["status"] == "Drifted"
    assert body["sensors"][0]["sensor"] == "sensor_11"
    assert body["drifted_sensors"] == 1


def test_drift_endpoint_needs_early_life_readings(test_engines: pd.DataFrame) -> None:
    """Verify a batch with no early-life cycles is rejected with a reason."""
    late = test_engines[test_engines["cycle"] > 30]

    response = TestClient(api.app).post("/drift", json=fleet_payload(late))

    assert response.status_code == 422
    assert "first 30 cycles" in response.json()["detail"]
