"""Tests for the FastAPI prediction service."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMRegressor

import src.api.main as api
from src.data.loader import load_data
from src.features.build_features import prepare_features
from src.models.train import get_feature_columns, train_baseline_model


@pytest.fixture(scope="module")
def model() -> LGBMRegressor:
    """Train one model for the whole module, independent of models/ on disk."""
    train = prepare_features(load_data("train_FD001.txt"), is_train=True)
    cols = get_feature_columns(train)
    return train_baseline_model(train[cols], train["RUL"])


@pytest.fixture
def client(model: LGBMRegressor) -> Iterator[TestClient]:
    """Serve the app with the test model swapped in for the saved one."""
    api.app.dependency_overrides[api.get_model] = lambda: model
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def test_engines() -> pd.DataFrame:
    """Raw test-set telemetry for all 100 test engines."""
    return load_data("test_FD001.txt")


def engine_payload(
    test_engines: pd.DataFrame, engine_id: int, n_cycles: int | None = None
) -> dict:
    """Build a /predict request body from one test engine's history."""
    rows = test_engines[test_engines["engine_id"] == engine_id].drop(
        columns="engine_id"
    )
    if n_cycles is not None:
        rows = rows.iloc[:n_cycles]
    return {"cycles": rows.to_dict(orient="records")}


def test_health(client: TestClient) -> None:
    """Verify the health endpoint reports the service is up."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_matches_direct_model(
    client: TestClient, model: LGBMRegressor, test_engines: pd.DataFrame
) -> None:
    """Verify the API returns what the model predicts on the same features."""
    engine = test_engines[test_engines["engine_id"] == 31]
    latest = prepare_features(engine, is_train=False).iloc[[-1]]
    expected = round(float(model.predict(latest[model.feature_name_])[0]), 2)

    response = client.post("/predict", json=engine_payload(test_engines, 31))

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_rul"] == pytest.approx(expected)
    assert body["last_cycle"] == 196
    assert body["cycles_received"] == 196
    assert body["warning"] is None


def test_short_history_returns_warning(
    client: TestClient, test_engines: pd.DataFrame
) -> None:
    """Verify fewer cycles than the trend look-back still predicts, with a warning."""
    response = client.post("/predict", json=engine_payload(test_engines, 1, 10))

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["predicted_rul"] <= api.RUL_CAP
    assert "less reliable" in body["warning"]


def _text_sensor(payload: dict) -> None:
    payload["cycles"][1]["sensor_2"] = "hello"


def _missing_sensor(payload: dict) -> None:
    del payload["cycles"][0]["sensor_7"]


def _wrong_order(payload: dict) -> None:
    payload["cycles"].reverse()


def _no_cycles(payload: dict) -> None:
    payload["cycles"] = []


@pytest.mark.parametrize(
    "break_payload",
    [_text_sensor, _missing_sensor, _wrong_order, _no_cycles],
)
def test_invalid_requests_are_rejected(
    client: TestClient,
    test_engines: pd.DataFrame,
    break_payload: Callable[[dict], None],
) -> None:
    """Verify malformed histories get a 422 error instead of a prediction."""
    payload = engine_payload(test_engines, 1, 5)
    break_payload(payload)

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_model_file_returns_503(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, test_engines: pd.DataFrame
) -> None:
    """Verify a clear 503 error when no trained model has been saved."""
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    api.get_model.cache_clear()

    response = TestClient(api.app).post(
        "/predict", json=engine_payload(test_engines, 1, 5)
    )
    api.get_model.cache_clear()

    assert response.status_code == 503
    assert "Run `python -m src.models.train`" in response.json()["detail"]
