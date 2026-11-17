"""Tests for the dashboard helpers and the Streamlit page."""

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pandas as pd
import pytest
import streamlit as st
from fastapi.testclient import TestClient
from lightgbm import LGBMRegressor
from streamlit.testing.v1 import AppTest

import src.api.main as api
from src.dashboard.client import (
    API_URL,
    fetch_history_predictions,
    fleet_table,
    health_status,
    true_rul_by_cycle,
)

APP_PATH = "../src/dashboard/app.py"


SAVED_METRICS = {
    "metrics": {
        "cv_rmse_mean": 11.97,
        "cv_rmse_std": 0.54,
        "cv_near_failure_rmse_mean": 6.87,
        "test_rmse": 13.22,
        "test_rmse_raw": 14.49,
    },
    "n_features": 30,
    "n_train_engines": 100,
    "trained_at": "2026-11-14T00:00:00+00:00",
}


@pytest.fixture
def api_through_httpx(
    model: LGBMRegressor, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    """Route the dashboard's HTTP calls to the real API app, in-process."""
    api.app.dependency_overrides[api.get_model] = lambda: model
    (tmp_path / "rul_model_metrics.json").write_text(json.dumps(SAVED_METRICS))
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "rul_model.joblib")
    test_client = TestClient(api.app)

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        return test_client.post(url.removeprefix(API_URL), **kwargs)

    def fake_get(url: str, **kwargs: object) -> httpx.Response:
        return test_client.get(url.removeprefix(API_URL), **kwargs)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    st.cache_data.clear()
    yield
    st.cache_data.clear()
    api.app.dependency_overrides.clear()


def test_true_rul_counts_back_from_the_answer_file(test_engines: pd.DataFrame) -> None:
    """Verify true RUL is the final answer plus the cycles still to come, capped."""
    rul = true_rul_by_cycle(test_engines).set_index(["engine_id", "cycle"])["true_rul"]

    # Engine 31 has 196 cycles and a true RUL of 8 at its last one
    assert rul[(31, 196)] == 8
    assert rul[(31, 150)] == 8 + 46
    # Early cycles are capped at 125, like the training target
    assert rul[(31, 1)] == 125


@pytest.mark.parametrize(
    ("predicted", "label"),
    [(5, "Replace soon"), (29.9, "Replace soon"), (30, "Watch"), (60, "Healthy")],
)
def test_health_status_bands(predicted: float, label: str) -> None:
    """Verify the band edges map to the expected maintenance labels."""
    assert health_status(predicted)[0] == label


@pytest.mark.usefixtures("api_through_httpx")
def test_fetch_history_predictions_returns_every_cycle(
    test_engines: pd.DataFrame,
) -> None:
    """Verify the client turns the API reply into one row per cycle."""
    engine = test_engines[test_engines["engine_id"] == 31]

    predictions = fetch_history_predictions(engine)

    assert list(predictions.columns) == ["cycle", "predicted_rul"]
    assert len(predictions) == 196


def test_fleet_table_ranks_most_urgent_first() -> None:
    """Verify the fleet summary sorts by latest prediction and labels each engine."""
    predictions = {
        1: pd.DataFrame({"cycle": [1, 2, 3], "predicted_rul": [120.0, 90.0, 70.0]}),
        2: pd.DataFrame({"cycle": [1, 2], "predicted_rul": [40.0, 12.0]}),
    }

    table = fleet_table(predictions, trend_cycles=2)

    assert table["engine_id"].tolist() == [2, 1]
    assert table["status"].tolist() == ["Replace soon", "Healthy"]
    assert table["cycles"].tolist() == [2, 3]
    assert table.loc[1, "trend"] == [90.0, 70.0]


@pytest.mark.usefixtures("api_through_httpx")
def test_home_page_shows_live_results() -> None:
    """Verify the landing page shows saved model results and fleet numbers."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()

    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Test error (RMSE)"] == "13.2 cycles"
    assert metrics["Engines monitored"] == "100"
    assert metrics["Cross-validated RMSE"] == "12.0 ± 0.5"


@pytest.mark.usefixtures("api_through_httpx")
def test_fleet_page_summarizes_all_engines() -> None:
    """Verify the fleet page scores all 100 engines and counts the bands."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.switch_page("app_pages/fleet.py").run()

    assert not at.exception
    assert len(at.dataframe[0].value) == 100
    total = sum(int(m.value.split()[0]) for m in at.metric)
    assert total == 100


@pytest.mark.usefixtures("api_through_httpx")
def test_engine_page_shows_engine_and_updates() -> None:
    """Verify the engine page renders metrics and reacts to engine and cycle."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.switch_page("app_pages/engine.py").run()

    assert not at.exception
    assert at.metric[0].value == "8 cycles"  # engine 31's true RUL at its last cycle

    at.slider[0].set_value(10).run()
    assert not at.exception
    assert len(at.warning) == 1  # short-history warning before cycle 31

    at.selectbox[0].set_value(1).run()
    assert not at.exception
    assert at.subheader[0].value.startswith("Engine 1 ")


@pytest.mark.usefixtures("api_through_httpx")
def test_simulation_replays_engine_and_raises_alerts() -> None:
    """Verify play advances the replay, alerts fire, and it stops at the end."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.switch_page("app_pages/simulate.py").run()
    assert at.subheader[0].value == "Engine 31, cycle 0 of 196"

    at.segmented_control(key="sim_speed").set_value("Very fast")
    at.button(key="sim_play").click().run()
    assert not at.exception
    assert at.subheader[0].value == "Engine 31, cycle 10 of 196"

    # The timer does not run in tests, so each rerun stands in for one tick
    for _ in range(25):
        at.run()
    assert at.subheader[0].value == "Engine 31, cycle 196 of 196"
    assert at.button(key="sim_play").label == "Play"
    alerts = [m.value for m in at.markdown if "entered" in m.value]
    assert [a.split(": entered ")[1] for a in alerts] == ["watch", "replace soon"]

    at.button(key="sim_reset").click().run()
    assert at.subheader[0].value == "Engine 31, cycle 0 of 196"


@pytest.mark.usefixtures("api_through_httpx")
def test_how_it_works_page_shows_live_scores_and_limits() -> None:
    """Verify the page fills its model note from the API and lists limits."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.switch_page("app_pages/about.py").run()

    assert not at.exception
    headers = [h.value for h in at.header]
    assert headers == ["What the experiments showed", "Limitations"]


def test_dashboard_explains_when_api_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a clear error instead of a crash when the API is unreachable."""

    def refuse(url: str, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", refuse)
    monkeypatch.setattr(httpx, "get", refuse)
    st.cache_data.clear()

    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    # The landing page still renders, with a hint instead of live numbers
    assert not at.exception
    assert "uvicorn src.api.main:app" in at.info[0].value

    at.switch_page("app_pages/fleet.py").run()
    assert not at.exception
    assert len(at.error) == 1
    assert "uvicorn src.api.main:app" in at.error[0].value
