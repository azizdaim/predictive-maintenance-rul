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
    cycles_text,
    fetch_history_predictions,
    first_alerts,
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
    """Verify the landing page shows the error, fleet status, and urgent engine."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()

    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "within about 13 flights" in text  # test RMSE 13.22, rounded
    assert "of 100 monitored engines need replacing soon" in text
    # The hero engine is the most urgent one, with a button to open it
    assert any(b.label.startswith("Inspect engine ") for b in at.button)


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
    assert at.header[0].value.startswith("Engine 1 ")


@pytest.mark.usefixtures("api_through_httpx")
def test_simulation_replays_engine_and_raises_alerts() -> None:
    """Verify play advances the replay, alerts fire, and it stops at the end."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.switch_page("app_pages/simulate.py").run()
    assert at.header[0].value == "Engine 31, cycle 0 of 196"

    at.segmented_control(key="sim_speed").set_value("Very fast")
    at.button(key="sim_play").click().run()
    assert not at.exception
    assert at.header[0].value == "Engine 31, cycle 10 of 196"

    # The timer does not run in tests, so each rerun stands in for one tick
    for _ in range(25):
        at.run()
    assert at.header[0].value == "Engine 31, cycle 196 of 196"
    assert at.button(key="sim_play").label == "Replay"
    # Each alert says how much life was really left: the replay's payoff
    alerts = [m.value for m in at.markdown if "before failure" in m.value]
    assert [a.split(": ")[1].split(",")[0] for a in alerts[:2]] == [
        "watch",
        "replace soon",
    ]
    assert any(m.value.startswith("**Engine 31 was flagged") for m in at.markdown)

    at.button(key="sim_reset").click().run()
    assert at.header[0].value == "Engine 31, cycle 0 of 196"


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


@pytest.mark.parametrize(
    ("embedded", "expected"),
    [("1", "temporarily unavailable"), ("0", "uvicorn src.api.main:app")],
)
def test_api_message_suits_visitors_on_the_hosted_site(
    monkeypatch: pytest.MonkeyPatch, embedded: str, expected: str
) -> None:
    """Verify visitors get a plain message and developers get the start command."""
    from src.dashboard.data import api_unavailable_message

    monkeypatch.setenv("RUL_EMBEDDED_API", embedded)

    message = api_unavailable_message()

    assert expected in message
    if embedded == "1":
        assert "uvicorn" not in message


def test_cycles_text_uses_singular_for_one() -> None:
    """Verify "1 cycle" and "+1 cycle", but "8 cycles" and "-3 cycles"."""
    assert cycles_text(1) == "1 cycle"
    assert cycles_text(8.4) == "8 cycles"
    assert cycles_text(1, signed=True) == "+1 cycle"
    assert cycles_text(-3, signed=True) == "-3 cycles"


def test_first_alerts_follow_the_replay_steps() -> None:
    """Verify alerts fire at the first replay step that reaches a worse band."""
    cycles = list(range(1, 21))
    # Predicted life falls 10 per cycle from 200: watch below 60 at cycle 16,
    # replace soon below 30 at cycle 19
    predictions = pd.DataFrame(
        {"cycle": cycles, "predicted_rul": [210 - 10 * c for c in cycles]}
    )
    true_rul = pd.DataFrame({"cycle": cycles, "true_rul": [40 - c for c in cycles]})

    alerts = first_alerts(predictions, true_rul, step=5)

    # Steps land on cycles 5, 10, 15, 20: watch is first seen at 20, together
    # with replace soon, so only the most serious alert fires
    assert alerts == {"Replace soon": (20, 20.0)}
    alerts = first_alerts(predictions, true_rul, step=2)
    assert alerts == {"Watch": (16, 24.0), "Replace soon": (20, 20.0)}


@pytest.mark.usefixtures("api_through_httpx")
def test_engine_page_opens_the_engine_named_in_the_address() -> None:
    """Verify ?engine=34 opens engine 34, and picking another updates the address."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.query_params["engine"] = "34"
    at.run()
    at.switch_page("app_pages/engine.py").run()

    assert not at.exception
    assert at.header[0].value.startswith("Engine 34 ")

    at.selectbox(key="engine_id").set_value(7).run()
    assert at.header[0].value.startswith("Engine 7 ")
    assert at.query_params["engine"] == "7"


@pytest.mark.usefixtures("api_through_httpx")
def test_drift_page_compares_fleets() -> None:
    """Verify the training fleet reads stable and a new-conditions fleet drifted."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.switch_page("app_pages/drift.py").run()

    assert not at.exception
    assert at.metric[0].value == "0 of 14"
    assert any("Its predictions here can be trusted" in m.value for m in at.markdown)

    at.segmented_control(key="drift_fleet").set_value("FD002").run()

    assert not at.exception
    assert at.metric[0].value == "14 of 14"
    verdict = next(m.value for m in at.markdown if m.value.startswith("All 14"))
    assert "six operating conditions" in verdict
    assert "3.4 times higher" in verdict
    assert at.table[0].value.shape == (14, 4)


def test_drift_page_explains_when_api_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the drift page names itself in the error and keeps the fleet picker."""

    def refuse(url: str, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", refuse)
    monkeypatch.setattr(httpx, "get", refuse)
    st.cache_data.clear()

    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.switch_page("app_pages/drift.py").run()

    assert not at.exception
    assert at.error[0].proto.title == "Drift check unavailable"
    assert at.segmented_control(key="drift_fleet").value == "FD001"
