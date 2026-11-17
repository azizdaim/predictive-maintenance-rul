"""Tests for running the prediction API inside the dashboard's process."""

import socket
from collections.abc import Iterator

import httpx
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

import src.api.main as api
from src.dashboard.client import fetch_history_predictions
from src.dashboard.embedded_api import embedded_api_enabled, start_embedded_api


@pytest.fixture
def free_port() -> int:
    """Ask the operating system for a port nothing is listening on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def served_model(model: LGBMRegressor) -> Iterator[None]:
    """Serve the test model instead of loading the saved one from disk."""
    api.app.dependency_overrides[api.get_model] = lambda: model
    yield
    api.app.dependency_overrides.clear()


@pytest.mark.usefixtures("served_model")
def test_embedded_api_answers_over_http(
    free_port: int, test_engines: pd.DataFrame
) -> None:
    """Verify the in-process server handles real HTTP requests, then stops."""
    server = start_embedded_api(port=free_port)
    url = f"http://127.0.0.1:{free_port}"
    try:
        assert httpx.get(f"{url}/health").json() == {"status": "ok"}

        engine = test_engines[test_engines["engine_id"] == 31]
        predictions = fetch_history_predictions(engine, api_url=url)
        assert len(predictions) == len(engine)
    finally:
        server.should_exit = True


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("0", False)])
def test_embedded_api_is_opt_in(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    """Verify only RUL_EMBEDDED_API=1 turns the in-process API on."""
    monkeypatch.setenv("RUL_EMBEDDED_API", value)
    assert embedded_api_enabled() is expected


def test_embedded_api_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a normal run keeps using the separate API server."""
    monkeypatch.delenv("RUL_EMBEDDED_API", raising=False)
    assert embedded_api_enabled() is False
