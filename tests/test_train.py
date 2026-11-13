"""Unit tests for model training and evaluation module."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.data.loader import load_data
from src.features.build_features import prepare_features
from src.models.train import (
    EXPERIMENT_NAME,
    cross_validate_by_engine,
    evaluate_on_test_set,
    get_feature_columns,
    run_experiment,
    sample_one_row_per_engine,
    split_by_engine,
    train_baseline_model,
)


def test_split_by_engine_isolation() -> None:
    """Verify that engine split has zero overlap between train and val."""
    df_raw = load_data("train_FD001.txt")
    df_prepared = prepare_features(df_raw, is_train=True)

    train_df, val_df = split_by_engine(df_prepared, train_ratio=0.8)

    train_engines = set(train_df["engine_id"].unique())
    val_engines = set(val_df["engine_id"].unique())

    # Assert exactly 80 train and 20 val engines
    assert len(train_engines) == 80
    assert len(val_engines) == 20

    # Critical: ZERO overlap between train and val engines
    assert train_engines.intersection(val_engines) == set()


def test_get_feature_columns_excludes_metadata() -> None:
    """Verify metadata columns (engine_id, cycle, RUL) are excluded from features."""
    df_raw = load_data("train_FD001.txt")
    df_prepared = prepare_features(df_raw, is_train=True)

    feature_cols = get_feature_columns(df_prepared)

    assert "engine_id" not in feature_cols
    assert "cycle" not in feature_cols
    assert "RUL" not in feature_cols

    # Active sensors and rolling features must be included
    assert "sensor_2" in feature_cols
    assert "sensor_2_roll_mean_5" in feature_cols
    assert "sensor_2_trend_30" in feature_cols
    assert len(feature_cols) == 62


def test_baseline_model_training_and_evaluation() -> None:
    """Verify model fits cleanly and produces realistic benchmark scores."""
    df_raw = load_data("train_FD001.txt")
    df_prepared = prepare_features(df_raw, is_train=True)

    train_df, _ = split_by_engine(df_prepared, train_ratio=0.8)
    feature_cols = get_feature_columns(train_df)

    x_train, y_train = train_df[feature_cols], train_df["RUL"]
    model = train_baseline_model(x_train, y_train)

    # Evaluate on the official NASA test set
    test_metrics = evaluate_on_test_set(model, window=5, clip_threshold=125)

    assert "rmse" in test_metrics
    assert "nasa_score" in test_metrics

    # Test RMSE should comfortably be under 25 cycles for FD001
    assert test_metrics["rmse"] < 25.0
    assert test_metrics["nasa_score"] > 0.0


def test_sample_one_row_per_engine_is_reproducible() -> None:
    """Verify sampling returns one row per engine and honours the seed."""
    df = prepare_features(load_data("train_FD001.txt"), is_train=True)

    first = sample_one_row_per_engine(df, seed=7)
    repeat = sample_one_row_per_engine(df, seed=7)
    other = sample_one_row_per_engine(df, seed=8)

    assert sorted(first["engine_id"]) == sorted(df["engine_id"].unique())
    assert first["cycle"].tolist() == repeat["cycle"].tolist()
    assert first["cycle"].tolist() != other["cycle"].tolist()


def test_cross_validation_keeps_engines_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify no fold shares engines and every engine is validated once."""
    df = prepare_features(load_data("train_FD001.txt"), is_train=True)
    train_folds: list[set[int]] = []
    val_folds: list[set[int]] = []
    predicted_rows: list[int] = []

    def fake_predict(x: pd.DataFrame) -> np.ndarray:
        predicted_rows.append(len(x))
        return np.zeros(len(x))

    def fake_train(x_train: pd.DataFrame, y_train: pd.Series) -> SimpleNamespace:
        train_folds.append(set(df.loc[x_train.index, "engine_id"]))
        return SimpleNamespace(predict=fake_predict)

    def fake_evaluate(
        model: SimpleNamespace, val_df: pd.DataFrame, feature_cols: list[str]
    ) -> dict[str, float]:
        val_folds.append(set(val_df["engine_id"]))
        return {"rmse": 1.0, "nasa_per_engine": 2.0}

    monkeypatch.setattr("src.models.train.train_baseline_model", fake_train)
    monkeypatch.setattr("src.models.train.evaluate_per_engine", fake_evaluate)

    metrics = cross_validate_by_engine(df, n_splits=5)

    assert len(val_folds) == 5
    for train_engines, val_engines in zip(train_folds, val_folds, strict=True):
        assert len(train_engines) == 80
        assert len(val_engines) == 20
        assert train_engines.isdisjoint(val_engines)

    assert set().union(*val_folds) == set(df["engine_id"])
    assert metrics["cv_rmse_mean"] == 1.0
    assert metrics["cv_rmse_std"] == 0.0
    assert metrics["cv_nasa_per_engine_mean"] == 2.0

    # Every engine is validated once, so each near-failure row is scored once.
    assert sum(predicted_rows) == (df["RUL"] < 50).sum()
    assert metrics["cv_near_failure_rmse_mean"] > 0.0


def test_run_experiment_logs_cv_and_final_model(tmp_path: Path) -> None:
    """Verify a run logs cross-validation metrics and the final model."""
    tracking_uri = tmp_path.as_uri()

    model_path = tmp_path / "model.joblib"

    metrics = run_experiment(tracking_uri=tracking_uri, model_path=model_path)

    assert model_path.exists()

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs([experiment.experiment_id])

    assert len(runs) == 1
    assert runs[0].data.metrics == metrics
    assert runs[0].data.params["window"] == "5"
    assert client.list_artifacts(runs[0].info.run_id, "model")

    # Raw labels keep the 11 engines above the cap, so RMSE must differ.
    assert metrics["test_rmse_raw"] != metrics["test_rmse"]
    assert metrics["test_nasa_per_engine"] == metrics["test_nasa_score"] / 100
    assert metrics["cv_rmse_std"] > 0.0
    # A sum over every validation row would be in the tens of thousands.
    assert metrics["cv_nasa_per_engine_mean"] < 100

