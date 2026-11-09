"""Unit tests for model training and evaluation module."""

from pathlib import Path

from mlflow.tracking import MlflowClient

from src.data.loader import load_data
from src.features.build_features import prepare_features
from src.models.train import (
    EXPERIMENT_NAME,
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
    assert len(feature_cols) == 47


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


def test_run_experiment_logs_comparable_metrics(tmp_path: Path) -> None:
    """Verify a run is logged and NASA scores are normalized per engine."""
    tracking_uri = tmp_path.as_uri()

    metrics = run_experiment(tracking_uri=tracking_uri)

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
    # A sum over every validation row would be in the tens of thousands.
    assert metrics["val_nasa_per_engine"] < 100

