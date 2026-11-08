"""Unit tests for model training and evaluation module."""

from src.data.loader import load_data
from src.features.build_features import prepare_features
from src.models.train import (
    evaluate_on_test_set,
    get_feature_columns,
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

