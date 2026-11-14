"""Tests for the ablation study."""

import pandas as pd
import pytest

import src.models.ablation as ablation
from src.data.loader import load_data
from src.features.build_features import prepare_features
from src.models.train import get_feature_columns


@pytest.fixture(scope="module")
def columns() -> list[str]:
    """Feature columns of the real training pipeline."""
    df = prepare_features(load_data("train_FD001.txt"), is_train=True)
    return get_feature_columns(df)


def test_feature_type_groups_cover_every_feature_once(columns: list[str]) -> None:
    """Verify each feature lands in exactly one feature-type group."""
    groups = ablation.feature_type_groups(columns)

    assigned = [col for cols in groups.values() for col in cols]
    assert sorted(assigned) == sorted(columns)
    assert len(groups["30-cycle trends"]) == 15
    assert groups["Operating settings"] == ["setting_1", "setting_2"]


def test_engine_section_groups_cover_every_sensor_feature(columns: list[str]) -> None:
    """Verify every sensor-derived feature maps to one engine section."""
    groups = ablation.engine_section_groups(columns)

    assigned = [col for cols in groups.values() for col in cols]
    sensor_features = [col for col in columns if col.startswith("sensor_")]
    assert sorted(assigned) == sorted(sensor_features)
    # Each sensor contributes 4 features: raw, rolling mean, rolling std, trend
    assert len(groups["High-pressure turbine"]) == 4
    assert len(groups["Fan"]) == 16


def test_run_ablation_compares_each_variant_to_the_full_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify variants drop the right columns and changes are measured per shuffle."""
    df = pd.DataFrame(
        {
            "engine_id": [1],
            "cycle": [1],
            "RUL": [100],
            "a": [0.0],
            "b": [0.0],
            "c": [0.0],
        }
    )
    seen: list[list[str]] = []

    def fake_cv(reduced: pd.DataFrame, seed: int) -> dict[str, float]:
        features = get_feature_columns(reduced)
        seen.append(features)
        # Fewer features means a higher error, plus a little noise per seed
        rmse = 10.0 + (3 - len(features)) * 2.0 + seed * 0.1
        return {
            "cv_rmse_mean": rmse,
            "cv_near_failure_rmse_mean": rmse / 2,
            "cv_nasa_per_engine_mean": 1.0,
        }

    monkeypatch.setattr(ablation, "cross_validate_by_engine", fake_cv)

    results = ablation.run_ablation(
        df, {"Study": {"A": ["a"], "B and C": ["b", "c"]}}, seeds=(0, 1)
    ).set_index("variant")

    assert ["a", "b", "c"] in seen
    assert ["b", "c"] in seen and ["a"] in seen
    assert results.loc[ablation.FULL_MODEL, "rmse_change"] == 0
    assert results.loc["Without a", "rmse_change"] == pytest.approx(2.0)
    assert results.loc["Without b and c", "rmse_change"] == pytest.approx(4.0)
    assert results.loc["Without b and c", "worse_in_shuffles"] == 2
