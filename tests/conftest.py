"""Shared pytest fixtures, available to every test file automatically."""

import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from src.data.loader import load_data
from src.features.build_features import prepare_features
from src.models.train import get_feature_columns, train_baseline_model


@pytest.fixture(scope="session")
def model() -> LGBMRegressor:
    """Train one model for the whole test run, independent of models/ on disk."""
    train = prepare_features(load_data("train_FD001.txt"), is_train=True)
    cols = get_feature_columns(train)
    return train_baseline_model(train[cols], train["RUL"])


@pytest.fixture(scope="session")
def test_engines() -> pd.DataFrame:
    """Raw test-set telemetry for all 100 test engines."""
    return load_data("test_FD001.txt")
