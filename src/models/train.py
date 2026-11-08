"""Model training and evaluation pipeline for turbofan RUL prediction."""

import pandas as pd
from lightgbm import LGBMRegressor

from src.data.loader import DATASET_PATH, load_data
from src.evaluation.metrics import evaluate_predictions
from src.features.build_features import prepare_features

# Columns that should NOT be used as model input features
METADATA_COLS = ["engine_id", "cycle", "RUL"]


def split_by_engine(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by engine_id to prevent time-series data leakage.

    Args:
        df: Input DataFrame containing 'engine_id'.
        train_ratio: Fraction of engines to allocate to train. Default 0.8.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df)
    """
    engines = df["engine_id"].unique()
    n_train = int(len(engines) * train_ratio)

    train_engines = engines[:n_train]
    val_engines = engines[n_train:]

    train_df = df[df["engine_id"].isin(train_engines)].copy()
    val_df = df[df["engine_id"].isin(val_engines)].copy()

    return train_df, val_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature column names, excluding metadata and target."""
    return [c for c in df.columns if c not in METADATA_COLS]


def train_baseline_model(
    x_train: pd.DataFrame, y_train: pd.Series
) -> LGBMRegressor:
    """Train baseline LightGBM regressor.

    Args:
        x_train: Training feature matrix.
        y_train: Training target (RUL).

    Returns:
        LGBMRegressor: Fitted LightGBM model.
    """
    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_on_test_set(
    model: LGBMRegressor, window: int = 5, clip_threshold: int = 125
) -> dict[str, float]:
    """Evaluate trained model on the official NASA C-MAPSS test set.

    In the NASA test set, we take the last recorded cycle of each engine
    and compare predictions against the ground-truth RUL in RUL_FD001.txt.

    Args:
        model: Trained LightGBM model.
        window: Rolling window used during feature engineering.
        clip_threshold: Upper limit for RUL target clipping.

    Returns:
        dict[str, float]: Test RMSE and NASA score.
    """
    # 1. Load test trajectories and build features
    test_raw = load_data("test_FD001.txt")
    test_featured = prepare_features(test_raw, is_train=False, window=window)

    # 2. Extract the last cycle for each test engine
    last_test_rows = test_featured.groupby("engine_id").last().reset_index()
    feature_cols = get_feature_columns(last_test_rows)
    x_test = last_test_rows[feature_cols]

    # 3. Load ground truth test RUL
    rul_file = DATASET_PATH / "RUL_FD001.txt"
    y_test = pd.read_csv(rul_file, header=None, names=["RUL"])["RUL"]
    y_test_clipped = y_test.clip(upper=clip_threshold)

    # 4. Predict and evaluate
    preds = model.predict(x_test)
    return evaluate_predictions(y_test_clipped, preds)
