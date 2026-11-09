"""Model training and evaluation pipeline for turbofan RUL prediction."""

import logging

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature

from src.data.loader import DATASET_PATH, PROJECT_ROOT, load_data
from src.evaluation.metrics import evaluate_predictions
from src.features.build_features import prepare_features

logger = logging.getLogger(__name__)

# Columns that should NOT be used as model input features
METADATA_COLS = ["engine_id", "cycle", "RUL"]

EXPERIMENT_NAME = "turbofan_rul_fd001"
DEFAULT_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()

# With only 20 validation engines, a single draw of cut-off cycles is very noisy
VAL_NUM_DRAWS = 50

BASELINE_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": 42,
    "verbose": -1,
}


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


def sample_one_row_per_engine(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Pick one random cycle per engine to mimic the truncated NASA test set.

    Rolling features look backwards only, so a sampled row uses no information
    from after its cut-off cycle.

    Args:
        df: Featured DataFrame containing 'engine_id'.
        seed: Random seed for reproducible sampling.

    Returns:
        pd.DataFrame: Exactly one row for each engine.
    """
    return df.groupby("engine_id").sample(n=1, random_state=seed).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature column names, excluding metadata and target."""
    return [c for c in df.columns if c not in METADATA_COLS]


def evaluate_per_engine(
    model: LGBMRegressor,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 42,
    n_draws: int = VAL_NUM_DRAWS,
) -> dict[str, float]:
    """Evaluate on random per-engine cut-offs, averaged over several draws.

    The NASA score is a sum over rows, so it is divided by the engine count to
    compare validation against a test set with a different number of engines.

    Args:
        model: Trained LightGBM model.
        val_df: Featured validation DataFrame.
        feature_cols: Model input columns.
        seed: Seed of the first draw. Later draws use consecutive seeds.
        n_draws: Number of random cut-off draws to average over.

    Returns:
        dict[str, float]: Mean 'rmse' and mean 'nasa_per_engine' across draws.
    """
    draws = []
    for offset in range(n_draws):
        rows = sample_one_row_per_engine(val_df, seed=seed + offset)
        draws.append(
            evaluate_predictions(rows["RUL"], model.predict(rows[feature_cols]))
        )

    n_engines = val_df["engine_id"].nunique()
    return {
        "rmse": float(np.mean([d["rmse"] for d in draws])),
        "nasa_per_engine": float(np.mean([d["nasa_score"] for d in draws])) / n_engines,
    }


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
    model = LGBMRegressor(**BASELINE_PARAMS)
    model.fit(x_train, y_train)
    return model


def evaluate_on_test_set(
    model: LGBMRegressor, window: int = 5, clip_threshold: int | None = 125
) -> dict[str, float]:
    """Evaluate trained model on the official NASA C-MAPSS test set.

    In the NASA test set, we take the last recorded cycle of each engine
    and compare predictions against the ground-truth RUL in RUL_FD001.txt.

    Args:
        model: Trained LightGBM model.
        window: Rolling window used during feature engineering.
        clip_threshold: Upper limit for test RUL labels. If None, the raw
            labels are used.

    Returns:
        dict[str, float]: Test 'rmse', 'nasa_score', and 'nasa_per_engine'.
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
    metrics = evaluate_predictions(y_test_clipped, preds)
    metrics["nasa_per_engine"] = metrics["nasa_score"] / len(y_test)
    return metrics


def run_experiment(
    window: int = 5,
    clip_threshold: int = 125,
    train_ratio: float = 0.8,
    val_seed: int = 42,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> dict[str, float]:
    """Train the baseline, evaluate it, and log the run to MLflow.

    Args:
        window: Rolling window size for feature engineering.
        clip_threshold: Piecewise RUL cap for training and clipped test labels.
        train_ratio: Fraction of engines allocated to training.
        val_seed: Seed of the first per-engine validation draw.
        tracking_uri: MLflow tracking location.

    Returns:
        dict[str, float]: All metrics logged for the run.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = prepare_features(
        load_data("train_FD001.txt"),
        is_train=True,
        window=window,
        clip_threshold=clip_threshold,
    )
    train_df, val_df = split_by_engine(df, train_ratio=train_ratio)
    feature_cols = get_feature_columns(train_df)
    x_train = train_df[feature_cols]

    with mlflow.start_run(run_name=f"baseline_lightgbm_w{window}"):
        mlflow.log_params(
            {
                "model_type": "LightGBM",
                **BASELINE_PARAMS,
                "window": window,
                "clip_threshold": clip_threshold,
                "train_ratio": train_ratio,
                "val_seed": val_seed,
                "val_num_draws": VAL_NUM_DRAWS,
            }
        )

        model = train_baseline_model(x_train, train_df["RUL"])

        all_rows = evaluate_predictions(
            val_df["RUL"], model.predict(val_df[feature_cols])
        )
        per_engine = evaluate_per_engine(model, val_df, feature_cols, seed=val_seed)
        clipped = evaluate_on_test_set(
            model, window=window, clip_threshold=clip_threshold
        )
        raw = evaluate_on_test_set(model, window=window, clip_threshold=None)

        metrics = {
            "val_rmse_all_rows": all_rows["rmse"],
            "val_rmse": per_engine["rmse"],
            "val_nasa_per_engine": per_engine["nasa_per_engine"],
            "test_rmse": clipped["rmse"],
            "test_nasa_score": clipped["nasa_score"],
            "test_nasa_per_engine": clipped["nasa_per_engine"],
            "test_rmse_raw": raw["rmse"],
            "test_nasa_score_raw": raw["nasa_score"],
        }
        mlflow.log_metrics(metrics)

        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(x_train, model.predict(x_train)),
        )

    return metrics


def main() -> None:
    """Run the baseline experiment and print its metrics."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name, value in run_experiment().items():
        logger.info("%-20s %.2f", name, value)


if __name__ == "__main__":
    main()
