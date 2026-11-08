"""Evaluation metrics for turbofan Remaining Useful Life (RUL) prediction."""

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def compute_rmse(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    Args:
        y_true: Ground truth Remaining Useful Life values.
        y_pred: Model predicted Remaining Useful Life values.

    Returns:
        float: RMSE value (average error in flight cycles).
    """
    return float(root_mean_squared_error(y_true, y_pred))


def compute_nasa_score(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """Calculate the NASA C-MAPSS asymmetric evaluation score.

    Penalizes late predictions (overestimating remaining life) much more
    heavily than early predictions (underestimating remaining life).

    Formula:
        d = y_pred - y_true
        s_i = exp(-d / 13) - 1  if d < 0 (early prediction)
        s_i = exp(d / 10) - 1   if d >= 0 (late prediction)
        Score = sum(s_i)

    Args:
        y_true: Ground truth Remaining Useful Life values.
        y_pred: Model predicted Remaining Useful Life values.

    Returns:
        float: Total NASA asymmetric penalty score.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)

    # Use np.where to apply the asymmetric penalty piece-wise
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)

    return float(np.sum(scores))


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute all evaluation metrics and return as a clean summary dict.

    Args:
        y_true: Ground truth RUL.
        y_pred: Predicted RUL.

    Returns:
        dict[str, float]: Dictionary with 'rmse' and 'nasa_score'.
    """
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "nasa_score": compute_nasa_score(y_true, y_pred),
    }

