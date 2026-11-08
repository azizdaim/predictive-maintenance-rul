"""Unit tests for the evaluation metrics module."""

import numpy as np

from src.evaluation.metrics import (
    compute_nasa_score,
    compute_rmse,
    evaluate_predictions,
)


def test_compute_rmse() -> None:
    """Verify RMSE returns 0 on identical inputs and correct values on offsets."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred_perfect = np.array([10.0, 20.0, 30.0])
    y_pred_offset = np.array([12.0, 22.0, 32.0])  # error of +2 on every sample

    assert compute_rmse(y_true, y_pred_perfect) == 0.0
    assert compute_rmse(y_true, y_pred_offset) == 2.0


def test_nasa_score_asymmetry() -> None:
    """Verify that NASA score penalizes late predictions more than early ones."""
    y_true = np.array([100.0])

    # Case 1: Predicted 10 cycles early (safe) -> d = -10
    early_pred = np.array([90.0])
    early_score = compute_nasa_score(y_true, early_pred)

    # Case 2: Predicted 10 cycles late (dangerous) -> d = +10
    late_pred = np.array([110.0])
    late_score = compute_nasa_score(y_true, late_pred)

    # Perfect prediction has score 0
    assert compute_nasa_score(y_true, y_true) == 0.0

    # Late prediction penalty MUST be greater than early prediction penalty
    assert late_score > early_score
    assert np.isclose(early_score, np.exp(10.0 / 13.0) - 1.0)
    assert np.isclose(late_score, np.exp(10.0 / 10.0) - 1.0)


def test_evaluate_predictions_output_format() -> None:
    """Verify evaluate_predictions returns dictionary with expected keys."""
    y_true = np.array([50.0, 40.0])
    y_pred = np.array([48.0, 42.0])

    results = evaluate_predictions(y_true, y_pred)

    assert "rmse" in results
    assert "nasa_score" in results
    assert isinstance(results["rmse"], float)
    assert isinstance(results["nasa_score"], float)

