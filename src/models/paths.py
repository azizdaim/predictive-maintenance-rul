"""Where the trained model and its evaluation summary are saved.

Kept apart from train.py so the API can find the model without importing the
training code and its heavy dependencies, such as MLflow.
"""

from pathlib import Path

from src.data.loader import PROJECT_ROOT

# Fixed location of the latest final model, loaded by the API
MODEL_PATH = PROJECT_ROOT / "models" / "rul_model.joblib"


def metrics_path_for(model_path: Path) -> Path:
    """Return where a model's evaluation summary is saved, next to the model."""
    return model_path.with_name(f"{model_path.stem}_metrics.json")
