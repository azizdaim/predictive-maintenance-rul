"""FastAPI service for turbofan RUL predictions."""

from functools import lru_cache
from itertools import pairwise
from typing import Annotated

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from lightgbm import LGBMRegressor
from pydantic import BaseModel, Field, model_validator

from src.features.build_features import TREND_LAG, prepare_features
from src.models.train import MODEL_PATH

app = FastAPI(title="Turbofan RUL API")

# Training targets were capped here, so predictions are kept in the same range
RUL_CAP = 125.0


class CycleReading(BaseModel):
    """One cycle of raw engine telemetry, in the C-MAPSS column format."""

    cycle: int = Field(ge=1)
    setting_1: float
    setting_2: float
    setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class EngineHistory(BaseModel):
    """All cycles recorded so far for one engine, oldest first."""

    cycles: list[CycleReading] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def check_cycle_order(self) -> "EngineHistory":
        """Reject histories whose cycle numbers are not strictly increasing."""
        numbers = [reading.cycle for reading in self.cycles]
        if any(later <= earlier for earlier, later in pairwise(numbers)):
            raise ValueError("cycles must be in strictly increasing order")
        return self


class Prediction(BaseModel):
    """Predicted remaining useful life for the latest cycle."""

    predicted_rul: float
    last_cycle: int
    cycles_received: int
    warning: str | None = None


@lru_cache
def get_model() -> LGBMRegressor:
    """Load the trained model once and reuse it for every request."""
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run `python -m src.models.train` first.",
        )
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the service is running."""
    return {"status": "ok"}


@app.post("/predict")
def predict(
    history: EngineHistory,
    model: Annotated[LGBMRegressor, Depends(get_model)],
) -> Prediction:
    """Predict RUL at the latest cycle from an engine's sensor history."""
    raw = pd.DataFrame([reading.model_dump() for reading in history.cycles])
    raw.insert(0, "engine_id", 1)

    # Same feature code as training, so serving cannot drift from it
    features = prepare_features(raw, is_train=False)
    latest = features.iloc[[-1]][model.feature_name_]

    # Keep within 0 to the training cap, since raw model output can overshoot
    predicted = min(max(float(model.predict(latest)[0]), 0.0), RUL_CAP)

    warning = None
    if len(history.cycles) <= TREND_LAG:
        warning = (
            f"Only {len(history.cycles)} cycles received. Trend features need "
            f"more than {TREND_LAG}, so this prediction is less reliable."
        )

    return Prediction(
        predicted_rul=round(predicted, 2),
        last_cycle=history.cycles[-1].cycle,
        cycles_received=len(history.cycles),
        warning=warning,
    )
