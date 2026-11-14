"""FastAPI service for turbofan RUL predictions."""

from functools import lru_cache
from itertools import pairwise
from typing import Annotated

import joblib
import numpy as np
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


class CyclePrediction(BaseModel):
    """Predicted remaining useful life at one cycle."""

    cycle: int
    predicted_rul: float


class HistoryPrediction(BaseModel):
    """Predicted remaining useful life at every cycle of a history."""

    predictions: list[CyclePrediction]


@lru_cache
def get_model() -> LGBMRegressor:
    """Load the trained model once and reuse it for every request."""
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run `python -m src.models.train` first.",
        )
    return joblib.load(MODEL_PATH)


def predict_every_cycle(history: EngineHistory, model: LGBMRegressor) -> np.ndarray:
    """Predict RUL at each cycle, using only that cycle and the ones before it.

    Features look backwards only, so one pass over the full history gives the
    same value at each cycle as sending the history cut off at that cycle.
    """
    raw = pd.DataFrame([reading.model_dump() for reading in history.cycles])
    raw.insert(0, "engine_id", 1)

    # Same feature code as training, so serving cannot drift from it
    features = prepare_features(raw, is_train=False)

    # Keep within 0 to the training cap, since raw model output can overshoot
    return np.clip(model.predict(features[model.feature_name_]), 0.0, RUL_CAP)


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
    predicted = float(predict_every_cycle(history, model)[-1])

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


@app.post("/predict/history")
def predict_history(
    history: EngineHistory,
    model: Annotated[LGBMRegressor, Depends(get_model)],
) -> HistoryPrediction:
    """Predict RUL at every cycle of an engine's history in one request."""
    predicted = predict_every_cycle(history, model)

    return HistoryPrediction(
        predictions=[
            CyclePrediction(cycle=reading.cycle, predicted_rul=round(float(rul), 2))
            for reading, rul in zip(history.cycles, predicted, strict=True)
        ]
    )
