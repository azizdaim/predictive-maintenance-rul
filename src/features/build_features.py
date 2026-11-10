"""Feature engineering module for turbofan sensor telemetry."""

import pandas as pd

from src.data.preprocessing import add_rul_target, drop_constant_columns

# Shortest test engine has 31 cycles, so this look-back is available for all of them
TREND_LAG = 30


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = 5,
) -> pd.DataFrame:
    """Compute rolling mean and standard deviation for sensor columns per engine.

    Args:
        df: Input DataFrame containing 'engine_id' and sensor columns.
        sensor_cols: List of sensor column names to engineer features for.
        window: Number of historical cycles in the rolling window. Default is 5.

    Returns:
        pd.DataFrame: A copy of the DataFrame with new rolling feature columns added.
    """
    df_out = df.copy()

    for col in sensor_cols:
        # 1. Create the engine-isolated rolling object
        roller = df_out.groupby("engine_id")[col].rolling(window=window, min_periods=1)

        # 2. Compute Rolling Mean and assign to df_out
        mean_col_name = f"{col}_roll_mean_{window}"
        df_out[mean_col_name] = roller.mean().reset_index(level=0, drop=True)

        # 3. Compute Rolling Std, fill the cycle #1 NaN with 0.0, and assign
        std_col_name = f"{col}_roll_std_{window}"
        df_out[std_col_name] = (
            roller.std().fillna(0.0).reset_index(level=0, drop=True)
        )
    return df_out


def add_trend_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = 5,
    lag: int = TREND_LAG,
) -> pd.DataFrame:
    """Add how much each sensor's rolling mean changed over the last `lag` cycles.

    Rolling means are used instead of raw readings to reduce noise. Rows with
    fewer than `lag` earlier cycles in their engine are filled with 0.0.

    Args:
        df: DataFrame containing 'engine_id' and rolling mean columns.
        sensor_cols: Sensor column names that already have rolling means.
        window: Window used to compute the rolling means.
        lag: Number of cycles to look back.

    Returns:
        pd.DataFrame: A copy of the DataFrame with new trend columns added.
    """
    df_out = df.copy()
    by_engine = df_out.groupby("engine_id")

    for col in sensor_cols:
        trend = by_engine[f"{col}_roll_mean_{window}"].diff(lag)
        df_out[f"{col}_trend_{lag}"] = trend.fillna(0.0)

    return df_out


def prepare_features(
    df: pd.DataFrame,
    is_train: bool = True,
    window: int = 5,
    clip_threshold: int | None = 125,
    trend_lag: int = TREND_LAG,
) -> pd.DataFrame:
    """Run full feature engineering pipeline on raw C-MAPSS telemetry.

    Args:
        df: Raw C-MAPSS DataFrame.
        is_train: Whether this is training data (if True, computes RUL target).
        window: Rolling window size. Default is 5.
        clip_threshold: Piecewise RUL clipping cap. Default is 125.
        trend_lag: Number of cycles the trend features look back.

    Returns:
        pd.DataFrame: Cleaned DataFrame with rolling and trend features, and RUL.
    """
    # 1. Drop the 7 dead/constant sensors
    df_clean = drop_constant_columns(df)

    # 2. Add RUL target only for training data
    if is_train:
        df_clean = add_rul_target(df_clean, clip_threshold=clip_threshold)

    # 3. Find all remaining active sensors
    sensor_cols = [c for c in df_clean.columns if c.startswith("sensor_")]

    # 4. Compute rolling mean and std for active sensors
    df_featured = add_rolling_features(df_clean, sensor_cols=sensor_cols, window=window)

    # 5. Add how much each sensor has drifted over the longer term
    return add_trend_features(
        df_featured, sensor_cols=sensor_cols, window=window, lag=trend_lag
    )
