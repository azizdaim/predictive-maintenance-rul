"""Data preprocessing utilities for the C-MAPSS dataset."""

import pandas as pd

# The 7 constant columns we identified during EDA
CONSTANT_COLUMNS = [
    "setting_3",
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop zero-variance sensor and setting columns.

    Args:
        df: Input DataFrame containing raw C-MAPSS telemetry.

    Returns:
        pd.DataFrame: DataFrame with constant columns removed.
    """
    df_processed = df.drop(columns=CONSTANT_COLUMNS, errors="ignore")
    return df_processed

def add_rul_target(
    df: pd.DataFrame,
    clip_threshold: int | None = 125
    ) -> pd.DataFrame:
    """Calculate Remaining Useful Life (RUL) for each engine cycle.

    Args:
        df: Input DataFrame containing 'engine_id' and 'cycle'.
        clip_threshold: Upper limit for RUL (piecewise linear target).
            If None, raw linear RUL is kept. Default is 125.

    Returns:
        pd.DataFrame: DataFrame with new 'RUL' column added.
    """
    # 1. Make a copy of df to avoid mutating the original input
    df_out = df.copy()

    # 2. Compute max cycle per engine and subtract current cycle
    max_cycles =  df.groupby("engine_id")["cycle"].transform("max")
    df_out['RUL'] = max_cycles - df_out['cycle']


    # 3. If clip_threshold is provided, clip the RUL column
    # (Hint: pandas has a built-in method: df_out["RUL"].clip(upper=...))
    if clip_threshold is not None:
        df_out["RUL"] = df_out["RUL"].clip(upper=clip_threshold)

    # 4. Return df_out
    return df_out
