"""Data loading utilities for the C-MAPSS turbofan dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw"


def load_data(filename: str) -> pd.DataFrame:
    """Load a C-MAPSS data file and return a DataFrame with named columns.

    Args:
        filename: Name of the raw text file in data/raw (e.g., 'train_FD001.txt').

    Returns:
        pd.DataFrame: Clean DataFrame containing engine run-to-failure cycles,
            operational settings, and sensor measurements.
    """
    columns = ["engine_id", "cycle"]
    columns += [f"setting_{i}" for i in range(1, 4)]
    columns += [f"sensor_{i}" for i in range(1, 22)]

    return pd.read_csv(
        DATASET_PATH / filename,
        sep=r"\s+",
        header=None,
        names=columns,
    )
