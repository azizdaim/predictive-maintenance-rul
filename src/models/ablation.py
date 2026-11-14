"""Ablation study: remove one group of features at a time and measure the damage.

Each variant drops a group of features, is scored with the same engine-grouped
cross-validation as the full model, and is compared shuffle by shuffle. A group
whose removal barely changes the error is not pulling its weight.

Run with: python -m src.models.ablation
"""

import logging
import re

import mlflow
import numpy as np
import pandas as pd

from src.data.loader import PROJECT_ROOT, load_data
from src.features.build_features import prepare_features
from src.models.train import (
    BASELINE_PARAMS,
    DEFAULT_TRACKING_URI,
    cross_validate_by_engine,
    get_feature_columns,
)

logger = logging.getLogger(__name__)

ABLATION_EXPERIMENT = "turbofan_rul_ablation"
REPORT_PATH = PROJECT_ROOT / "reports" / "ablation_results.csv"
FULL_MODEL = "All features"

# Which engine section each active C-MAPSS sensor measures (Saxena et al., 2008)
ENGINE_SECTIONS: dict[str, list[int]] = {
    "Fan": [6, 8, 13, 15],
    "Low-pressure compressor": [2],
    "High-pressure compressor": [3, 7, 9, 11, 14],
    "Combustor": [12, 17],
    "High-pressure turbine": [20],
    "Low-pressure turbine": [4, 21],
}

_SENSOR_NUMBER = re.compile(r"^sensor_(\d+)")


def feature_type_groups(columns: list[str]) -> dict[str, list[str]]:
    """Split feature columns by how they were made.

    Args:
        columns: Model feature column names.

    Returns:
        dict[str, list[str]]: Group name to the columns in that group.
    """
    groups: dict[str, list[str]] = {
        "Operating settings": [],
        "Raw sensor readings": [],
        "Rolling averages": [],
        "Rolling spreads": [],
        "30-cycle trends": [],
    }
    for col in columns:
        if col.startswith("setting_"):
            groups["Operating settings"].append(col)
        elif "_roll_mean_" in col:
            groups["Rolling averages"].append(col)
        elif "_roll_std_" in col:
            groups["Rolling spreads"].append(col)
        elif "_trend_" in col:
            groups["30-cycle trends"].append(col)
        else:
            groups["Raw sensor readings"].append(col)
    return groups


def engine_section_groups(columns: list[str]) -> dict[str, list[str]]:
    """Split sensor-derived feature columns by the engine section they measure.

    Args:
        columns: Model feature column names.

    Returns:
        dict[str, list[str]]: Section name to every feature built from its sensors.
    """
    section_of = {
        number: section
        for section, numbers in ENGINE_SECTIONS.items()
        for number in numbers
    }
    groups: dict[str, list[str]] = {section: [] for section in ENGINE_SECTIONS}
    for col in columns:
        match = _SENSOR_NUMBER.match(col)
        if match:
            groups[section_of[int(match.group(1))]].append(col)
    return groups


def run_ablation(
    df: pd.DataFrame,
    groups: dict[str, dict[str, list[str]]],
    seeds: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    """Cross-validate the full model and every drop-one-group variant.

    Args:
        df: Featured training data.
        groups: Study name to its groups, for example by feature type.
        seeds: Fold shuffles to repeat each variant over.

    Returns:
        pd.DataFrame: One row per variant, with mean scores and the change from
            the full model, where a positive change means the variant is worse.
    """
    variants = [("Baseline", FULL_MODEL, [])]
    for study, study_groups in groups.items():
        variants += [
            (study, f"Without {name.lower()}", cols)
            for name, cols in study_groups.items()
        ]

    scores = {}
    for _, variant, dropped in variants:
        reduced = df.drop(columns=dropped)
        scores[variant] = np.array(
            [
                [
                    result["cv_rmse_mean"],
                    result["cv_near_failure_rmse_mean"],
                    result["cv_nasa_per_engine_mean"],
                ]
                for result in (
                    cross_validate_by_engine(reduced, seed=seed) for seed in seeds
                )
            ]
        )
        logger.info("%-40s RMSE %.2f", variant, scores[variant][:, 0].mean())

    full = scores[FULL_MODEL]
    rows = []
    for study, variant, dropped in variants:
        change = scores[variant] - full
        rows.append(
            {
                "study": study,
                "variant": variant,
                "features_removed": len(dropped),
                "rmse": scores[variant][:, 0].mean(),
                "near_failure_rmse": scores[variant][:, 1].mean(),
                "nasa_per_engine": scores[variant][:, 2].mean(),
                "rmse_change": change[:, 0].mean(),
                "near_failure_change": change[:, 1].mean(),
                "worse_in_shuffles": int((change[:, 0] > 0).sum()),
                "shuffles": len(seeds),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run both ablation studies, log them to MLflow, and save a report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = prepare_features(load_data("train_FD001.txt"), is_train=True)
    columns = get_feature_columns(df)
    groups = {
        "Feature type": feature_type_groups(columns),
        "Engine section": engine_section_groups(columns),
    }

    results = run_ablation(df, groups)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.round(3).to_csv(REPORT_PATH, index=False)

    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    mlflow.set_experiment(ABLATION_EXPERIMENT)
    with mlflow.start_run(run_name="ablation_study"):
        mlflow.log_params({**BASELINE_PARAMS, "shuffles": int(results["shuffles"][0])})
        mlflow.log_artifact(str(REPORT_PATH))
        for row in results.itertuples():
            with mlflow.start_run(run_name=row.variant, nested=True):
                mlflow.log_params(
                    {"study": row.study, "features_removed": row.features_removed}
                )
                mlflow.log_metrics(
                    {
                        "cv_rmse": row.rmse,
                        "cv_near_failure_rmse": row.near_failure_rmse,
                        "rmse_change": row.rmse_change,
                    }
                )

    pd.set_option("display.width", 160)
    logger.info(
        "\n%s",
        results[
            ["variant", "features_removed", "rmse", "rmse_change",
             "near_failure_change", "worse_in_shuffles"]
        ].round(2).to_string(index=False),
    )
    logger.info("Saved report to %s", REPORT_PATH)


if __name__ == "__main__":
    main()
