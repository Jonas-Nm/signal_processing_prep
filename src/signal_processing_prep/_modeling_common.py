"""Shared contracts and table preparation for baseline modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EXCLUDED_COLUMNS = {
    "record_name",
    "label",
    "frequency_window",
    "window_start_seconds",
    "window_end_seconds",
    "window_center_seconds",
    "source_name",
    "source_path",
    "sampling_rate_hz",
    "n_samples",
    "duration_seconds",
    "window_n_samples",
    "_row_index",
}

SUPERVISED_METADATA_COLUMNS = (
    "record_name",
    "source_name",
    "label",
    "window_start_seconds",
    "window_end_seconds",
    "window_center_seconds",
)

ANOMALY_METADATA_COLUMNS = (
    "record_name",
    "label",
    "window_start_seconds",
    "window_end_seconds",
    "window_center_seconds",
)


@dataclass(frozen=True)
class ModelEvaluation:
    """Result from a simple baseline modeling run."""

    model_name: str
    task_type: str
    estimator: Any
    feature_columns: tuple[str, ...]
    metrics: dict[str, float | str]
    predictions: pd.DataFrame
    split_strategy: str
    labels: tuple[str, ...] = ()
    confusion_matrix: np.ndarray | None = None
    feature_importances: pd.Series | None = None


def top_anomalies(evaluation: ModelEvaluation, *, n: int = 5) -> pd.DataFrame:
    """Return the highest-scoring anomaly rows from an unsupervised evaluation."""
    if evaluation.task_type != "unsupervised_anomaly_score":
        raise ValueError("top_anomalies requires an unsupervised anomaly-score evaluation.")
    if n <= 0:
        raise ValueError("n must be positive.")
    if "anomaly_score" not in evaluation.predictions.columns:
        raise ValueError("Evaluation predictions do not contain anomaly_score.")
    return (
        evaluation.predictions.sort_values("anomaly_score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def anomaly_summary_text(evaluation: ModelEvaluation, *, top_n: int = 5) -> str:
    """Summarize anomaly-score results without claiming ground-truth faults."""
    anomalies = top_anomalies(evaluation, n=top_n)
    n_samples = int(evaluation.metrics.get("n_samples", len(evaluation.predictions)))
    score_min = float(evaluation.predictions["anomaly_score"].min())
    score_max = float(evaluation.predictions["anomaly_score"].max())
    rows = anomalies["row_index"].tolist()
    model_name = evaluation.model_name.replace("_", " ").title()
    fraction_text = ""
    if "anomaly_fraction" in evaluation.metrics:
        anomaly_fraction = float(evaluation.metrics["anomaly_fraction"])
        fraction_text = f"The exploratory anomaly fraction was {anomaly_fraction:.3f}; "
    return (
        f"{model_name} scored {n_samples} row(s) without using labels. "
        f"{fraction_text}"
        f"score range was {score_min:.3g} to {score_max:.3g}. "
        f"Top {len(anomalies)} row index value(s) by anomaly score: {rows}. "
        "These are candidates for inspection, not confirmed faults."
    )


def numeric_feature_matrix(
    features: pd.DataFrame,
    *,
    excluded_columns: set[str] | None = None,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Select numeric feature values for exploratory all-row fitting."""
    if feature_columns is not None:
        missing_columns = [column for column in feature_columns if column not in features.columns]
        if missing_columns:
            raise ValueError(f"Requested feature column(s) not found: {missing_columns}.")
        numeric = features.loc[:, list(feature_columns)]
    else:
        excluded = DEFAULT_EXCLUDED_COLUMNS | (excluded_columns or set())
        numeric = features.drop(columns=[column for column in excluded if column in features.columns])
    numeric = numeric.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.empty:
        return numeric, ()
    valid_row_mask = numeric.notna().any(axis=1)
    source_positions = np.flatnonzero(valid_row_mask.to_numpy()).tolist()
    numeric = numeric.loc[valid_row_mask]
    if numeric.empty:
        numeric.attrs["source_positions"] = []
        return numeric, ()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric.attrs["source_positions"] = source_positions
    return numeric, tuple(str(column) for column in numeric.columns)


def standardized_anomaly_matrix(
    features: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...], pd.DataFrame]:
    """Prepare an all-row numeric matrix and aligned metadata for anomaly scoring."""
    x, selected_columns = numeric_feature_matrix(features, feature_columns=feature_columns)
    if x.empty:
        raise ValueError("At least one numeric feature column is required.")
    if len(x) < 2:
        raise ValueError("At least two numeric feature rows are required.")
    source_rows = features.reset_index(drop=True).loc[x.attrs["source_positions"]]
    return x, selected_columns, source_rows


def anomaly_prediction_frame(
    x: pd.DataFrame,
    source_rows: pd.DataFrame,
    anomaly_score: np.ndarray,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Create the common anomaly-score output schema."""
    prediction_data: dict[str, object] = {
        "row_index": x.index,
        "anomaly_score": np.asarray(anomaly_score, dtype=np.float64),
    }
    if extra_columns:
        prediction_data.update(extra_columns)
    prediction_frame = pd.DataFrame(prediction_data)
    prediction_frame["anomaly_rank"] = (
        prediction_frame["anomaly_score"].rank(method="first", ascending=False).astype(int)
    )
    return attach_available_metadata(prediction_frame, source_rows, ANOMALY_METADATA_COLUMNS)


def attach_available_metadata(
    predictions: pd.DataFrame,
    source: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Attach available metadata columns while preserving positional alignment."""
    metadata_columns = [column for column in columns if column in source.columns]
    if not metadata_columns:
        return predictions
    if len(source) != len(predictions):
        raise ValueError("Metadata source and predictions must have the same row count.")
    metadata = source.loc[:, metadata_columns].reset_index(drop=True)
    return pd.concat([predictions.reset_index(drop=True), metadata], axis=1)
