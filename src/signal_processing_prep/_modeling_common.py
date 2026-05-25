"""Shared typed contracts and table preparation for baseline modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from signal_processing_prep.artifacts import FeatureTable, PredictionTable

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
    """Result from an interpretable supervised or anomaly-scoring model."""

    model_name: str
    task_type: str
    estimator: Any
    feature_columns: tuple[str, ...]
    metrics: dict[str, float | str]
    predictions: PredictionTable
    split_strategy: str
    labels: tuple[str, ...] = ()
    confusion_matrix: np.ndarray | None = None
    feature_importances: pd.Series | None = None

    @property
    def prediction_frame(self) -> pd.DataFrame:
        """Return an inspection-oriented DataFrame view of predictions."""
        return self.predictions.to_dataframe()


def top_anomalies(evaluation: ModelEvaluation, *, n: int = 5) -> pd.DataFrame:
    """Return highest-scoring anomaly rows for user inspection."""
    if evaluation.task_type != "unsupervised_anomaly_score":
        raise ValueError("top_anomalies requires an unsupervised anomaly-score evaluation.")
    if n <= 0:
        raise ValueError("n must be positive.")
    frame = evaluation.prediction_frame
    if "anomaly_score" not in frame.columns:
        raise ValueError("Evaluation predictions do not contain anomaly_score.")
    return frame.sort_values("anomaly_score", ascending=False).head(n).reset_index(drop=True)


def anomaly_summary_text(evaluation: ModelEvaluation, *, top_n: int = 5) -> str:
    """Summarize anomaly-score results without asserting confirmed faults."""
    frame = evaluation.prediction_frame
    anomalies = top_anomalies(evaluation, n=top_n)
    n_samples = int(evaluation.metrics.get("n_samples", len(frame)))
    score_min = float(frame["anomaly_score"].min())
    score_max = float(frame["anomaly_score"].max())
    rows = anomalies["row_index"].tolist()
    model_name = evaluation.model_name.replace("_", " ").title()
    fraction_text = ""
    if "anomaly_fraction" in evaluation.metrics:
        fraction_text = (
            f"The exploratory anomaly fraction was "
            f"{float(evaluation.metrics['anomaly_fraction']):.3f}; "
        )
    return (
        f"{model_name} scored {n_samples} row(s) without using labels. "
        f"{fraction_text}score range was {score_min:.3g} to {score_max:.3g}. "
        f"Top {len(anomalies)} row index value(s) by anomaly score: {rows}. "
        "These are candidates for inspection, not confirmed faults."
    )


def numeric_feature_matrix(
    features: FeatureTable,
    *,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Select numeric calculated features for model fitting."""
    return features.numeric_matrix(feature_columns=feature_columns)


def standardized_anomaly_matrix(
    features: FeatureTable,
    *,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...], pd.DataFrame]:
    """Prepare feature values and aligned display metadata for anomaly scoring."""
    x, selected_columns = numeric_feature_matrix(features, feature_columns=feature_columns)
    if x.empty:
        raise ValueError("At least one numeric feature column is required.")
    if len(x) < 2:
        raise ValueError("At least two numeric feature rows are required.")
    source_rows = features.to_dataframe().reset_index(drop=True).loc[x.attrs["source_positions"]]
    return x, selected_columns, source_rows


def anomaly_prediction_table(
    x: pd.DataFrame,
    source_rows: pd.DataFrame,
    anomaly_score: np.ndarray,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> PredictionTable:
    """Create the common ranked anomaly result artifact."""
    prediction_data: dict[str, object] = {
        "row_index": x.index,
        "anomaly_score": np.asarray(anomaly_score, dtype=np.float64),
    }
    if extra_columns:
        prediction_data.update(extra_columns)
    frame = pd.DataFrame(prediction_data)
    frame["anomaly_rank"] = frame["anomaly_score"].rank(method="first", ascending=False).astype(int)
    return PredictionTable.anomaly_scores(
        attach_available_metadata(frame, source_rows, ANOMALY_METADATA_COLUMNS)
    )


def attach_available_metadata(
    predictions: pd.DataFrame, source: pd.DataFrame, columns: tuple[str, ...]
) -> pd.DataFrame:
    """Attach presentation columns while preserving positional alignment."""
    available = [column for column in columns if column in source.columns]
    if not available:
        return predictions
    if len(source) != len(predictions):
        raise ValueError("Metadata source and predictions must have the same row count.")
    return pd.concat(
        [predictions.reset_index(drop=True), source.loc[:, available].reset_index(drop=True)],
        axis=1,
    )
