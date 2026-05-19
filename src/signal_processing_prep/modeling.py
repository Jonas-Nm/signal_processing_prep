"""Simple baseline models for extracted signal features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_EXCLUDED_COLUMNS = {
    "record_name",
    "label",
    "frequency_window",
    "window_start_seconds",
    "window_end_seconds",
    "window_center_seconds",
    "source_name",
}

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


def run_supervised_baselines(
    features: pd.DataFrame,
    *,
    label_column: str = "label",
    test_size: float = 0.3,
    random_state: int = 0,
) -> dict[str, ModelEvaluation]:
    """Train logistic-regression and random-forest baselines when labels exist."""
    x, y, feature_columns = _supervised_xy(features, label_column)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in the interval (0, 1).")
    if y.nunique() < 2:
        raise ValueError("At least two labels are required for supervised modeling.")
    if len(y) < 4:
        raise ValueError("At least four labeled rows are required for train/test evaluation.")
    if y.value_counts().min() < 2:
        raise ValueError("Each label must have at least two rows for train/test evaluation.")

    labels = tuple(sorted(y.astype(str).unique()))
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    split_strategy = (
        f"train_test_split test_size={test_size:g}, stratified=True, "
        f"random_state={random_state}"
    )

    models: dict[str, Any] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced",
        ),
    }
    return {
        name: _fit_supervised_model(
            name,
            estimator,
            x_train,
            x_test,
            y_train,
            y_test,
            feature_columns,
            split_strategy,
            labels,
        )
        for name, estimator in models.items()
    }


def run_isolation_forest(
    features: pd.DataFrame,
    *,
    contamination: float | str = "auto",
    random_state: int = 0,
) -> ModelEvaluation:
    """Fit an Isolation Forest and return anomaly scores without overclaiming labels."""
    x, feature_columns = _numeric_feature_matrix(features)
    if x.empty:
        raise ValueError("At least one numeric feature column is required.")
    source_rows = features.reset_index(drop=True).loc[x.attrs["source_positions"]]
    estimator = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = estimator.fit_predict(x)
    anomaly_score = -estimator.decision_function(x)
    prediction_frame = pd.DataFrame(
        {
            "row_index": x.index,
            "anomaly_score": anomaly_score,
            "isolation_forest_prediction": predictions,
        }
    )
    prediction_frame = _attach_available_metadata(
        prediction_frame,
        source_rows,
        ANOMALY_METADATA_COLUMNS,
    )
    return ModelEvaluation(
        model_name="isolation_forest",
        task_type="unsupervised_anomaly_score",
        estimator=estimator,
        feature_columns=tuple(feature_columns),
        metrics={
            "n_samples": float(len(x)),
            "contamination": str(contamination),
            "anomaly_fraction": float(np.mean(predictions == -1)),
        },
        predictions=prediction_frame,
        split_strategy="fit on all rows; anomaly scores are exploratory",
    )


def top_anomalies(
    evaluation: ModelEvaluation,
    *,
    n: int = 5,
) -> pd.DataFrame:
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


def anomaly_summary_text(
    evaluation: ModelEvaluation,
    *,
    top_n: int = 5,
) -> str:
    """Summarize anomaly-score results without claiming ground-truth faults."""
    anomalies = top_anomalies(evaluation, n=top_n)
    n_samples = int(evaluation.metrics.get("n_samples", len(evaluation.predictions)))
    anomaly_fraction = float(evaluation.metrics.get("anomaly_fraction", 0.0))
    score_min = float(evaluation.predictions["anomaly_score"].min())
    score_max = float(evaluation.predictions["anomaly_score"].max())
    rows = anomalies["row_index"].tolist()
    return (
        f"Isolation Forest scored {n_samples} row(s) without using labels. "
        f"The exploratory anomaly fraction was {anomaly_fraction:.3f}; "
        f"score range was {score_min:.3g} to {score_max:.3g}. "
        f"Top {len(anomalies)} row index value(s) by anomaly score: {rows}. "
        "These are candidates for inspection, not confirmed faults."
    )


def _fit_supervised_model(
    name: str,
    estimator: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_columns: tuple[str, ...],
    split_strategy: str,
    labels: tuple[str, ...],
) -> ModelEvaluation:
    estimator.fit(x_train, y_train)
    predicted = estimator.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision_weighted": float(precision_score(y_test, predicted, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, predicted, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
        "n_train": float(len(y_train)),
        "n_test": float(len(y_test)),
    }
    prediction_frame = pd.DataFrame(
        {
            "row_index": x_test.index,
            "true_label": y_test.to_numpy(),
            "predicted_label": predicted,
        }
    )
    return ModelEvaluation(
        model_name=name,
        task_type="supervised_classification",
        estimator=estimator,
        feature_columns=feature_columns,
        metrics=metrics,
        predictions=prediction_frame,
        split_strategy=split_strategy,
        labels=labels,
        confusion_matrix=confusion_matrix(y_test, predicted, labels=list(labels)),
        feature_importances=_feature_importances(name, estimator, feature_columns),
    )


def _supervised_xy(
    features: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.Series, tuple[str, ...]]:
    if label_column not in features.columns:
        raise ValueError(f"Label column not found: {label_column}")
    labeled = features.dropna(subset=[label_column])
    if labeled.empty:
        raise ValueError("No labeled rows are available for supervised modeling.")
    x, feature_columns = _numeric_feature_matrix(labeled, excluded_columns={label_column})
    if x.empty:
        raise ValueError("At least one numeric feature column is required for supervised modeling.")
    y = labeled[label_column].astype(str)
    return x, y, tuple(feature_columns)


def _numeric_feature_matrix(
    features: pd.DataFrame,
    *,
    excluded_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    excluded = DEFAULT_EXCLUDED_COLUMNS | (excluded_columns or set())
    numeric = features.drop(columns=[column for column in excluded if column in features.columns])
    numeric = numeric.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.empty:
        return numeric, ()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric.attrs["source_positions"] = list(range(len(numeric)))
    return numeric, tuple(str(column) for column in numeric.columns)


def _attach_available_metadata(
    predictions: pd.DataFrame,
    source: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    metadata_columns = [column for column in columns if column in source.columns]
    if not metadata_columns:
        return predictions
    if len(source) != len(predictions):
        raise ValueError("Metadata source and predictions must have the same row count.")
    metadata = source.loc[:, metadata_columns].reset_index(drop=True)
    return pd.concat([predictions.reset_index(drop=True), metadata], axis=1)


def _feature_importances(
    name: str,
    estimator: Any,
    feature_columns: tuple[str, ...],
) -> pd.Series | None:
    if name == "random_forest":
        return pd.Series(estimator.feature_importances_, index=feature_columns).sort_values(ascending=False)
    if name == "logistic_regression":
        classifier = estimator.named_steps["classifier"]
        coefficients = np.mean(np.abs(classifier.coef_), axis=0)
        return pd.Series(coefficients, index=feature_columns).sort_values(ascending=False)
    return None
