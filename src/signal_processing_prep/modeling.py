"""Simple baseline models for extracted signal features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


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


def run_supervised_baselines(
    features: pd.DataFrame,
    *,
    label_column: str = "label",
    group_column: str | None = None,
    test_size: float = 0.3,
    random_state: int = 0,
) -> dict[str, ModelEvaluation]:
    """Train logistic-regression and random-forest baselines when labels exist.

    When possible, the train/test split is grouped by source record so sliding
    windows from one acquisition do not leak across the evaluation boundary.
    """
    x, y, feature_columns, labeled = _supervised_xy(features, label_column)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in the interval (0, 1).")
    if y.nunique() < 2:
        raise ValueError("At least two labels are required for supervised modeling.")
    if len(y) < 4:
        raise ValueError("At least four labeled rows are required for train/test evaluation.")
    if y.value_counts().min() < 2:
        raise ValueError("Each label must have at least two rows for train/test evaluation.")

    labels = tuple(sorted(y.astype(str).unique()))
    train_index, test_index, split_strategy = _supervised_split(
        x,
        y,
        labeled,
        group_column=group_column,
        test_size=test_size,
        random_state=random_state,
    )
    x_train = x.loc[train_index]
    x_test = x.loc[test_index]
    y_train = y.loc[train_index]
    y_test = y.loc[test_index]
    test_metadata = labeled.loc[test_index]

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
            test_metadata,
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
    prediction_frame["anomaly_rank"] = (
        prediction_frame["anomaly_score"].rank(method="first", ascending=False).astype(int)
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


def run_one_class_svm(
    features: pd.DataFrame,
    *,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    nu: float = 0.05,
) -> ModelEvaluation:
    """Fit a One-Class SVM on numeric features and return exploratory anomaly scores."""
    if not 0.0 < nu <= 1.0:
        raise ValueError("nu must be in the interval (0, 1].")
    x, feature_columns, source_rows = _standardized_anomaly_matrix(features)
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)),
        ]
    )
    predictions = estimator.fit_predict(x)
    anomaly_score = -estimator.decision_function(x)
    prediction_frame = _anomaly_prediction_frame(
        x,
        source_rows,
        anomaly_score,
        {"one_class_svm_prediction": predictions},
    )
    return ModelEvaluation(
        model_name="one_class_svm",
        task_type="unsupervised_anomaly_score",
        estimator=estimator,
        feature_columns=feature_columns,
        metrics={
            "n_samples": float(len(x)),
            "nu": float(nu),
            "anomaly_fraction": float(np.mean(predictions == -1)),
        },
        predictions=prediction_frame,
        split_strategy="fit on all rows; anomaly scores are exploratory",
    )


def run_local_outlier_factor(
    features: pd.DataFrame,
    *,
    n_neighbors: int = 20,
    contamination: float | str = "auto",
) -> ModelEvaluation:
    """Fit Local Outlier Factor on numeric features and return anomaly scores."""
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")
    x, feature_columns, source_rows = _standardized_anomaly_matrix(features)
    resolved_neighbors = min(n_neighbors, max(len(x) - 1, 1))
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LocalOutlierFactor(
                    n_neighbors=resolved_neighbors,
                    contamination=contamination,
                ),
            ),
        ]
    )
    predictions = estimator.fit_predict(x)
    model = estimator.named_steps["model"]
    anomaly_score = -model.negative_outlier_factor_
    prediction_frame = _anomaly_prediction_frame(
        x,
        source_rows,
        anomaly_score,
        {"local_outlier_factor_prediction": predictions},
    )
    return ModelEvaluation(
        model_name="local_outlier_factor",
        task_type="unsupervised_anomaly_score",
        estimator=estimator,
        feature_columns=feature_columns,
        metrics={
            "n_samples": float(len(x)),
            "n_neighbors": float(resolved_neighbors),
            "contamination": str(contamination),
            "anomaly_fraction": float(np.mean(predictions == -1)),
        },
        predictions=prediction_frame,
        split_strategy="fit on all rows; anomaly scores are exploratory",
    )


def run_pca_reconstruction(
    features: pd.DataFrame,
    *,
    n_components: int | float = 0.95,
) -> ModelEvaluation:
    """Score windows by PCA reconstruction error on standardized numeric features."""
    x, feature_columns, source_rows = _standardized_anomaly_matrix(features)
    if isinstance(n_components, int) and n_components <= 0:
        raise ValueError("n_components must be positive.")
    if isinstance(n_components, float) and not 0.0 < n_components < 1.0:
        raise ValueError("float n_components must be in the interval (0, 1).")
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
        ]
    )
    scaled = estimator.named_steps["scaler"].fit_transform(x)
    pca = estimator.named_steps["pca"]
    transformed = pca.fit_transform(scaled)
    reconstructed = pca.inverse_transform(transformed)
    anomaly_score = np.mean(np.square(scaled - reconstructed), axis=1)
    prediction_frame = _anomaly_prediction_frame(x, source_rows, anomaly_score)
    return ModelEvaluation(
        model_name="pca_reconstruction",
        task_type="unsupervised_anomaly_score",
        estimator=estimator,
        feature_columns=feature_columns,
        metrics={
            "n_samples": float(len(x)),
            "n_components": float(pca.n_components_),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        },
        predictions=prediction_frame,
        split_strategy="fit on all rows; reconstruction errors are exploratory",
    )


def run_dbscan_outlier_scores(
    features: pd.DataFrame,
    *,
    eps: float = 1.5,
    min_samples: int = 5,
) -> ModelEvaluation:
    """Fit DBSCAN and score sparse or noise windows as clustering outliers.

    DBSCAN is sensitive to ``eps`` and should be treated as an exploratory
    clustering diagnostic rather than a calibrated detector.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive.")
    x, feature_columns, source_rows = _standardized_anomaly_matrix(features)
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", DBSCAN(eps=eps, min_samples=min_samples)),
        ]
    )
    labels = estimator.fit_predict(x)
    scaled = estimator.named_steps["scaler"].transform(x)
    anomaly_score = _dbscan_scores(scaled, labels, min_samples=min_samples)
    prediction_frame = _anomaly_prediction_frame(
        x,
        source_rows,
        anomaly_score,
        {"dbscan_label": labels},
    )
    return ModelEvaluation(
        model_name="dbscan_outlier_scores",
        task_type="unsupervised_anomaly_score",
        estimator=estimator,
        feature_columns=feature_columns,
        metrics={
            "n_samples": float(len(x)),
            "eps": float(eps),
            "min_samples": float(min_samples),
            "noise_fraction": float(np.mean(labels == -1)),
            "n_clusters": float(len(set(labels) - {-1})),
        },
        predictions=prediction_frame,
        split_strategy="fit on all rows; DBSCAN scores are exploratory and eps-sensitive",
    )


def run_robust_z_score(
    features: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> ModelEvaluation:
    """Score unusually high feature values with summed positive robust z-scores."""
    x, selected_columns, source_rows = _standardized_anomaly_matrix(
        features,
        feature_columns=feature_columns,
    )
    median = x.median(axis=0)
    mad = (x - median).abs().median(axis=0)
    usable_scale = mad.where(np.isfinite(mad) & (mad != 0.0))
    robust_z = 0.6745 * (x - median) / usable_scale
    anomaly_score = robust_z.clip(lower=0.0).fillna(0.0).sum(axis=1).to_numpy()
    prediction_frame = _anomaly_prediction_frame(x, source_rows, anomaly_score)
    return ModelEvaluation(
        model_name="robust_z_score",
        task_type="unsupervised_anomaly_score",
        estimator=None,
        feature_columns=selected_columns,
        metrics={"n_samples": float(len(x))},
        predictions=prediction_frame,
        split_strategy="fit on all rows; robust positive z-scores are exploratory",
    )


def run_mahalanobis_distance(
    features: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> ModelEvaluation:
    """Score rows by squared Mahalanobis distance from the feature centroid."""
    x, selected_columns, source_rows = _standardized_anomaly_matrix(
        features,
        feature_columns=feature_columns,
    )
    values = x.to_numpy(dtype=np.float64)
    centroid = values.mean(axis=0)
    covariance = np.atleast_2d(np.cov(values, rowvar=False))
    inverse_covariance = np.linalg.pinv(covariance)
    centered = values - centroid
    anomaly_score = np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered)
    prediction_frame = _anomaly_prediction_frame(x, source_rows, anomaly_score)
    return ModelEvaluation(
        model_name="mahalanobis_distance",
        task_type="unsupervised_anomaly_score",
        estimator={
            "centroid": centroid,
            "inverse_covariance": inverse_covariance,
        },
        feature_columns=selected_columns,
        metrics={"n_samples": float(len(x))},
        predictions=prediction_frame,
        split_strategy="fit on all rows; Mahalanobis distances are exploratory",
    )


def run_robust_mahalanobis_distance(
    features: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] | None = None,
    support_fraction: float | None = None,
    random_state: int | None = 0,
) -> ModelEvaluation:
    """Score rows by robust squared Mahalanobis distance using MCD covariance."""
    if support_fraction is not None and (
        not np.isfinite(support_fraction) or not 0.0 < support_fraction <= 1.0
    ):
        raise ValueError("support_fraction must be in the interval (0, 1].")
    x, selected_columns, source_rows = _standardized_anomaly_matrix(
        features,
        feature_columns=feature_columns,
    )
    values = x.to_numpy(dtype=np.float64)
    if np.linalg.matrix_rank(values - values.mean(axis=0)) < values.shape[1]:
        raise ValueError(
            "Robust Mahalanobis fitting failed; selected numeric features may be "
            "degenerate or insufficient for robust covariance estimation."
        )
    estimator = MinCovDet(
        support_fraction=support_fraction,
        random_state=random_state,
    )
    try:
        estimator.fit(values)
        if np.linalg.matrix_rank(estimator.covariance_) < estimator.covariance_.shape[0]:
            raise ValueError("Robust covariance matrix is rank deficient.")
        anomaly_score = estimator.mahalanobis(values)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "Robust Mahalanobis fitting failed; selected numeric features may be "
            "degenerate or insufficient for robust covariance estimation."
        ) from error
    if not np.isfinite(anomaly_score).all():
        raise ValueError(
            "Robust Mahalanobis fitting produced non-finite scores; selected "
            "numeric features may be degenerate."
        )
    prediction_frame = _anomaly_prediction_frame(x, source_rows, anomaly_score)
    return ModelEvaluation(
        model_name="robust_mahalanobis_distance",
        task_type="unsupervised_anomaly_score",
        estimator=estimator,
        feature_columns=selected_columns,
        metrics={
            "n_samples": float(len(x)),
            "support_fraction": (
                "auto" if support_fraction is None else float(support_fraction)
            ),
            "support_count": float(np.sum(estimator.support_)),
        },
        predictions=prediction_frame,
        split_strategy="fit on all rows; robust Mahalanobis distances are exploratory",
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


def _fit_supervised_model(
    name: str,
    estimator: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    test_metadata: pd.DataFrame,
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
            "row_index": _prediction_row_index(x_test, test_metadata),
            "true_label": y_test.to_numpy(),
            "predicted_label": predicted,
        }
    )
    prediction_frame = _attach_available_metadata(
        prediction_frame,
        test_metadata.reset_index(drop=True),
        SUPERVISED_METADATA_COLUMNS,
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
) -> tuple[pd.DataFrame, pd.Series, tuple[str, ...], pd.DataFrame]:
    if label_column not in features.columns:
        raise ValueError(f"Label column not found: {label_column}")
    labeled = features.dropna(subset=[label_column]).copy()
    if labeled.empty:
        raise ValueError("No labeled rows are available for supervised modeling.")
    labeled["_row_index"] = labeled.index
    labeled = labeled.reset_index(drop=True)
    x, feature_columns = _numeric_feature_matrix(labeled, excluded_columns={label_column})
    if x.empty:
        raise ValueError("At least one numeric feature column is required for supervised modeling.")
    y = labeled[label_column].astype(str)
    labeled = labeled.loc[x.index]
    y = y.loc[x.index]
    return x, y, tuple(feature_columns), labeled


def _supervised_split(
    x: pd.DataFrame,
    y: pd.Series,
    labeled: pd.DataFrame,
    *,
    group_column: str | None,
    test_size: float,
    random_state: int,
) -> tuple[pd.Index, pd.Index, str]:
    resolved_group_column = _resolve_group_column(labeled, group_column)
    if resolved_group_column is None:
        train_index, test_index = train_test_split(
            x.index,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        return (
            pd.Index(train_index),
            pd.Index(test_index),
            f"train_test_split test_size={test_size:g}, stratified=True, random_state={random_state}",
        )

    groups = labeled[resolved_group_column].astype(str)
    _validate_group_labels(groups, y, resolved_group_column)
    splitter = GroupShuffleSplit(n_splits=50, test_size=test_size, random_state=random_state)
    for train_positions, test_positions in splitter.split(x, y, groups):
        train_index = x.index[train_positions]
        test_index = x.index[test_positions]
        if _split_preserves_labels(y.loc[train_index], y.loc[test_index]):
            return (
                train_index,
                test_index,
                (
                    f"group_shuffle_split group_column={resolved_group_column}, "
                    f"test_size={test_size:g}, random_state={random_state}"
                ),
            )
    raise ValueError(
        "Grouped train/test split could not preserve every label in both train and test sets. "
        "Use more source records per label or explicitly pass group_column=None only if row-level "
        "leakage is acceptable for the analysis."
    )


def _resolve_group_column(labeled: pd.DataFrame, group_column: str | None) -> str | None:
    if group_column is not None:
        if group_column not in labeled.columns:
            raise ValueError(f"Group column not found: {group_column}")
        return group_column
    for candidate in ("source_name", "record_name"):
        if candidate in labeled.columns and not labeled[candidate].isna().all():
            return candidate
    return None


def _validate_group_labels(groups: pd.Series, y: pd.Series, group_column: str) -> None:
    label_counts_by_group = y.groupby(groups).nunique()
    if (label_counts_by_group > 1).any():
        raise ValueError(f"Each {group_column} group must contain only one label.")
    groups_per_label = groups.groupby(y).nunique()
    if (groups_per_label < 2).any():
        raise ValueError(
            f"Each label must have at least two {group_column} groups for grouped evaluation."
        )


def _split_preserves_labels(y_train: pd.Series, y_test: pd.Series) -> bool:
    labels = set(y_train.astype(str).unique()) | set(y_test.astype(str).unique())
    return labels == set(y_train.astype(str).unique()) == set(y_test.astype(str).unique())


def _prediction_row_index(x_test: pd.DataFrame, test_metadata: pd.DataFrame) -> pd.Series | pd.Index:
    if "_row_index" in test_metadata.columns:
        return test_metadata["_row_index"].reset_index(drop=True)
    return x_test.index


def _numeric_feature_matrix(
    features: pd.DataFrame,
    *,
    excluded_columns: set[str] | None = None,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
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


def _standardized_anomaly_matrix(
    features: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...], pd.DataFrame]:
    x, selected_columns = _numeric_feature_matrix(features, feature_columns=feature_columns)
    if x.empty:
        raise ValueError("At least one numeric feature column is required.")
    if len(x) < 2:
        raise ValueError("At least two numeric feature rows are required.")
    source_rows = features.reset_index(drop=True).loc[x.attrs["source_positions"]]
    return x, selected_columns, source_rows


def _anomaly_prediction_frame(
    x: pd.DataFrame,
    source_rows: pd.DataFrame,
    anomaly_score: np.ndarray,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
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
    return _attach_available_metadata(
        prediction_frame,
        source_rows,
        ANOMALY_METADATA_COLUMNS,
    )


def _dbscan_scores(values: np.ndarray, labels: np.ndarray, *, min_samples: int) -> np.ndarray:
    cluster_labels = sorted(set(labels) - {-1})
    local_radius = _nearest_neighbor_radius(values, min_samples=min_samples)
    if not cluster_labels:
        return local_radius

    centers = np.vstack([values[labels == label].mean(axis=0) for label in cluster_labels])
    distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
    nearest_distance = distances.min(axis=1)
    cluster_sizes = {label: int(np.sum(labels == label)) for label in cluster_labels}
    sparsity = np.array(
        [1.0 / cluster_sizes[label] if label != -1 else 1.0 for label in labels],
        dtype=np.float64,
    )
    max_distance = float(np.max(nearest_distance)) if nearest_distance.size else 0.0
    noise_boost = np.where(labels == -1, max_distance + 1.0, 0.0)
    return nearest_distance + sparsity + local_radius + noise_boost


def _nearest_neighbor_radius(values: np.ndarray, *, min_samples: int) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float64)
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    sorted_distances = np.sort(distances, axis=1)
    neighbor_index = min(max(min_samples - 1, 1), len(values) - 1)
    return sorted_distances[:, neighbor_index]


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
