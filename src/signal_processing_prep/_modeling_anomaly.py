"""Exploratory anomaly scoring models for extracted signal features."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from signal_processing_prep._modeling_common import (
    ModelEvaluation,
    anomaly_prediction_table,
    numeric_feature_matrix,
    standardized_anomaly_matrix,
)
from signal_processing_prep.artifacts import FeatureTable


def run_isolation_forest(
    features: FeatureTable,
    *,
    contamination: float | str = "auto",
    random_state: int = 0,
) -> ModelEvaluation:
    """Fit an Isolation Forest and return anomaly scores without asserting labels."""
    x, feature_columns = numeric_feature_matrix(features)
    if x.empty:
        raise ValueError("At least one numeric feature column is required.")
    source_rows = features.to_dataframe().reset_index(drop=True).loc[x.attrs["source_positions"]]
    estimator = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = estimator.fit_predict(x)
    anomaly_score = -estimator.decision_function(x)
    prediction_table = anomaly_prediction_table(
        x,
        source_rows,
        anomaly_score,
        {"isolation_forest_prediction": predictions},
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
        predictions=prediction_table,
        split_strategy="fit on all rows; anomaly scores are exploratory",
    )


def run_one_class_svm(
    features: FeatureTable,
    *,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    nu: float = 0.05,
) -> ModelEvaluation:
    """Fit a One-Class SVM on numeric features and return exploratory scores."""
    if not 0.0 < nu <= 1.0:
        raise ValueError("nu must be in the interval (0, 1].")
    x, feature_columns, source_rows = standardized_anomaly_matrix(features)
    estimator = Pipeline(
        [("scaler", StandardScaler()), ("model", OneClassSVM(kernel=kernel, gamma=gamma, nu=nu))]
    )
    predictions = estimator.fit_predict(x)
    anomaly_score = -estimator.decision_function(x)
    prediction_table = anomaly_prediction_table(
        x, source_rows, anomaly_score, {"one_class_svm_prediction": predictions}
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
        predictions=prediction_table,
        split_strategy="fit on all rows; anomaly scores are exploratory",
    )


def run_local_outlier_factor(
    features: FeatureTable,
    *,
    n_neighbors: int = 20,
    contamination: float | str = "auto",
) -> ModelEvaluation:
    """Fit Local Outlier Factor on numeric features and return anomaly scores."""
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")
    x, feature_columns, source_rows = standardized_anomaly_matrix(features)
    resolved_neighbors = min(n_neighbors, max(len(x) - 1, 1))
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LocalOutlierFactor(n_neighbors=resolved_neighbors, contamination=contamination),
            ),
        ]
    )
    predictions = estimator.fit_predict(x)
    model = estimator.named_steps["model"]
    anomaly_score = -model.negative_outlier_factor_
    prediction_table = anomaly_prediction_table(
        x, source_rows, anomaly_score, {"local_outlier_factor_prediction": predictions}
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
        predictions=prediction_table,
        split_strategy="fit on all rows; anomaly scores are exploratory",
    )


def run_pca_reconstruction(
    features: FeatureTable,
    *,
    n_components: int | float = 0.95,
) -> ModelEvaluation:
    """Score rows by PCA reconstruction error on standardized numeric features."""
    x, feature_columns, source_rows = standardized_anomaly_matrix(features)
    if isinstance(n_components, int) and n_components <= 0:
        raise ValueError("n_components must be positive.")
    if isinstance(n_components, float) and not 0.0 < n_components < 1.0:
        raise ValueError("float n_components must be in the interval (0, 1).")
    estimator = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n_components))])
    scaled = estimator.named_steps["scaler"].fit_transform(x)
    pca = estimator.named_steps["pca"]
    transformed = pca.fit_transform(scaled)
    reconstructed = pca.inverse_transform(transformed)
    anomaly_score = np.mean(np.square(scaled - reconstructed), axis=1)
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
        predictions=anomaly_prediction_table(x, source_rows, anomaly_score),
        split_strategy="fit on all rows; reconstruction errors are exploratory",
    )


def run_dbscan_outlier_scores(
    features: FeatureTable,
    *,
    eps: float = 1.5,
    min_samples: int = 5,
) -> ModelEvaluation:
    """Fit DBSCAN and score sparse or noise windows as clustering outliers."""
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive.")
    x, feature_columns, source_rows = standardized_anomaly_matrix(features)
    estimator = Pipeline(
        [("scaler", StandardScaler()), ("model", DBSCAN(eps=eps, min_samples=min_samples))]
    )
    labels = estimator.fit_predict(x)
    scaled = estimator.named_steps["scaler"].transform(x)
    anomaly_score = _dbscan_scores(scaled, labels, min_samples=min_samples)
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
        predictions=anomaly_prediction_table(
            x, source_rows, anomaly_score, {"dbscan_label": labels}
        ),
        split_strategy="fit on all rows; DBSCAN scores are exploratory and eps-sensitive",
    )


def run_robust_z_score(
    features: FeatureTable,
    *,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> ModelEvaluation:
    """Score unusually high feature values with summed positive robust z-scores."""
    x, selected_columns, source_rows = standardized_anomaly_matrix(
        features, feature_columns=feature_columns
    )
    median = x.median(axis=0)
    mad = (x - median).abs().median(axis=0)
    usable_scale = mad.where(np.isfinite(mad) & (mad != 0.0))
    robust_z = 0.6745 * (x - median) / usable_scale
    anomaly_score = robust_z.clip(lower=0.0).fillna(0.0).sum(axis=1).to_numpy()
    return ModelEvaluation(
        model_name="robust_z_score",
        task_type="unsupervised_anomaly_score",
        estimator=None,
        feature_columns=selected_columns,
        metrics={"n_samples": float(len(x))},
        predictions=anomaly_prediction_table(x, source_rows, anomaly_score),
        split_strategy="fit on all rows; robust positive z-scores are exploratory",
    )


def run_mahalanobis_distance(
    features: FeatureTable,
    *,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> ModelEvaluation:
    """Score rows by squared Mahalanobis distance from the feature centroid."""
    x, selected_columns, source_rows = standardized_anomaly_matrix(
        features, feature_columns=feature_columns
    )
    values = x.to_numpy(dtype=np.float64)
    centroid = values.mean(axis=0)
    covariance = np.atleast_2d(np.cov(values, rowvar=False))
    inverse_covariance = np.linalg.pinv(covariance)
    centered = values - centroid
    anomaly_score = np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered)
    return ModelEvaluation(
        model_name="mahalanobis_distance",
        task_type="unsupervised_anomaly_score",
        estimator={"centroid": centroid, "inverse_covariance": inverse_covariance},
        feature_columns=selected_columns,
        metrics={"n_samples": float(len(x))},
        predictions=anomaly_prediction_table(x, source_rows, anomaly_score),
        split_strategy="fit on all rows; Mahalanobis distances are exploratory",
    )


def run_robust_mahalanobis_distance(
    features: FeatureTable,
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
    x, selected_columns, source_rows = standardized_anomaly_matrix(
        features, feature_columns=feature_columns
    )
    values = x.to_numpy(dtype=np.float64)
    if np.linalg.matrix_rank(values - values.mean(axis=0)) < values.shape[1]:
        raise ValueError(
            "Robust Mahalanobis fitting failed; selected numeric features may be "
            "degenerate or insufficient for robust covariance estimation."
        )
    estimator = MinCovDet(support_fraction=support_fraction, random_state=random_state)
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
    return ModelEvaluation(
        model_name="robust_mahalanobis_distance",
        task_type="unsupervised_anomaly_score",
        estimator=estimator,
        feature_columns=selected_columns,
        metrics={
            "n_samples": float(len(x)),
            "support_fraction": "auto" if support_fraction is None else float(support_fraction),
            "support_count": float(np.sum(estimator.support_)),
        },
        predictions=anomaly_prediction_table(x, source_rows, anomaly_score),
        split_strategy="fit on all rows; robust Mahalanobis distances are exploratory",
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
