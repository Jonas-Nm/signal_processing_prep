"""Configured model strategies for extracted feature artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from signal_processing_prep._modeling_anomaly import (
    run_dbscan_outlier_scores as _run_dbscan_outlier_scores,
    run_isolation_forest as _run_isolation_forest,
    run_local_outlier_factor as _run_local_outlier_factor,
    run_mahalanobis_distance as _run_mahalanobis_distance,
    run_one_class_svm as _run_one_class_svm,
    run_pca_reconstruction as _run_pca_reconstruction,
    run_robust_mahalanobis_distance as _run_robust_mahalanobis_distance,
    run_robust_z_score as _run_robust_z_score,
)
from signal_processing_prep._modeling_common import (
    ModelEvaluation,
    anomaly_summary_text,
    top_anomalies,
)
from signal_processing_prep._modeling_regions import (
    consensus_candidate_regions,
    evaluate_candidate_region_overlap,
    group_candidate_regions,
)
from signal_processing_prep._modeling_supervised import run_supervised_baselines as _run_supervised
from signal_processing_prep.artifacts import FeatureTable
from signal_processing_prep.errors import ConfigurationError


class AnomalyScorer(Protocol):
    """Replaceable anomaly-scoring strategy."""

    def score(self, features: FeatureTable) -> ModelEvaluation:
        """Score feature rows without asserting confirmed faults."""
        ...


@dataclass(frozen=True)
class IsolationForestScorer:
    """Configured Isolation Forest scorer."""

    contamination: float | str = "auto"
    random_state: int = 0

    def __post_init__(self) -> None:
        """Validate global scoring policy before records are analyzed."""
        _validate_contamination(self.contamination)

    def score(self, features: FeatureTable) -> ModelEvaluation:
        return _run_isolation_forest(
            features, contamination=self.contamination, random_state=self.random_state
        )


@dataclass(frozen=True)
class OneClassSvmScorer:
    """Configured one-class SVM scorer."""

    kernel: str = "rbf"
    gamma: str | float = "scale"
    nu: float = 0.05

    def __post_init__(self) -> None:
        """Validate global one-class SVM policy."""
        if not 0.0 < self.nu <= 1.0:
            raise ConfigurationError("nu must be in the interval (0, 1].")
        if isinstance(self.gamma, float | int) and float(self.gamma) <= 0.0:
            raise ConfigurationError("gamma must be positive when numeric.")

    def score(self, features: FeatureTable) -> ModelEvaluation:
        return _run_one_class_svm(features, kernel=self.kernel, gamma=self.gamma, nu=self.nu)


@dataclass(frozen=True)
class LocalOutlierFactorScorer:
    """Configured local-outlier-factor scorer."""

    n_neighbors: int = 20
    contamination: float | str = "auto"

    def __post_init__(self) -> None:
        """Validate global local-neighborhood policy."""
        if self.n_neighbors <= 0:
            raise ConfigurationError("n_neighbors must be positive.")
        _validate_contamination(self.contamination)

    def score(self, features: FeatureTable) -> ModelEvaluation:
        return _run_local_outlier_factor(
            features, n_neighbors=self.n_neighbors, contamination=self.contamination
        )


@dataclass(frozen=True)
class PcaReconstructionScorer:
    """Configured PCA reconstruction-error scorer."""

    n_components: int | float = 0.95

    def __post_init__(self) -> None:
        """Validate global reconstruction policy."""
        if isinstance(self.n_components, int) and self.n_components <= 0:
            raise ConfigurationError("n_components must be positive.")
        if isinstance(self.n_components, float) and not 0.0 < self.n_components < 1.0:
            raise ConfigurationError("float n_components must be in the interval (0, 1).")

    def score(self, features: FeatureTable) -> ModelEvaluation:
        return _run_pca_reconstruction(features, n_components=self.n_components)


@dataclass(frozen=True)
class DbscanOutlierScorer:
    """Configured DBSCAN outlier scorer."""

    eps: float = 1.5
    min_samples: int = 5

    def __post_init__(self) -> None:
        """Validate global density-clustering policy."""
        if self.eps <= 0:
            raise ConfigurationError("eps must be positive.")
        if self.min_samples <= 0:
            raise ConfigurationError("min_samples must be positive.")

    def score(self, features: FeatureTable) -> ModelEvaluation:
        return _run_dbscan_outlier_scores(features, eps=self.eps, min_samples=self.min_samples)


@dataclass(frozen=True)
class RobustZScoreScorer:
    """Configured robust positive-z anomaly scorer."""

    feature_columns: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate any explicit calculated-feature selection."""
        _validate_selected_columns(self.feature_columns)

    def score(self, features: FeatureTable) -> ModelEvaluation:
        return _run_robust_z_score(features, feature_columns=self.feature_columns)


@dataclass(frozen=True)
class MahalanobisScorer:
    """Configured Mahalanobis-distance scorer."""

    feature_columns: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate any explicit calculated-feature selection."""
        _validate_selected_columns(self.feature_columns)

    def score(self, features: FeatureTable) -> ModelEvaluation:
        return _run_mahalanobis_distance(features, feature_columns=self.feature_columns)


@dataclass(frozen=True)
class RobustMahalanobisScorer:
    """Configured robust covariance Mahalanobis scorer."""

    feature_columns: tuple[str, ...] | None = None
    support_fraction: float | None = None
    random_state: int | None = 0

    def __post_init__(self) -> None:
        """Validate global robust-covariance policy."""
        _validate_selected_columns(self.feature_columns)
        if self.support_fraction is not None and not 0.0 < self.support_fraction <= 1.0:
            raise ConfigurationError("support_fraction must be in the interval (0, 1].")

    def score(self, features: FeatureTable) -> ModelEvaluation:
        return _run_robust_mahalanobis_distance(
            features,
            feature_columns=self.feature_columns,
            support_fraction=self.support_fraction,
            random_state=self.random_state,
        )


@dataclass(frozen=True)
class SupervisedBaselineSuite:
    """Configured interpretable supervised baseline suite."""

    label_column: str = "label"
    group_column: str | None = None
    test_size: float = 0.3
    random_state: int = 0

    def __post_init__(self) -> None:
        """Validate global evaluation policy before feature rows are inspected."""
        if not 0.0 < self.test_size < 1.0:
            raise ConfigurationError("test_size must be in the interval (0, 1).")

    def evaluate(self, features: FeatureTable) -> dict[str, ModelEvaluation]:
        return _run_supervised(
            features,
            label_column=self.label_column,
            group_column=self.group_column,
            test_size=self.test_size,
            random_state=self.random_state,
        )


def _validate_contamination(value: float | str) -> None:
    if value != "auto" and not (
        isinstance(value, float | int) and 0.0 < float(value) <= 0.5
    ):
        raise ConfigurationError("contamination must be 'auto' or in the interval (0, 0.5].")


def _validate_selected_columns(columns: tuple[str, ...] | None) -> None:
    if columns is not None and not columns:
        raise ConfigurationError("feature_columns must not be empty when provided.")


__all__ = [
    "ModelEvaluation",
    "AnomalyScorer",
    "IsolationForestScorer",
    "OneClassSvmScorer",
    "LocalOutlierFactorScorer",
    "PcaReconstructionScorer",
    "DbscanOutlierScorer",
    "RobustZScoreScorer",
    "MahalanobisScorer",
    "RobustMahalanobisScorer",
    "SupervisedBaselineSuite",
    "anomaly_summary_text",
    "consensus_candidate_regions",
    "evaluate_candidate_region_overlap",
    "group_candidate_regions",
    "top_anomalies",
]
