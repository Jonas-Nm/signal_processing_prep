"""Public facade for simple baseline modeling workflows."""

from signal_processing_prep._modeling_anomaly import (
    run_dbscan_outlier_scores,
    run_isolation_forest,
    run_local_outlier_factor,
    run_mahalanobis_distance,
    run_one_class_svm,
    run_pca_reconstruction,
    run_robust_mahalanobis_distance,
    run_robust_z_score,
)
from signal_processing_prep._modeling_common import (
    ModelEvaluation,
    anomaly_summary_text,
    top_anomalies,
)
from signal_processing_prep._modeling_supervised import run_supervised_baselines

__all__ = [
    "ModelEvaluation",
    "anomaly_summary_text",
    "run_dbscan_outlier_scores",
    "run_isolation_forest",
    "run_local_outlier_factor",
    "run_mahalanobis_distance",
    "run_one_class_svm",
    "run_pca_reconstruction",
    "run_robust_mahalanobis_distance",
    "run_robust_z_score",
    "run_supervised_baselines",
    "top_anomalies",
]
