"""Tests for simple baseline modeling."""

import numpy as np
import pandas as pd
import pytest

from signal_processing_prep.features import FrequencyBand, SlidingWindowConfig, sliding_window_features
from signal_processing_prep.modeling import (
    anomaly_summary_text,
    run_dbscan_outlier_scores,
    run_isolation_forest,
    run_local_outlier_factor,
    run_mahalanobis_distance,
    run_one_class_svm,
    run_pca_reconstruction,
    run_robust_mahalanobis_distance,
    run_robust_z_score,
    run_supervised_baselines,
    top_anomalies,
)
from signal_processing_prep.records import SignalRecord


def _separable_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_name": [f"r{i}" for i in range(12)],
            "label": ["normal"] * 6 + ["fault"] * 6,
            "rms": [0.9, 1.0, 1.1, 0.95, 1.05, 1.0, 3.0, 3.1, 2.9, 3.2, 2.8, 3.05],
            "crest_factor": [1.5, 1.4, 1.6, 1.5, 1.4, 1.5, 4.0, 4.2, 3.8, 4.1, 3.9, 4.0],
            "dominant_frequency_hz": [50, 51, 49, 50, 52, 48, 200, 205, 198, 202, 199, 201],
        }
    )


def _window_features_with_outlier() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    base = rng.normal(0.0, 0.05, size=(30, 2))
    normal = np.column_stack([base, base[:, 0] + base[:, 1]])
    outlier = np.array([[0.0, 0.0, 3.0]])
    values = np.vstack([normal, outlier])
    starts = np.arange(values.shape[0], dtype=float) * 0.1
    return pd.DataFrame(
        {
            "record_name": ["synthetic"] * values.shape[0],
            "label": [None] * values.shape[0],
            "window_start_seconds": starts,
            "window_end_seconds": starts + 0.2,
            "window_center_seconds": starts + 0.1,
            "window_n_samples": [200.0] * values.shape[0],
            "rms": values[:, 0],
            "crest_factor": values[:, 1],
            "band_energy_high": values[:, 2],
        }
    )


def _synthetic_transient_window_features() -> pd.DataFrame:
    sampling_rate_hz = 1000.0
    time = np.arange(0.0, 4.0, 1.0 / sampling_rate_hz)
    values = 0.2 * np.sin(2.0 * np.pi * 50.0 * time)
    burst_mask = (time >= 2.0) & (time < 2.12)
    values[burst_mask] += 2.0 * np.sin(2.0 * np.pi * 300.0 * time[burst_mask])
    record = SignalRecord(
        values=values,
        sampling_rate_hz=sampling_rate_hz,
        name="synthetic_transient_record",
    )
    return sliding_window_features(
        record,
        SlidingWindowConfig(
            window_seconds=0.2,
            step_seconds=0.05,
            frequency_bands=(FrequencyBand("burst_band_250_350", 250.0, 350.0),),
            frequency_window="hann",
            normalize_frequency_window_power=True,
        ),
    )


def test_run_supervised_baselines_returns_metrics_and_predictions() -> None:
    """Supervised baselines train when labels are present."""
    results = run_supervised_baselines(_separable_features(), test_size=0.33, random_state=1)

    assert set(results) == {"logistic_regression", "random_forest"}
    for evaluation in results.values():
        assert evaluation.task_type == "supervised_classification"
        assert evaluation.metrics["n_train"] > 0
        assert evaluation.metrics["n_test"] > 0
        assert evaluation.confusion_matrix is not None
        assert evaluation.confusion_matrix.shape == (2, 2)
        assert {"row_index", "true_label", "predicted_label", "record_name", "label"}.issubset(
            evaluation.predictions.columns
        )
        assert evaluation.feature_importances is not None
        assert "group_shuffle_split" in evaluation.split_strategy


def test_run_supervised_baselines_rejects_missing_or_insufficient_labels() -> None:
    """Supervised modeling remains optional and explicit."""
    features = _separable_features().drop(columns=["label"])

    with pytest.raises(ValueError, match="Label column not found"):
        run_supervised_baselines(features)

    with pytest.raises(ValueError, match="At least two labels"):
        run_supervised_baselines(pd.DataFrame({"label": ["a"] * 4, "rms": [1, 2, 3, 4]}))

    with pytest.raises(ValueError, match="Each label"):
        run_supervised_baselines(
            pd.DataFrame({"label": ["a", "a", "a", "b"], "rms": [1, 2, 3, 4]})
        )

    with pytest.raises(ValueError, match="numeric feature"):
        run_supervised_baselines(
            pd.DataFrame({"label": ["a", "a", "b", "b"], "record_name": ["r1", "r2", "r3", "r4"]})
        )


def test_run_isolation_forest_returns_scores_not_classes() -> None:
    """Unsupervised modeling reports anomaly scores as exploratory outputs."""
    features = pd.DataFrame(
        {
            "record_name": ["a", "b", "c", "d", "e"],
            "window_start_seconds": [0.0, 1.0, 2.0, 3.0, 4.0],
            "rms": [1.0, 1.1, 0.9, 1.05, 10.0],
            "crest_factor": [1.5, 1.4, 1.6, 1.5, 8.0],
        }
    )

    evaluation = run_isolation_forest(features, contamination=0.2, random_state=2)

    assert evaluation.task_type == "unsupervised_anomaly_score"
    assert "anomaly_score" in evaluation.predictions.columns
    assert "anomaly_rank" in evaluation.predictions.columns
    assert "exploratory" in evaluation.split_strategy
    assert evaluation.metrics["n_samples"] == 5.0
    assert np.isfinite(evaluation.predictions["anomaly_score"]).all()
    assert "record_name" in evaluation.predictions.columns
    assert "window_start_seconds" in evaluation.predictions.columns
    assert "window_start_seconds" not in evaluation.feature_columns


def test_supervised_baselines_keep_grouped_windows_together() -> None:
    """Grouped evaluation keeps windows from the same source on one side of the split."""
    features = pd.DataFrame(
        {
            "source_name": ["n1", "n1", "n2", "n2", "f1", "f1", "f2", "f2"],
            "label": ["normal", "normal", "normal", "normal", "fault", "fault", "fault", "fault"],
            "rms": [1.0, 1.1, 0.9, 1.05, 3.0, 3.1, 2.9, 3.2],
            "crest_factor": [1.5, 1.4, 1.6, 1.5, 4.0, 4.2, 3.8, 4.1],
        }
    )

    results = run_supervised_baselines(
        features,
        group_column="source_name",
        test_size=0.5,
        random_state=3,
    )

    for evaluation in results.values():
        assert "group_column=source_name" in evaluation.split_strategy
        assert evaluation.predictions["source_name"].value_counts().eq(2).all()


def test_top_anomalies_and_summary_text_are_inspection_oriented() -> None:
    """Anomaly helpers rank candidates without claiming confirmed faults."""
    features = pd.DataFrame(
        {
            "record_name": ["a", "b", "c", "d", "e"],
            "label": ["unknown", "unknown", "unknown", "unknown", "unknown"],
            "rms": [1.0, 1.1, 0.9, 1.05, 10.0],
            "crest_factor": [1.5, 1.4, 1.6, 1.5, 8.0],
        }
    )
    evaluation = run_isolation_forest(features, contamination=0.2, random_state=2)

    top = top_anomalies(evaluation, n=2)
    summary = anomaly_summary_text(evaluation, top_n=2)

    assert top.shape[0] == 2
    assert top["anomaly_score"].is_monotonic_decreasing
    assert "candidates for inspection" in summary
    assert "not confirmed faults" in summary
    assert "without using labels" in summary


def test_run_isolation_forest_preserves_metadata_with_duplicate_index() -> None:
    """Metadata alignment is positional even when feature indexes are duplicated."""
    features = pd.DataFrame(
        {
            "record_name": ["a", "b", "c", "d", "e"],
            "rms": [1.0, 1.1, 0.9, 1.05, 10.0],
            "crest_factor": [1.5, 1.4, 1.6, 1.5, 8.0],
        },
        index=[0, 0, 1, 1, 2],
    )

    evaluation = run_isolation_forest(features, contamination=0.2, random_state=2)

    assert evaluation.predictions["record_name"].tolist() == ["a", "b", "c", "d", "e"]
    assert evaluation.predictions["row_index"].tolist() == [0, 0, 1, 1, 2]


def test_modeling_drops_rows_with_no_observed_numeric_features() -> None:
    """Rows with all-missing numeric features are not fabricated by median imputation."""
    features = pd.DataFrame(
        {
            "record_name": ["a", "b", "c", "d", "e"],
            "rms": [1.0, 1.1, None, 1.05, 10.0],
            "crest_factor": [1.5, 1.4, None, 1.5, 8.0],
        }
    )

    evaluation = run_isolation_forest(features, contamination=0.25, random_state=2)

    assert evaluation.metrics["n_samples"] == 4.0
    assert evaluation.predictions["record_name"].tolist() == ["a", "b", "d", "e"]


def test_run_isolation_forest_rejects_no_numeric_features() -> None:
    """Unsupervised baselines need numeric feature columns."""
    with pytest.raises(ValueError, match="numeric feature"):
        run_isolation_forest(pd.DataFrame({"record_name": ["a", "b"]}))

    supervised = run_supervised_baselines(_separable_features(), test_size=0.33, random_state=1)
    with pytest.raises(ValueError, match="unsupervised"):
        top_anomalies(supervised["random_forest"])


@pytest.mark.parametrize(
    ("runner", "model_name"),
    [
        (run_one_class_svm, "one_class_svm"),
        (run_local_outlier_factor, "local_outlier_factor"),
        (run_dbscan_outlier_scores, "dbscan_outlier_scores"),
        (run_robust_z_score, "robust_z_score"),
        (run_mahalanobis_distance, "mahalanobis_distance"),
        (
            lambda features: run_robust_mahalanobis_distance(
                features,
                feature_columns=["rms", "crest_factor"],
            ),
            "robust_mahalanobis_distance",
        ),
    ],
)
def test_feature_based_anomaly_detectors_return_scores_and_metadata(runner, model_name) -> None:
    """Feature-based detectors share the exploratory anomaly-score contract."""
    features = _window_features_with_outlier()

    evaluation = runner(features)

    assert evaluation.model_name == model_name
    assert evaluation.task_type == "unsupervised_anomaly_score"
    assert evaluation.metrics["n_samples"] == float(len(features))
    assert np.isfinite(evaluation.predictions["anomaly_score"]).all()
    assert set(["window_start_seconds", "window_end_seconds", "window_center_seconds"]).issubset(
        evaluation.predictions.columns
    )
    assert "anomaly_rank" in evaluation.predictions.columns
    assert "window_start_seconds" not in evaluation.feature_columns
    assert "window_n_samples" not in evaluation.feature_columns


def test_robust_z_score_matches_positive_mad_score_for_selected_features() -> None:
    """Selected feature scoring follows the notebook robust-positive-z formula."""
    features = pd.DataFrame(
        {
            "record_name": ["a", "b", "c", "d", "outlier"],
            "rms": [1.0, 2.0, 3.0, 4.0, 20.0],
            "crest_factor": [2.0, 2.0, 2.0, 2.0, 8.0],
            "unused": [0.0, 0.0, 0.0, 0.0, 1000.0],
        }
    )

    evaluation = run_robust_z_score(features, feature_columns=["rms", "crest_factor"])

    expected_rms = np.clip(0.6745 * (features["rms"] - 3.0) / 1.0, 0.0, None)
    assert evaluation.feature_columns == ("rms", "crest_factor")
    assert np.allclose(evaluation.predictions["anomaly_score"], expected_rms)
    assert top_anomalies(evaluation, n=1).loc[0, "record_name"] == "outlier"


def test_robust_z_score_zero_mad_feature_contributes_zero() -> None:
    """Constant robust-z features do not produce non-finite anomaly scores."""
    features = pd.DataFrame({"rms": [1.0, 1.0, 1.0], "crest_factor": [2.0, 2.0, 2.0]})

    evaluation = run_robust_z_score(features)

    assert np.isfinite(evaluation.predictions["anomaly_score"]).all()
    assert (evaluation.predictions["anomaly_score"] == 0.0).all()


def test_mahalanobis_distance_handles_collinear_features_and_ranks_outlier() -> None:
    """Pseudo-inverse covariance supports correlated feature sets."""
    features = pd.DataFrame(
        {
            "record_name": ["a", "b", "c", "d", "outlier"],
            "rms": [0.0, 0.1, -0.1, 0.05, 5.0],
            "twice_rms": [0.0, 0.2, -0.2, 0.1, 10.0],
        }
    )

    evaluation = run_mahalanobis_distance(features)

    assert np.isfinite(evaluation.predictions["anomaly_score"]).all()
    assert top_anomalies(evaluation, n=1).loc[0, "record_name"] == "outlier"


def test_robust_mahalanobis_distance_ranks_small_displaced_minority() -> None:
    """Robust covariance highlights a displaced minority without fitting to it."""
    rng = np.random.default_rng(8)
    normal = rng.normal(0.0, 0.1, size=(40, 2))
    anomalies = np.array([[3.0, 3.0], [3.2, 2.8]])
    values = np.vstack([normal, anomalies])
    features = pd.DataFrame({"rms": values[:, 0], "crest_factor": values[:, 1]})

    evaluation = run_robust_mahalanobis_distance(
        features,
        support_fraction=0.75,
        random_state=0,
    )

    assert evaluation.metrics["support_fraction"] == 0.75
    assert 0.0 < evaluation.metrics["support_count"] < evaluation.metrics["n_samples"]
    assert set(top_anomalies(evaluation, n=2)["row_index"]) == {40, 41}


@pytest.mark.parametrize(
    "runner",
    [run_robust_z_score, run_mahalanobis_distance, run_robust_mahalanobis_distance],
)
def test_selected_feature_anomaly_detectors_validate_columns_and_rows(runner) -> None:
    """Explicit scoring columns must exist and include enough usable numeric rows."""
    features = pd.DataFrame({"rms": [1.0, 2.0], "state": ["normal", "fault"]})

    with pytest.raises(ValueError, match="not found"):
        runner(features, feature_columns=["missing"])
    with pytest.raises(ValueError, match="numeric feature"):
        runner(features, feature_columns=["state"])
    with pytest.raises(ValueError, match="At least two"):
        runner(pd.DataFrame({"rms": [1.0]}), feature_columns=["rms"])


def test_robust_mahalanobis_validates_support_and_degenerate_features() -> None:
    """Robust covariance configuration and fitting failures remain explicit."""
    features = pd.DataFrame({"rms": [1.0, 1.0, 1.0, 1.0], "crest_factor": [2.0] * 4})

    with pytest.raises(ValueError, match="support_fraction"):
        run_robust_mahalanobis_distance(features, support_fraction=0.0)
    with pytest.raises(ValueError, match="degenerate or insufficient"):
        run_robust_mahalanobis_distance(features)


def test_pca_reconstruction_ranks_clear_feature_outlier_near_top() -> None:
    """PCA reconstruction error highlights a clear transient-like feature outlier."""
    features = _window_features_with_outlier()

    evaluation = run_pca_reconstruction(features, n_components=1)
    top = top_anomalies(evaluation, n=3)

    assert evaluation.model_name == "pca_reconstruction"
    assert np.isfinite(evaluation.predictions["anomaly_score"]).all()
    assert "exploratory" in evaluation.split_strategy
    assert int(features.index[-1]) in top["row_index"].tolist()
    assert "window_start_seconds" not in evaluation.feature_columns


def test_pca_reconstruction_rejects_float_one_component_request() -> None:
    """PCA variance-ratio requests must stay inside scikit-learn's valid interval."""
    features = _window_features_with_outlier()

    with pytest.raises(ValueError, match="interval"):
        run_pca_reconstruction(features, n_components=1.0)


def test_dbscan_all_noise_scores_use_neighbor_radius_fallback() -> None:
    """All-noise DBSCAN output still gets a meaningful distance-based ranking."""
    features = pd.DataFrame(
        {
            "rms": [0.0, 0.1, 0.2, 5.0],
            "crest_factor": [0.0, 0.1, 0.2, 8.0],
        }
    )

    evaluation = run_dbscan_outlier_scores(features, eps=0.01, min_samples=2)

    assert evaluation.metrics["n_clusters"] == 0.0
    assert evaluation.metrics["noise_fraction"] == 1.0
    assert evaluation.predictions["anomaly_score"].nunique() > 1
    assert top_anomalies(evaluation, n=1).loc[0, "row_index"] == 3


def test_pca_reconstruction_ranks_synthetic_transient_window_near_top() -> None:
    """A real transient record remains detectable after sliding-window feature extraction."""
    features = _synthetic_transient_window_features()

    evaluation = run_pca_reconstruction(features, n_components=0.8)
    top = top_anomalies(evaluation, n=5)

    overlaps_burst = (top["window_start_seconds"] <= 2.12) & (top["window_end_seconds"] >= 2.0)
    assert overlaps_burst.any()
    assert np.isfinite(evaluation.predictions["anomaly_score"]).all()
    assert "window_center_seconds" in evaluation.predictions.columns


def test_feature_based_anomaly_detectors_reject_too_few_rows() -> None:
    """Detectors fail clearly when unsupervised scoring has too little data."""
    features = pd.DataFrame({"record_name": ["a"], "rms": [1.0]})

    with pytest.raises(ValueError, match="At least two"):
        run_one_class_svm(features)

    with pytest.raises(ValueError, match="At least two"):
        run_local_outlier_factor(features)

    with pytest.raises(ValueError, match="At least two"):
        run_pca_reconstruction(features)

    with pytest.raises(ValueError, match="At least two"):
        run_dbscan_outlier_scores(features)
