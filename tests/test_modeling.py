"""Tests for simple baseline modeling."""

import numpy as np
import pandas as pd
import pytest

from signal_processing_prep.modeling import (
    anomaly_summary_text,
    run_isolation_forest,
    run_supervised_baselines,
    top_anomalies,
)


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
    assert "exploratory" in evaluation.split_strategy
    assert evaluation.metrics["n_samples"] == 5.0
    assert np.isfinite(evaluation.predictions["anomaly_score"]).all()
    assert "record_name" in evaluation.predictions.columns
    assert "window_start_seconds" in evaluation.predictions.columns


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
