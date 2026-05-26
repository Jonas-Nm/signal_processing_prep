"""Tests for configured modeling strategies and typed artifacts."""

import numpy as np
import pandas as pd
import pytest

from signal_processing_prep.artifacts import FeatureTable, PredictionTable
from signal_processing_prep.features import FeatureExtractor, FrequencyBand, SlidingWindowConfig
from signal_processing_prep.modeling import (
    DbscanOutlierScorer,
    IsolationForestScorer,
    LocalOutlierFactorScorer,
    MahalanobisScorer,
    ModelEvaluation,
    OneClassSvmScorer,
    PcaReconstructionScorer,
    RobustMahalanobisScorer,
    RobustZScoreScorer,
    SupervisedBaselineSuite,
    anomaly_summary_text,
    consensus_candidate_regions,
    evaluate_candidate_region_overlap,
    group_candidate_regions,
    top_anomalies,
)
from signal_processing_prep.records import AnnotationInterval, SignalRecord
from signal_processing_prep.preprocessing import segment_signal
from signal_processing_prep.synthetic import sine_wave


def _table(frame: pd.DataFrame, *columns: str) -> FeatureTable:
    return FeatureTable.from_dataframe(frame, feature_columns=columns)


def _separable_features() -> FeatureTable:
    return _table(
        pd.DataFrame(
            {
                "source_name": [f"n{i}" for i in range(4)] + [f"f{i}" for i in range(4)],
                "label": ["normal"] * 4 + ["fault"] * 4,
                "rms": [0.9, 1.0, 1.1, 0.95, 3.0, 3.1, 2.9, 3.2],
                "crest_factor": [1.5, 1.4, 1.6, 1.5, 4.0, 4.2, 3.8, 4.1],
            }
        ),
        "rms",
        "crest_factor",
    )


def _window_features_with_outlier() -> FeatureTable:
    rng = np.random.default_rng(4)
    base = rng.normal(0.0, 0.05, size=(30, 2))
    values = np.vstack([np.column_stack([base, base[:, 0] + base[:, 1]]), [0.0, 0.0, 3.0]])
    starts = np.arange(len(values), dtype=float) * 0.1
    return _table(
        pd.DataFrame(
            {
                "record_name": ["synthetic"] * len(values),
                "window_start_seconds": starts,
                "window_end_seconds": starts + 0.2,
                "window_center_seconds": starts + 0.1,
                "window_n_samples": [200.0] * len(values),
                "rms": values[:, 0],
                "crest_factor": values[:, 1],
                "band_energy_high": values[:, 2],
            }
        ),
        "rms",
        "crest_factor",
        "band_energy_high",
    )


def _synthetic_transient_window_features() -> FeatureTable:
    sampling_rate_hz = 1000.0
    time = np.arange(0.0, 4.0, 1.0 / sampling_rate_hz)
    values = 0.2 * np.sin(2.0 * np.pi * 50.0 * time)
    burst = (time >= 2.0) & (time < 2.12)
    values[burst] += 2.0 * np.sin(2.0 * np.pi * 300.0 * time[burst])
    record = SignalRecord(values, sampling_rate_hz=sampling_rate_hz, name="transient")
    return FeatureExtractor().extract_windows(
        record,
        SlidingWindowConfig(
            window_seconds=0.2,
            step_seconds=0.05,
            frequency_bands=(FrequencyBand("burst_band_250_350", 250.0, 350.0),),
        ),
    )


def _scored_windows(
    intervals: list[tuple[float, float]],
    scores: list[float],
    *,
    record_names: list[str] | None = None,
) -> ModelEvaluation:
    frame = pd.DataFrame(
        {
            "row_index": np.arange(len(intervals), dtype=int),
            "window_start_seconds": [interval[0] for interval in intervals],
            "window_end_seconds": [interval[1] for interval in intervals],
            "window_center_seconds": [0.5 * sum(interval) for interval in intervals],
            "anomaly_score": scores,
        }
    )
    if record_names is not None:
        frame["record_name"] = record_names
    frame["anomaly_rank"] = frame["anomaly_score"].rank(method="first", ascending=False).astype(int)
    return ModelEvaluation(
        model_name="test_scores",
        task_type="unsupervised_anomaly_score",
        estimator=None,
        feature_columns=("rms",),
        metrics={"n_samples": float(len(frame))},
        predictions=PredictionTable.anomaly_scores(frame),
        split_strategy="test",
    )


def test_supervised_suite_returns_typed_predictions_and_metrics() -> None:
    results = SupervisedBaselineSuite(test_size=0.5, random_state=1).evaluate(_separable_features())

    assert set(results) == {"logistic_regression", "random_forest"}
    for evaluation in results.values():
        assert isinstance(evaluation.predictions, PredictionTable)
        assert evaluation.task_type == "supervised_classification"
        assert evaluation.metrics["n_train"] > 0
        assert evaluation.confusion_matrix is not None
        assert {"true_label", "predicted_label", "source_name"}.issubset(
            evaluation.prediction_frame.columns
        )
        assert "group_shuffle_split" in evaluation.split_strategy


def test_supervised_suite_rejects_invalid_label_contracts() -> None:
    with pytest.raises(ValueError, match="Label column not found"):
        SupervisedBaselineSuite().evaluate(_table(pd.DataFrame({"rms": [1, 2]}), "rms"))
    with pytest.raises(ValueError, match="At least two labels"):
        SupervisedBaselineSuite().evaluate(
            _table(pd.DataFrame({"label": ["normal"] * 4, "rms": [1, 2, 3, 4]}), "rms")
        )


def test_supervised_suite_keeps_windows_from_each_source_together() -> None:
    frame = pd.DataFrame(
        {
            "source_name": ["n1", "n1", "n2", "n2", "f1", "f1", "f2", "f2"],
            "label": ["normal"] * 4 + ["fault"] * 4,
            "rms": [1.0, 1.1, 0.9, 1.05, 3.0, 3.1, 2.9, 3.2],
            "crest_factor": [1.5, 1.4, 1.6, 1.5, 4.0, 4.2, 3.8, 4.1],
        }
    )

    results = SupervisedBaselineSuite(group_column="source_name", test_size=0.5, random_state=3).evaluate(
        _table(frame, "rms", "crest_factor")
    )

    for evaluation in results.values():
        assert "group_column=source_name" in evaluation.split_strategy
        assert evaluation.prediction_frame["source_name"].value_counts().eq(2).all()


def test_segmented_feature_artifact_preserves_groups_for_model_evaluation() -> None:
    windows = []
    for label, frequency, prefix in (("normal", 10.0, "n"), ("fault", 80.0, "f")):
        for index in range(2):
            windows.extend(
                segment_signal(
                    sine_wave(frequency_hz=frequency + index, duration_seconds=2.0, label=label, name=f"{prefix}{index}"),
                    window_seconds=1.0,
                )
            )
    features = FeatureExtractor().extract_records(windows)
    results = SupervisedBaselineSuite(group_column="source_name", test_size=0.5).evaluate(features)

    assert features.to_dataframe().groupby("source_name").size().eq(2).all()
    assert all("group_column=source_name" in result.split_strategy for result in results.values())


def test_isolation_forest_returns_artifact_and_excludes_window_identity_columns() -> None:
    evaluation = IsolationForestScorer(contamination=0.2, random_state=2).score(_window_features_with_outlier())
    frame = evaluation.prediction_frame

    assert isinstance(evaluation.predictions, PredictionTable)
    assert evaluation.task_type == "unsupervised_anomaly_score"
    assert {"anomaly_score", "anomaly_rank", "window_start_seconds"}.issubset(frame.columns)
    assert np.isfinite(frame["anomaly_score"]).all()
    assert "window_start_seconds" not in evaluation.feature_columns
    assert "window_n_samples" not in evaluation.feature_columns


@pytest.mark.parametrize(
    ("scorer", "model_name"),
    [
        (OneClassSvmScorer(), "one_class_svm"),
        (LocalOutlierFactorScorer(n_neighbors=10), "local_outlier_factor"),
        (DbscanOutlierScorer(eps=1.5, min_samples=8), "dbscan_outlier_scores"),
        (RobustZScoreScorer(), "robust_z_score"),
        (MahalanobisScorer(), "mahalanobis_distance"),
    ],
)
def test_anomaly_scorers_share_typed_prediction_contract(scorer, model_name) -> None:
    evaluation = scorer.score(_window_features_with_outlier())

    assert evaluation.model_name == model_name
    assert evaluation.metrics["n_samples"] == 31.0
    assert np.isfinite(evaluation.prediction_frame["anomaly_score"]).all()
    assert "anomaly_rank" in evaluation.prediction_frame


def test_robust_z_score_supports_explicit_calculated_features() -> None:
    frame = pd.DataFrame(
        {
            "record_name": ["a", "b", "c", "d", "outlier"],
            "rms": [1.0, 2.0, 3.0, 4.0, 20.0],
            "crest_factor": [2.0, 2.0, 2.0, 2.0, 8.0],
            "unused": [0.0, 0.0, 0.0, 0.0, 1000.0],
        }
    )
    evaluation = RobustZScoreScorer(feature_columns=("rms", "crest_factor")).score(
        _table(frame, "rms", "crest_factor", "unused")
    )
    expected = np.clip(0.6745 * (frame["rms"] - 3.0), 0.0, None)

    assert evaluation.feature_columns == ("rms", "crest_factor")
    assert np.allclose(evaluation.prediction_frame["anomaly_score"], expected)
    assert top_anomalies(evaluation, n=1).loc[0, "record_name"] == "outlier"


def test_mahalanobis_and_robust_variant_rank_clear_outliers() -> None:
    rng = np.random.default_rng(8)
    values = np.vstack([rng.normal(0.0, 0.1, size=(40, 2)), [[3.0, 3.0], [3.2, 2.8]]])
    table = _table(pd.DataFrame({"rms": values[:, 0], "crest_factor": values[:, 1]}), "rms", "crest_factor")

    robust = RobustMahalanobisScorer(support_fraction=0.75, random_state=0).score(table)
    ordinary = MahalanobisScorer().score(table)

    assert set(top_anomalies(robust, n=2)["row_index"]) == {40, 41}
    assert np.isfinite(ordinary.prediction_frame["anomaly_score"]).all()


def test_pca_reconstruction_detects_localized_synthetic_transient() -> None:
    evaluation = PcaReconstructionScorer(n_components=0.8).score(_synthetic_transient_window_features())
    top = top_anomalies(evaluation, n=5)

    overlap = (top["window_start_seconds"] <= 2.12) & (top["window_end_seconds"] >= 2.0)
    assert overlap.any()
    assert "window_center_seconds" in evaluation.prediction_frame


def test_anomaly_summary_is_inspection_oriented() -> None:
    evaluation = IsolationForestScorer(contamination=0.2, random_state=2).score(_window_features_with_outlier())
    summary = anomaly_summary_text(evaluation, top_n=2)

    assert "candidates for inspection" in summary
    assert "not confirmed faults" in summary
    assert "without using labels" in summary


def test_group_candidate_regions_merges_overlapping_top_windows() -> None:
    evaluation = _scored_windows(
        [(0.0, 0.2), (0.1, 0.3), (1.0, 1.2)],
        [10.0, 9.0, 8.0],
    )

    regions = group_candidate_regions(evaluation, top_n=3)

    assert len(regions) == 2
    assert regions.loc[0, "region_start_seconds"] == pytest.approx(0.0)
    assert regions.loc[0, "region_end_seconds"] == pytest.approx(0.3)
    assert regions.loc[0, "window_count"] == 2
    assert regions.loc[0, "peak_anomaly_score"] == pytest.approx(10.0)
    assert regions.loc[1, "region_start_seconds"] == pytest.approx(1.0)


def test_group_candidate_regions_honors_top_n_and_merge_gap() -> None:
    evaluation = _scored_windows(
        [(0.0, 0.1), (0.125, 0.225), (1.0, 1.1)],
        [3.0, 2.0, 1.0],
    )

    separate = group_candidate_regions(evaluation, top_n=2)
    joined = group_candidate_regions(evaluation, top_n=2, merge_gap_seconds=0.03)

    assert len(separate) == 2
    assert len(joined) == 1
    assert joined.loc[0, "window_count"] == 2
    assert joined.loc[0, "region_end_seconds"] == pytest.approx(0.225)


def test_group_candidate_regions_does_not_merge_different_records_at_same_time() -> None:
    evaluation = _scored_windows(
        [(1.0, 1.2), (1.0, 1.2)],
        [4.0, 3.0],
        record_names=["run_a", "run_b"],
    )

    regions = group_candidate_regions(evaluation, top_n=2)

    assert len(regions) == 2
    assert regions["record_name"].tolist() == ["run_a", "run_b"]
    assert regions["window_count"].tolist() == [1, 1]


def test_group_candidate_regions_requires_window_timing_columns() -> None:
    predictions = PredictionTable.anomaly_scores(
        pd.DataFrame({"row_index": [0, 1], "anomaly_score": [2.0, 1.0], "anomaly_rank": [1, 2]})
    )
    evaluation = ModelEvaluation(
        model_name="scores",
        task_type="unsupervised_anomaly_score",
        estimator=None,
        feature_columns=("rms",),
        metrics={},
        predictions=predictions,
        split_strategy="test",
    )

    with pytest.raises(ValueError, match="window timing"):
        group_candidate_regions(evaluation)


def test_consensus_regions_count_each_detector_once_and_rank_agreement_first() -> None:
    evaluations = {
        "first": _scored_windows([(0.0, 0.1), (0.3, 0.4)], [5.0, 4.0]),
        "bridge": _scored_windows([(0.05, 0.35)], [6.0]),
        "isolated": _scored_windows([(2.0, 2.2)], [9.0]),
    }

    regions = consensus_candidate_regions(evaluations, top_n_per_method=2)

    assert regions.loc[0, "method_count"] == 2
    assert regions.loc[0, "methods"] == ("bridge", "first")
    assert regions.loc[0, "contributing_region_count"] == 3
    assert regions.loc[0, "region_start_seconds"] == pytest.approx(0.0)
    assert regions.loc[0, "region_end_seconds"] == pytest.approx(0.4)
    assert regions.loc[1, "method_count"] == 1


def test_consensus_candidate_regions_keeps_different_records_separate() -> None:
    evaluations = {
        "first": _scored_windows([(1.0, 1.2)], [4.0], record_names=["run_a"]),
        "second": _scored_windows([(1.0, 1.2)], [3.0], record_names=["run_b"]),
    }

    regions = consensus_candidate_regions(evaluations, top_n_per_method=1)

    assert len(regions) == 2
    assert set(regions["record_name"]) == {"run_a", "run_b"}
    assert regions["method_count"].tolist() == [1, 1]


def test_evaluate_candidate_region_overlap_reports_capture_and_non_overlap() -> None:
    regions = pd.DataFrame(
        {
            "region_rank": [1, 2],
            "region_start_seconds": [1.0, 3.0],
            "region_end_seconds": [2.0, 4.0],
        }
    )
    known = [AnnotationInterval(1.5, 2.5, kind="injected_event")]

    compared = evaluate_candidate_region_overlap(regions, known)

    assert compared.loc[0, "overlap_seconds"] == pytest.approx(0.5)
    assert compared.loc[0, "candidate_overlap_fraction"] == pytest.approx(0.5)
    assert compared.loc[0, "known_interval_capture_fraction"] == pytest.approx(0.5)
    assert bool(compared.loc[0, "overlaps_known_interval"]) is True
    assert compared.loc[0, "matched_interval_kind"] == "injected_event"
    assert compared.loc[1, "overlap_seconds"] == pytest.approx(0.0)
    assert bool(compared.loc[1, "overlaps_known_interval"]) is False


def test_evaluate_candidate_region_overlap_rejects_unscoped_multi_record_truth() -> None:
    regions = pd.DataFrame(
        {
            "record_name": ["run_a", "run_b"],
            "region_start_seconds": [1.0, 1.0],
            "region_end_seconds": [2.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="one record at a time"):
        evaluate_candidate_region_overlap(
            regions,
            [AnnotationInterval(1.5, 1.75, kind="injected_event")],
        )


def test_scorers_validate_requested_features_and_insufficient_rows() -> None:
    table = _table(pd.DataFrame({"rms": [1.0, 2.0], "state": ["normal", "fault"]}), "rms")
    with pytest.raises(ValueError, match="not found"):
        RobustZScoreScorer(feature_columns=("missing",)).score(table)
    with pytest.raises(ValueError, match="At least two"):
        PcaReconstructionScorer().score(_table(pd.DataFrame({"rms": [1.0]}), "rms"))
    with pytest.raises(ValueError, match="support_fraction"):
        RobustMahalanobisScorer(support_fraction=0.0)


def test_invalid_scorer_and_suite_configuration_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="n_neighbors"):
        LocalOutlierFactorScorer(n_neighbors=0)
    with pytest.raises(ValueError, match="contamination"):
        IsolationForestScorer(contamination=0.9)
    with pytest.raises(ValueError, match="n_components"):
        PcaReconstructionScorer(n_components=1.0)
    with pytest.raises(ValueError, match="test_size"):
        SupervisedBaselineSuite(test_size=1.0)
