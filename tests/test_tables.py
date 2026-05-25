"""Tests for typed analysis artifacts."""

import pandas as pd
import pytest

from signal_processing_prep.artifacts import FeatureTable, PredictionTable, QualityTable


def test_feature_table_owns_features_and_returns_copy_views() -> None:
    table = FeatureTable.from_dataframe(
        pd.DataFrame({"record_name": ["a", "b"], "sampling_rate_hz": [100.0, 100.0], "rms": [1.0, 2.0]}),
        feature_columns=("rms",),
    )
    matrix, columns = table.numeric_matrix()
    frame = table.to_dataframe()
    frame.loc[0, "rms"] = 999.0

    assert columns == ("rms",)
    assert matrix.columns.tolist() == ["rms"]
    assert table.to_dataframe().loc[0, "rms"] == 1.0


def test_feature_table_validates_columns_and_group_resolution() -> None:
    table = FeatureTable.from_dataframe(
        pd.DataFrame({"source_name": ["s", "s"], "rms": [1.0, 2.0]}),
        feature_columns=("rms",),
    )
    assert table.resolve_group_column() == "source_name"
    with pytest.raises(ValueError, match="explicitly declared"):
        FeatureTable.from_dataframe(
            pd.DataFrame({"rms": [1.0], "is_anomalous": [True]})
        )
    with pytest.raises(ValueError, match="reserved context"):
        FeatureTable.from_dataframe(
            pd.DataFrame({"rms": [1.0], "is_anomalous": [True]}),
            feature_columns=("rms", "is_anomalous"),
        )
    with pytest.raises(ValueError, match="Feature columns not found"):
        FeatureTable.from_dataframe(pd.DataFrame({"rms": [1.0]}), feature_columns=["missing"])


def test_quality_and_prediction_artifacts_validate_boundary_schemas() -> None:
    quality = QualityTable(pd.DataFrame({"record_name": ["a"], "issues": [""]}))
    assert not quality.empty
    prediction = PredictionTable.anomaly_scores(
        pd.DataFrame({"row_index": [0], "anomaly_score": [1.0], "anomaly_rank": [1]})
    )
    assert prediction.to_dataframe()["anomaly_score"].tolist() == [1.0]
    with pytest.raises(ValueError, match="Anomaly prediction columns"):
        PredictionTable.anomaly_scores(pd.DataFrame({"anomaly_score": [1.0]}))


def test_prediction_artifact_validates_rank_and_feature_alignment() -> None:
    features = FeatureTable.from_dataframe(
        pd.DataFrame({"record_name": ["a", "b"], "rms": [1.0, 2.0]}),
        feature_columns=("rms",),
    )
    valid = PredictionTable.anomaly_scores(
        pd.DataFrame(
            {
                "row_index": [1, 0],
                "record_name": ["b", "a"],
                "anomaly_score": [2.0, 1.0],
                "anomaly_rank": [1, 2],
            }
        )
    )
    valid.validate_alignment(features)

    with pytest.raises(ValueError, match="not aligned"):
        PredictionTable.anomaly_scores(
            pd.DataFrame(
                {
                    "row_index": [0],
                    "record_name": ["unrelated"],
                    "anomaly_score": [1.0],
                    "anomaly_rank": [1],
                }
            )
        ).validate_alignment(features)
    with pytest.raises(ValueError, match="ranks"):
        PredictionTable.anomaly_scores(
            pd.DataFrame(
                {"row_index": [0, 1], "anomaly_score": [1.0, 2.0], "anomaly_rank": [1, 2]}
            )
        )
