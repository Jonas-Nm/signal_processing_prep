"""Tests for typed high-level workflow orchestration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signal_processing_prep._modeling_common import ModelEvaluation
from signal_processing_prep.artifacts import FeatureTable, PredictionTable, QualityTable
from signal_processing_prep.config import AnalysisConfig, FilteringConfig, ModelingConfig
from signal_processing_prep.errors import ConfigurationError, RecordDataError
from signal_processing_prep.frequency_domain import band_energy
from signal_processing_prep.pipeline import SignalAnalysisPipeline, run_synthetic_analysis
from signal_processing_prep.reporting import AnalysisReport
from signal_processing_prep.synthetic import add_signals, sine_wave


def test_synthetic_analysis_produces_typed_artifacts_and_views() -> None:
    result = run_synthetic_analysis(
        AnalysisConfig(
            frequency_bands_hz={"low": (0.0, 100.0)},
            modeling=ModelingConfig(random_state=1),
        )
    )

    assert len(result.records) == 6
    assert result.quality_frame.shape[0] == 6
    assert result.feature_frame.shape[0] == 6
    assert "band_energy_low" in result.feature_frame.columns
    assert result.anomaly_model is not None
    assert "# Signal Analysis Summary" in result.markdown_summary


def test_pipeline_uses_supervised_suite_when_labels_are_sufficient() -> None:
    records = [
        sine_wave(frequency_hz=10.0 + index, label="normal", name=f"n{index}")
        for index in range(4)
    ] + [
        sine_wave(frequency_hz=80.0 + index, label="fault", name=f"f{index}")
        for index in range(4)
    ]

    result = SignalAnalysisPipeline(AnalysisConfig(modeling=ModelingConfig(random_state=2))).run_records(records)

    assert set(result.supervised_models) == {"logistic_regression", "random_forest"}
    assert result.anomaly_model is None
    assert "accuracy" in result.markdown_summary


def test_pipeline_filtering_is_recorded_in_processing_history() -> None:
    mixed = add_signals(
        [
            sine_wave(frequency_hz=20.0, duration_seconds=2.0, sampling_rate_hz=1000.0),
            sine_wave(frequency_hz=200.0, duration_seconds=2.0, sampling_rate_hz=1000.0),
        ],
        name="mixed",
    )
    result = SignalAnalysisPipeline(
        AnalysisConfig(
            filtering=FilteringConfig(enabled=True, kind="lowpass", high_cut_hz=50.0),
            modeling=ModelingConfig(enabled=False),
        )
    ).run_records([mixed])

    assert result.records[0].processing_history[-1].operation == "filter"
    assert result.records[0].processing_history[-1].parameters["filter_kind"] == "lowpass"
    assert result.feature_frame.loc[0, "dominant_frequency_hz"] == pytest.approx(20.0)
    assert band_energy(result.records[0], low_hz=190.0, high_hz=210.0) < 0.05


def test_pipeline_skips_only_record_data_failures() -> None:
    good = sine_wave(name="good")
    bad = sine_wave(name="bad")
    invalid_values = bad.values.copy()
    invalid_values.setflags(write=True)
    invalid_values[3] = np.nan
    bad = type(bad)(invalid_values, bad.sampling_rate_hz, name="bad")

    result = SignalAnalysisPipeline(
        AnalysisConfig(modeling=ModelingConfig(enabled=False), invalid_record_policy="skip")
    ).run_records([good, bad])

    assert [record.name for record in result.records] == ["good"]
    assert result.quality_frame["record_name"].tolist() == ["good", "bad"]
    assert result.feature_frame["record_name"].tolist() == ["good"]
    assert any("Feature extraction skipped bad" in note for note in result.processing_notes)

    with pytest.raises(ConfigurationError, match="spectrogram_window_seconds"):
        SignalAnalysisPipeline(AnalysisConfig(spectrogram_window_seconds=-1.0))


def test_pipeline_can_be_built_from_yaml_and_load_dataset(tmp_path: Path) -> None:
    time = np.arange(2000, dtype=float) / 1000.0
    values = np.sin(2.0 * np.pi * 20.0 * time) + np.sin(2.0 * np.pi * 200.0 * time)
    pd.DataFrame({"time": time, "signal": values}).to_csv(tmp_path / "mixed.csv", index=False)
    config_path = tmp_path / "configured.yaml"
    config_path.write_text(
        f"""
paths:
  data_dir: "{tmp_path.as_posix()}"
loading:
  file_patterns: ["*.csv"]
  signal_column: signal
analysis:
  frequency_bands_hz:
    target: [0, 50]
  filtering:
    enabled: true
    kind: lowpass
    high_cut_hz: 50
  modeling:
    enabled: false
""",
        encoding="utf-8",
    )

    result = SignalAnalysisPipeline.from_config(config_path).run_dataset()

    assert "band_energy_target" in result.feature_frame.columns
    assert result.records[0].processing_history[-1].operation == "filter"


def test_configured_pipeline_skips_malformed_files_under_record_policy(tmp_path: Path) -> None:
    pd.DataFrame({"time": [0.0, 0.1], "signal": ["bad", "values"]}).to_csv(
        tmp_path / "bad.csv", index=False
    )
    pd.DataFrame({"time": [0.0, 0.1], "signal": [0.0, 1.0]}).to_csv(
        tmp_path / "good.csv", index=False
    )
    config_path = tmp_path / "configured.yaml"
    config_path.write_text(
        f"""
paths:
  data_dir: "{tmp_path.as_posix()}"
loading:
  file_patterns: ["*.csv"]
  signal_column: signal
analysis:
  invalid_record_policy: skip
  modeling:
    enabled: false
""",
        encoding="utf-8",
    )

    result = SignalAnalysisPipeline.from_config(config_path).run_dataset()

    assert [record.provenance.source_name for record in result.records] == ["good"]
    assert any("Loading skipped bad.csv" in note for note in result.processing_notes)


def test_configured_pipeline_raises_when_all_loaded_files_are_invalid(tmp_path: Path) -> None:
    pd.DataFrame({"time": [0.0, 0.1], "signal": ["bad", "values"]}).to_csv(
        tmp_path / "bad.csv", index=False
    )
    config_path = tmp_path / "configured.yaml"
    config_path.write_text(
        f"""
paths:
  data_dir: "{tmp_path.as_posix()}"
loading:
  file_patterns: ["*.csv"]
  signal_column: signal
analysis:
  invalid_record_policy: skip
  modeling:
    enabled: false
""",
        encoding="utf-8",
    )

    with pytest.raises(RecordDataError, match="No valid records remained"):
        SignalAnalysisPipeline.from_config(config_path).run_dataset()


def test_pipeline_accepts_replaceable_typed_collaborators() -> None:
    calls: list[str] = []

    class FakeExtractor:
        def extract_records(self, records):
            calls.append(f"extract:{records[0].name}")
            return FeatureTable.from_dataframe(
                pd.DataFrame({"record_name": [records[0].name], "label": [None], "rms": [1.0]}),
                feature_columns=("rms",),
            )

    class FakeQuality:
        def assess(self, records):
            calls.append(f"quality:{records[0].name}")
            return QualityTable(pd.DataFrame({"record_name": [records[0].name], "issues": [""]}))

    class FakePreprocessor:
        def process(self, record):
            calls.append(f"preprocess:{record.name}")
            return record

    class FakeScorer:
        def score(self, features):
            calls.append(f"score:{len(features)}")
            return ModelEvaluation(
                "fake",
                "unsupervised_anomaly_score",
                None,
                ("rms",),
                {"n_samples": 1.0},
                PredictionTable.anomaly_scores(
                    pd.DataFrame({"row_index": [0], "anomaly_score": [1.0], "anomaly_rank": [1]})
                ),
                "fake",
            )

    class FakeReports:
        def build(self, **artifacts):
            calls.append(f"report:{artifacts['record_count']}")
            return AnalysisReport("Fake dataset")

    result = SignalAnalysisPipeline(
        AnalysisConfig(),
        feature_extractor=FakeExtractor(),
        anomaly_scorer=FakeScorer(),
        report_builder=FakeReports(),
        quality_assessor=FakeQuality(),
        preprocessor=FakePreprocessor(),
    ).run_records([sine_wave(name="demo")])

    assert calls == ["quality:demo", "preprocess:demo", "extract:demo", "score:1", "report:1"]
    assert "Fake dataset" in result.markdown_summary


def test_pipeline_rejects_unaligned_scorer_predictions() -> None:
    class WrongScorer:
        def score(self, features):
            return ModelEvaluation(
                "wrong",
                "unsupervised_anomaly_score",
                None,
                ("rms",),
                {"n_samples": 1.0},
                PredictionTable.anomaly_scores(
                    pd.DataFrame(
                        {
                            "row_index": [0],
                            "record_name": ["unrelated"],
                            "anomaly_score": [1.0],
                            "anomaly_rank": [1],
                        }
                    )
                ),
                "wrong",
            )

    with pytest.raises(ValueError, match="not aligned"):
        SignalAnalysisPipeline(anomaly_scorer=WrongScorer()).run_records(
            [sine_wave(label=None, name="source")]
        )


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="At least one SignalRecord"):
        SignalAnalysisPipeline().run_records([])
