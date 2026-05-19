"""Tests for high-level analysis pipeline orchestration."""

import json
from pathlib import Path

import pandas as pd
import pytest

from signal_processing_prep.config import FilteringConfig
from signal_processing_prep.features import FrequencyBand
from signal_processing_prep.frequency_domain import band_energy
from signal_processing_prep.pipeline import (
    AnalysisPipelineConfig,
    analyze_records,
    run_synthetic_analysis,
)
from signal_processing_prep.synthetic import add_signals, sine_wave


def test_run_synthetic_analysis_produces_core_artifacts() -> None:
    """The synthetic pipeline produces quality, features, modeling, and summary outputs."""
    result = run_synthetic_analysis(
        AnalysisPipelineConfig(
            frequency_bands=(FrequencyBand("low", 0.0, 100.0),),
            random_state=1,
        )
    )

    assert len(result.records) == 6
    assert result.quality.shape[0] == 6
    assert result.features.shape[0] == 6
    assert "band_energy_low" in result.features.columns
    assert result.anomaly_model is not None
    assert result.supervised_models == {}
    assert "Supervised baselines skipped" in " ".join(result.modeling_notes)
    assert "# Signal Analysis Summary" in result.markdown_summary


def test_analyze_records_uses_supervised_models_when_labels_are_sufficient() -> None:
    """Supervised baselines are selected when labels support a train/test split."""
    records = [
        sine_wave(frequency_hz=10.0, label="normal", name="n1"),
        sine_wave(frequency_hz=11.0, label="normal", name="n2"),
        sine_wave(frequency_hz=12.0, label="normal", name="n3"),
        sine_wave(frequency_hz=13.0, label="normal", name="n4"),
        sine_wave(frequency_hz=80.0, label="fault", name="f1"),
        sine_wave(frequency_hz=81.0, label="fault", name="f2"),
        sine_wave(frequency_hz=82.0, label="fault", name="f3"),
        sine_wave(frequency_hz=83.0, label="fault", name="f4"),
    ]

    result = analyze_records(records, AnalysisPipelineConfig(random_state=2))

    assert set(result.supervised_models) == {"logistic_regression", "random_forest"}
    assert result.anomaly_model is None
    assert isinstance(result.features, pd.DataFrame)
    assert "accuracy" in result.markdown_summary


def test_analyze_records_also_scores_anomalies_when_some_rows_are_unlabeled() -> None:
    """Partially labeled datasets can train supervised baselines and still score anomalies."""
    records = [
        sine_wave(frequency_hz=10.0, label="normal", name="n1"),
        sine_wave(frequency_hz=11.0, label="normal", name="n2"),
        sine_wave(frequency_hz=12.0, label="normal", name="n3"),
        sine_wave(frequency_hz=13.0, label="normal", name="n4"),
        sine_wave(frequency_hz=80.0, label="fault", name="f1"),
        sine_wave(frequency_hz=81.0, label="fault", name="f2"),
        sine_wave(frequency_hz=82.0, label="fault", name="f3"),
        sine_wave(frequency_hz=83.0, label="fault", name="f4"),
        sine_wave(frequency_hz=200.0, label=None, name="unknown"),
    ]

    result = analyze_records(records, AnalysisPipelineConfig(random_state=2))

    assert set(result.supervised_models) == {"logistic_regression", "random_forest"}
    assert result.anomaly_model is not None
    assert result.anomaly_model.predictions.shape[0] == 9
    assert "some rows were unlabeled" in " ".join(result.modeling_notes)


def test_analyze_records_applies_optional_filter_before_features() -> None:
    """Configured filtering is part of pipeline feature extraction."""
    low = sine_wave(frequency_hz=20.0, duration_seconds=2.0, sampling_rate_hz=1000.0)
    high = sine_wave(frequency_hz=200.0, duration_seconds=2.0, sampling_rate_hz=1000.0)
    mixed = add_signals([low, high], label=None, name="mixed")

    result = analyze_records(
        [mixed],
        AnalysisPipelineConfig(
            run_modeling=False,
            filtering=FilteringConfig(enabled=True, kind="lowpass", high_cut_hz=50.0),
        ),
    )

    assert result.records[0].metadata["preprocessing"]["filter_kind"] == "lowpass"
    assert result.features.loc[0, "dominant_frequency_hz"] == pytest.approx(20.0)
    assert band_energy(result.records[0], low_hz=190.0, high_hz=210.0) < 0.05


def test_analyze_records_can_disable_modeling() -> None:
    """Modeling remains optional in the pipeline."""
    result = analyze_records(
        [sine_wave(name="demo")],
        AnalysisPipelineConfig(run_modeling=False),
    )

    assert result.supervised_models == {}
    assert result.anomaly_model is None
    assert "Modeling was disabled or skipped" in result.markdown_summary


def test_analyze_records_rejects_empty_input() -> None:
    """Empty datasets fail clearly."""
    with pytest.raises(ValueError, match="At least one SignalRecord"):
        analyze_records([])


def test_phase_nine_notebook_exists_and_uses_package_apis() -> None:
    """The main walkthrough notebook is present and calls package functions."""
    notebook_path = Path("notebooks/01_signal_analysis_walkthrough.ipynb")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert notebook["nbformat"] == 4
    assert 'config_path = Path("configs/synthetic.yaml")' in source
    assert "from signal_processing_prep.pipeline import" in source
    assert "make_synthetic_dataset()" in source
    assert "analyze_records(" in source
    assert "save_markdown_summary" in source
