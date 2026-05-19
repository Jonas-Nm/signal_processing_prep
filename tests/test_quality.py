"""Tests for signal quality checks."""

import numpy as np
import pytest

from signal_processing_prep.quality import (
    QualityCheckConfig,
    assess_dataset_quality,
    assess_signal_quality,
)
from signal_processing_prep.records import SignalRecord
from signal_processing_prep.synthetic import clipped_signal, sine_wave


def test_assess_signal_quality_reports_clean_tone_without_issues() -> None:
    """A normal sine wave has finite metrics and no quality flags."""
    record = sine_wave(duration_seconds=1.0, sampling_rate_hz=1000.0, name="tone")

    report = assess_signal_quality(record, QualityCheckConfig(min_duration_seconds=0.5))

    assert report.record_name == "tone"
    assert report.duration_seconds == 1.0
    assert report.has_missing_values is False
    assert report.is_clipped is False
    assert report.is_near_constant is False
    assert report.issues == ()


def test_assess_signal_quality_flags_missing_values_and_duration() -> None:
    """Missing samples and short records are reported explicitly."""
    record = SignalRecord(
        values=np.array([0.0, np.nan, 1.0, np.inf]),
        sampling_rate_hz=10.0,
        name="bad",
    )

    report = assess_signal_quality(record, QualityCheckConfig(min_duration_seconds=1.0))

    assert report.has_missing_values is True
    assert report.missing_fraction == pytest.approx(0.5)
    assert "missing_values" in report.issues
    assert "duration_below_minimum" in report.issues


def test_assess_signal_quality_flags_clipping_and_scaling() -> None:
    """Clipping, near-constant signals, and large amplitudes are separate flags."""
    clipped = clipped_signal(
        frequency_hz=20.0,
        amplitude=2.0,
        clip_limit=0.5,
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
    )
    constant = SignalRecord(values=np.ones(100), sampling_rate_hz=100.0)
    large = sine_wave(amplitude=10.0, duration_seconds=1.0, sampling_rate_hz=1000.0)

    clipped_report = assess_signal_quality(clipped)
    constant_report = assess_signal_quality(constant)
    large_report = assess_signal_quality(
        large,
        QualityCheckConfig(large_amplitude_threshold=5.0),
    )

    assert clipped_report.is_clipped is True
    assert "possible_clipping" in clipped_report.issues
    assert constant_report.is_near_constant is True
    assert "near_constant" in constant_report.issues
    assert large_report.has_large_amplitude is True
    assert "large_amplitude" in large_report.issues


def test_assess_signal_quality_flags_simple_nonstationarity() -> None:
    """Segment-level mean drift highlights obvious nonstationary behavior."""
    values = np.concatenate([np.zeros(100), np.ones(100)])
    record = SignalRecord(values=values, sampling_rate_hz=100.0)

    report = assess_signal_quality(
        record,
        QualityCheckConfig(
            stationarity_window_seconds=0.5,
            stationarity_mean_drift_threshold=0.5,
        ),
    )

    assert report.is_likely_nonstationary is True
    assert "nonstationarity_indicator" in report.issues


def test_assess_dataset_quality_returns_one_row_per_record() -> None:
    """Dataset quality checks are tabular and preserve metadata."""
    records = [
        sine_wave(name="first", label="a"),
        SignalRecord(values=np.ones(10), sampling_rate_hz=10.0, name="second"),
    ]

    quality = assess_dataset_quality(records)

    assert quality.shape[0] == 2
    assert quality["record_name"].tolist() == ["first", "second"]
    assert "issues" in quality.columns


def test_assess_dataset_quality_empty_input_has_stable_columns() -> None:
    """Empty quality output keeps the schema expected by reports."""
    quality = assess_dataset_quality([])

    assert quality.empty
    assert list(quality.columns) == [
        "record_name",
        "label",
        "n_samples",
        "sampling_rate_hz",
        "duration_seconds",
        "missing_fraction",
        "has_missing_values",
        "max_abs_amplitude",
        "mean",
        "std",
        "rms",
        "clipping_fraction",
        "is_clipped",
        "is_near_constant",
        "has_large_amplitude",
        "stationarity_mean_drift",
        "stationarity_std_cv",
        "is_likely_nonstationary",
        "issues",
    ]


def test_assess_signal_quality_rejects_invalid_config() -> None:
    """Invalid thresholds fail before producing misleading reports."""
    record = sine_wave()

    with pytest.raises(ValueError, match="min_duration_seconds must be positive"):
        assess_signal_quality(record, QualityCheckConfig(min_duration_seconds=0.0))

    with pytest.raises(ValueError, match="clipping_fraction_threshold"):
        assess_signal_quality(record, QualityCheckConfig(clipping_fraction_threshold=2.0))

    with pytest.raises(ValueError, match="large_amplitude_threshold"):
        assess_signal_quality(record, QualityCheckConfig(large_amplitude_threshold=0.0))
