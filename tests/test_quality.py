"""Tests for typed signal-quality assessment."""

import numpy as np
import pytest

from signal_processing_prep.config import QualityConfig
from signal_processing_prep.quality import QualityAssessor, assess_signal_quality
from signal_processing_prep.records import AcquisitionDiagnostics, SignalRecord
from signal_processing_prep.synthetic import clipped_signal, sine_wave


def test_assess_signal_quality_reports_clean_tone() -> None:
    report = assess_signal_quality(
        sine_wave(duration_seconds=1.0, sampling_rate_hz=1000.0, name="tone"),
        QualityConfig(min_duration_seconds=0.5),
    )
    assert report.record_name == "tone"
    assert report.issues == ()


def test_assessment_flags_record_quality_failures() -> None:
    missing = SignalRecord([0.0, np.nan, 1.0, np.inf], 10.0, name="bad")
    missing_report = assess_signal_quality(missing, QualityConfig(min_duration_seconds=1.0))
    clipped_report = assess_signal_quality(clipped_signal(amplitude=2.0, clip_limit=0.5))
    constant_report = assess_signal_quality(SignalRecord(np.ones(100), 100.0))

    assert missing_report.missing_fraction == pytest.approx(0.5)
    assert {"missing_values", "duration_below_minimum"}.issubset(missing_report.issues)
    assert "possible_clipping" in clipped_report.issues
    assert "near_constant" in constant_report.issues


def test_quality_reads_only_typed_acquisition_observations() -> None:
    record = SignalRecord(
        np.ones(10),
        10.0,
        attributes={"time_gap_count": 0, "sampling_rate_mismatch_fraction": 0.0},
        acquisition=AcquisitionDiagnostics(
            time_axis_valid=True,
            time_step_jitter_fraction=0.2,
            time_gap_count=2,
            sampling_rate_mismatch_fraction=0.3,
        ),
    )
    report = assess_signal_quality(record, QualityConfig(time_step_jitter_threshold=0.05))

    assert report.time_gap_count == 2
    assert report.has_time_axis_irregularity
    assert report.has_sampling_rate_mismatch


def test_quality_assessor_returns_typed_table() -> None:
    table = QualityAssessor().assess([sine_wave(name="first"), sine_wave(name="second")])
    frame = table.to_dataframe()

    assert frame["record_name"].tolist() == ["first", "second"]
    assert "issues" in frame.columns


def test_invalid_quality_policy_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="min_duration_seconds"):
        QualityConfig(min_duration_seconds=0.0)
