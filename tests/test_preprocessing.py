"""Tests for optional preprocessing utilities."""

import numpy as np
import pytest

from signal_processing_prep.config import FilteringConfig
from signal_processing_prep.frequency_domain import band_energy
from signal_processing_prep.preprocessing import (
    FilterSpec,
    apply_configured_filter,
    apply_filter,
    apply_window,
    interpolate_missing_values,
    segment_dataset,
    segment_signal,
)
from signal_processing_prep.records import AcquisitionDiagnostics, SignalProvenance, SignalRecord
from signal_processing_prep.synthetic import add_signals, sine_wave


def test_lowpass_filter_reduces_high_frequency_energy() -> None:
    """Optional filtering attenuates out-of-band content without changing metadata."""
    low = sine_wave(frequency_hz=20.0, duration_seconds=2.0, sampling_rate_hz=1000.0)
    high = sine_wave(frequency_hz=200.0, duration_seconds=2.0, sampling_rate_hz=1000.0)
    mixed = add_signals([low, high], name="mixed")

    filtered = apply_filter(mixed, FilterSpec(kind="lowpass", high_cut_hz=50.0, order=4))

    assert filtered.name == "mixed"
    assert filtered.sampling_rate_hz == mixed.sampling_rate_hz
    assert band_energy(filtered, low_hz=190.0, high_hz=210.0) < 0.05
    assert filtered.metadata["preprocessing"]["filter_kind"] == "lowpass"


def test_filter_preserves_existing_preprocessing_metadata() -> None:
    """Filtering appends to preprocessing history instead of replacing it."""
    record = SignalRecord(
        values=np.array([0.0, np.nan, 0.5, 0.0, -0.5, 0.0, 0.5, 0.0, -0.5, 0.0]),
        sampling_rate_hz=10.0,
        name="dirty",
    )
    cleaned = interpolate_missing_values(record, max_missing_fraction=0.2)

    filtered = apply_filter(
        cleaned,
        FilterSpec(
            kind="lowpass",
            high_cut_hz=2.0,
            order=1,
            zero_phase=False,
        ),
    )

    preprocessing = filtered.metadata["preprocessing"]
    assert preprocessing["missing_value_interpolation"]["missing_count"] == 1
    assert preprocessing["filter_kind"] == "lowpass"


def test_bandpass_filter_keeps_target_band() -> None:
    """Band-pass filtering keeps the requested band dominant."""
    low = sine_wave(frequency_hz=20.0, duration_seconds=2.0, sampling_rate_hz=1000.0)
    high = sine_wave(frequency_hz=200.0, duration_seconds=2.0, sampling_rate_hz=1000.0)
    mixed = add_signals([low, high])

    filtered = apply_filter(
        mixed,
        FilterSpec(kind="bandpass", low_cut_hz=150.0, high_cut_hz=250.0),
    )

    assert band_energy(filtered, low_hz=190.0, high_hz=210.0) > 0.3
    assert band_energy(filtered, low_hz=15.0, high_hz=25.0) < 0.05


def test_apply_configured_filter_returns_record_when_disabled() -> None:
    """Disabled filtering is explicit and leaves the original object untouched."""
    record = sine_wave()

    filtered = apply_configured_filter(record, FilteringConfig(enabled=False))

    assert filtered is record


def test_zero_phase_filter_rejects_short_records_without_fallback() -> None:
    """Short records do not silently switch to causal filtering."""
    record = sine_wave(duration_seconds=0.05, sampling_rate_hz=100.0)

    with pytest.raises(ValueError, match="too short for zero-phase filtering"):
        apply_filter(record, FilterSpec(kind="lowpass", high_cut_hz=20.0))


def test_zero_phase_filter_can_use_explicit_causal_fallback() -> None:
    """Causal fallback is available only when requested."""
    record = sine_wave(duration_seconds=0.05, sampling_rate_hz=100.0)

    filtered = apply_filter(
        record,
        FilterSpec(
            kind="lowpass",
            high_cut_hz=20.0,
            allow_causal_fallback=True,
        ),
    )

    assert filtered.n_samples == record.n_samples
    assert filtered.metadata["preprocessing"]["phase_mode"] == "causal_fallback"


def test_apply_window_multiplies_values_and_records_metadata() -> None:
    """Windowing is an explicit preprocessing step."""
    record = SignalRecord(values=np.ones(8), sampling_rate_hz=8.0, name="ones")

    windowed = apply_window(record, window="hann")

    assert windowed.values[0] == pytest.approx(0.0)
    assert windowed.values[4] == pytest.approx(1.0)
    assert windowed.metadata["window"]["name"] == "hann"
    assert windowed.metadata["preprocessing"]["window"]["name"] == "hann"


def test_apply_window_preserves_existing_preprocessing_metadata() -> None:
    """Windowing records consistent preprocessing metadata without losing history."""
    record = SignalRecord(
        values=np.array([1.0, np.nan, 3.0, 4.0]),
        sampling_rate_hz=4.0,
        metadata={"preprocessing": {"source_step": "manual"}},
    )
    cleaned = interpolate_missing_values(record, max_missing_fraction=0.25)

    windowed = apply_window(cleaned, window="hann")

    preprocessing = windowed.metadata["preprocessing"]
    assert preprocessing["source_step"] == "manual"
    assert preprocessing["missing_value_interpolation"]["missing_count"] == 1
    assert preprocessing["window"]["name"] == "hann"


def test_interpolate_missing_values_fills_small_gaps_and_records_metadata() -> None:
    """Small missing-value repairs are explicit and preserve record identity."""
    record = SignalRecord(
        values=np.array([0.0, np.nan, 2.0, np.inf, 4.0]),
        sampling_rate_hz=10.0,
        label="normal",
        name="dirty",
        metadata={"sensor": "accel"},
    )

    cleaned = interpolate_missing_values(record, max_missing_fraction=0.5)

    np.testing.assert_allclose(cleaned.values, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert cleaned.sampling_rate_hz == record.sampling_rate_hz
    assert cleaned.label == record.label
    assert cleaned.name == record.name
    assert cleaned.metadata["sensor"] == "accel"
    interpolation = cleaned.metadata["preprocessing"]["missing_value_interpolation"]
    assert interpolation["method"] == "linear"
    assert interpolation["missing_count"] == 2
    assert interpolation["missing_fraction"] == pytest.approx(0.4)


def test_interpolate_missing_values_refuses_large_missing_fraction() -> None:
    """Interpolation refuses records that exceed the configured repair budget."""
    record = SignalRecord(values=np.array([0.0, np.nan, np.nan, 3.0]), sampling_rate_hz=4.0)

    with pytest.raises(ValueError, match="Missing fraction exceeds"):
        interpolate_missing_values(record, max_missing_fraction=0.25)


def test_segment_signal_returns_explicit_windows() -> None:
    """Windowing utilities preserve timing metadata for each segment."""
    record = SignalRecord(
        values=np.arange(10, dtype=float),
        sampling_rate_hz=10.0,
        name="ramp_channel",
        provenance=SignalProvenance(source_name="ramp", source_path="capture.csv"),
        acquisition=AcquisitionDiagnostics(time_gap_count=0),
    )

    windows = segment_signal(record, window_seconds=0.4, step_seconds=0.2)

    assert len(windows) == 4
    assert windows[0].values.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert windows[1].metadata["window_start_seconds"] == 0.2
    assert windows[-1].metadata["window_end_seconds"] == 1.0
    assert {window.provenance.source_name for window in windows} == {"ramp"}
    assert windows[0].provenance.source_path == "capture.csv"
    assert windows[0].acquisition == record.acquisition


def test_segment_signal_can_include_partial_window() -> None:
    """Partial windows are opt-in rather than silently truncating or padding."""
    record = SignalRecord(values=np.arange(10, dtype=float), sampling_rate_hz=10.0)

    windows = segment_signal(
        record,
        window_seconds=0.4,
        step_seconds=0.4,
        include_partial=True,
    )

    assert [window.n_samples for window in windows] == [4, 4, 2]


def test_segment_dataset_returns_flat_window_list() -> None:
    """Multiple records can be segmented with one call."""
    records = [
        SignalRecord(values=np.arange(8, dtype=float), sampling_rate_hz=8.0, name="a"),
        SignalRecord(values=np.arange(8, dtype=float), sampling_rate_hz=8.0, name="b"),
    ]

    windows = segment_dataset(records, window_seconds=0.5)

    assert len(windows) == 4
    assert windows[0].name == "a_window_0"
    assert windows[2].name == "b_window_0"


def test_preprocessing_rejects_invalid_filter_and_window_settings() -> None:
    """Invalid preprocessing settings fail clearly."""
    record = sine_wave(sampling_rate_hz=100.0)

    with pytest.raises(ValueError, match="Filter kind"):
        apply_filter(record, FilterSpec(kind="not-a-filter"))

    with pytest.raises(ValueError, match="high_cut_hz"):
        apply_filter(record, FilterSpec(kind="lowpass"))

    with pytest.raises(ValueError, match="Nyquist"):
        apply_filter(record, FilterSpec(kind="highpass", low_cut_hz=60.0))

    with pytest.raises(ValueError, match="low_cut_hz must be less"):
        apply_filter(
            record,
            FilterSpec(kind="bandpass", low_cut_hz=20.0, high_cut_hz=10.0),
        )

    with pytest.raises(ValueError, match="window_seconds must be positive"):
        segment_signal(record, window_seconds=0.0)

    with pytest.raises(ValueError, match="step_seconds must not exceed"):
        segment_signal(record, window_seconds=0.5, step_seconds=1.0)

    low_rate_record = sine_wave(duration_seconds=1.0, sampling_rate_hz=10.0)
    with pytest.raises(ValueError, match="step_seconds must not exceed"):
        segment_signal(low_rate_record, window_seconds=0.04, step_seconds=0.05)
