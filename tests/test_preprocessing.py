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
    segment_dataset,
    segment_signal,
)
from signal_processing_prep.records import SignalRecord
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


def test_segment_signal_returns_explicit_windows() -> None:
    """Windowing utilities preserve timing metadata for each segment."""
    record = SignalRecord(values=np.arange(10, dtype=float), sampling_rate_hz=10.0, name="ramp")

    windows = segment_signal(record, window_seconds=0.4, step_seconds=0.2)

    assert len(windows) == 4
    assert windows[0].values.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert windows[1].metadata["window_start_seconds"] == 0.2
    assert windows[-1].metadata["window_end_seconds"] == 1.0


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
