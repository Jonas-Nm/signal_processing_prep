"""Tests for synthetic signal generators."""

import numpy as np
import pytest

from signal_processing_prep.synthetic import (
    add_signals,
    chirp_signal,
    clipped_signal,
    convolve_signals,
    impulse_train,
    make_synthetic_dataset,
    multiply_signals,
    noisy_sine_wave,
    sinc_signal,
    sine_wave,
    transient_burst,
    window_signal,
)


def test_sine_wave_shape_sampling_rate_and_label() -> None:
    """A sine wave has the expected sample count, sampling rate, and label."""
    record = sine_wave(
        frequency_hz=10.0,
        duration_seconds=2.0,
        sampling_rate_hz=100.0,
        label="healthy",
    )

    assert record.n_samples == 200
    assert record.sampling_rate_hz == 100.0
    assert record.label == "healthy"
    assert record.attributes["frequency_hz"] == 10.0
    assert np.max(np.abs(record.values)) <= 1.0


def test_noisy_sine_wave_is_reproducible_with_seed() -> None:
    """Noisy synthetic signals are reproducible when a seed is provided."""
    first = noisy_sine_wave(seed=42)
    second = noisy_sine_wave(seed=42)

    np.testing.assert_allclose(first.values, second.values)
    assert first.label == "noisy_sine"


def test_impulse_train_contains_sparse_impulses() -> None:
    """An impulse train contains nonzero samples at regular intervals."""
    record = impulse_train(
        duration_seconds=1.0,
        sampling_rate_hz=100.0,
        impulse_rate_hz=10.0,
        amplitude=2.0,
    )

    nonzero_indices = np.flatnonzero(record.values)
    assert len(nonzero_indices) == 10
    assert np.all(record.values[nonzero_indices] == 2.0)


def test_chirp_signal_records_frequency_range() -> None:
    """A chirp signal records its configured frequency sweep."""
    record = chirp_signal(
        start_frequency_hz=5.0,
        end_frequency_hz=50.0,
        duration_seconds=1.0,
        sampling_rate_hz=200.0,
    )

    assert record.n_samples == 200
    assert record.attributes["start_frequency_hz"] == 5.0
    assert record.attributes["end_frequency_hz"] == 50.0


def test_clipped_signal_respects_clip_limit() -> None:
    """A clipped signal stays inside the configured absolute clip limit."""
    record = clipped_signal(amplitude=2.0, clip_limit=0.5)

    assert np.max(record.values) <= 0.5
    assert np.min(record.values) >= -0.5
    assert record.label == "clipped"


def test_transient_burst_is_active_only_in_window() -> None:
    """A transient burst has energy inside the burst window and silence outside it."""
    record = transient_burst(
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
        burst_frequency_hz=100.0,
        burst_start_seconds=0.2,
        burst_duration_seconds=0.1,
    )
    time = record.time_seconds
    inside = (time >= 0.2) & (time < 0.3)
    outside = ~inside

    assert np.any(np.abs(record.values[inside]) > 0.0)
    assert np.all(record.values[outside] == 0.0)


def test_window_signal_generates_rectangular_and_smooth_windows() -> None:
    """Window signals expose rectangular and smoother taper shapes."""
    rectangular = window_signal(
        duration_seconds=1.0,
        sampling_rate_hz=100.0,
        window_type="rectangular",
        amplitude=2.0,
    )
    hann = window_signal(
        duration_seconds=1.0,
        sampling_rate_hz=100.0,
        window_type="hann",
    )

    assert rectangular.n_samples == 100
    assert rectangular.attributes["window_type"] == "rectangular"
    assert rectangular.attributes["active_window_start_seconds"] == 0.0
    assert rectangular.attributes["active_window_duration_seconds"] == 1.0
    assert np.all(rectangular.values == 2.0)
    assert hann.n_samples == 100
    assert hann.attributes["window_type"] == "hann"
    assert hann.values[0] == pytest.approx(0.0)
    assert hann.values[-1] == pytest.approx(0.0)
    assert np.max(hann.values) == pytest.approx(1.0, rel=1e-3)


def test_window_signal_can_be_embedded_in_longer_record() -> None:
    """Window start and duration define where the active window sits in the record."""
    record = window_signal(
        duration_seconds=2.0,
        sampling_rate_hz=100.0,
        window_start_seconds=0.5,
        window_duration_seconds=0.25,
        window_type="rectangular",
    )
    active = (record.time_seconds >= 0.5) & (record.time_seconds < 0.75)

    assert record.n_samples == 200
    assert np.all(record.values[~active] == 0.0)
    assert np.all(record.values[active] == 1.0)
    assert record.attributes["active_window_start_seconds"] == 0.5
    assert record.attributes["active_window_duration_seconds"] == 0.25


def test_window_signal_rejects_invalid_arguments() -> None:
    """Invalid window configuration fails clearly."""
    with pytest.raises(ValueError, match="window_type"):
        window_signal(window_type="unknown")

    with pytest.raises(ValueError, match="amplitude"):
        window_signal(amplitude=-1.0)

    with pytest.raises(ValueError, match="window_start_seconds"):
        window_signal(window_start_seconds=-0.1)

    with pytest.raises(ValueError, match="window_duration_seconds"):
        window_signal(window_duration_seconds=0.0)

    with pytest.raises(ValueError, match="fit within"):
        window_signal(duration_seconds=1.0, window_start_seconds=0.75, window_duration_seconds=0.5)


def test_sinc_signal_is_centered_and_records_bandwidth() -> None:
    """A sinc signal peaks near its configured center time."""
    record = sinc_signal(
        duration_seconds=1.0,
        sampling_rate_hz=100.0,
        bandwidth_hz=8.0,
        center_seconds=0.5,
        amplitude=3.0,
    )
    peak_index = int(np.argmax(record.values))

    assert record.n_samples == 100
    assert record.time_seconds[peak_index] == pytest.approx(0.5)
    assert record.values[peak_index] == pytest.approx(3.0, rel=1e-3)
    assert record.attributes["bandwidth_hz"] == 8.0
    assert record.attributes["center_seconds"] == 0.5


def test_sinc_signal_rejects_invalid_arguments() -> None:
    """Invalid sinc configuration fails clearly."""
    with pytest.raises(ValueError, match="bandwidth_hz"):
        sinc_signal(bandwidth_hz=0.0)

    with pytest.raises(ValueError, match="center_seconds"):
        sinc_signal(center_seconds=2.0, duration_seconds=1.0)


def test_make_synthetic_dataset_returns_representative_records() -> None:
    """The default synthetic dataset contains one record for each planned example."""
    records = make_synthetic_dataset()

    assert len(records) == 6
    assert {record.label for record in records} == {
        "sine",
        "noisy_sine",
        "impulse_train",
        "chirp",
        "clipped",
        "transient_burst",
    }


def test_add_signals_combines_same_length_records() -> None:
    """Synthetic records can be added sample by sample."""
    first = sine_wave(
        frequency_hz=5.0,
        duration_seconds=1.0,
        sampling_rate_hz=100.0,
        amplitude=1.0,
        name="low",
    )
    second = sine_wave(
        frequency_hz=20.0,
        duration_seconds=1.0,
        sampling_rate_hz=100.0,
        amplitude=0.5,
        name="high",
    )

    combined = add_signals([first, second], label="mixed", name="two_tone")

    np.testing.assert_allclose(combined.values, first.values + second.values)
    assert combined.sampling_rate_hz == 100.0
    assert combined.label == "mixed"
    assert combined.name == "two_tone"
    assert combined.attributes["operation"] == "add"
    assert combined.attributes["source_names"] == ("low", "high")


def test_multiply_signals_combines_same_length_records() -> None:
    """Synthetic records can be multiplied sample by sample."""
    carrier = sine_wave(
        frequency_hz=20.0,
        duration_seconds=1.0,
        sampling_rate_hz=200.0,
    )
    envelope = sine_wave(
        frequency_hz=2.0,
        duration_seconds=1.0,
        sampling_rate_hz=200.0,
        amplitude=0.5,
    )

    product = multiply_signals([carrier, envelope])

    np.testing.assert_allclose(product.values, carrier.values * envelope.values)
    assert product.attributes["operation"] == "multiply"


def test_convolve_signals_uses_requested_mode() -> None:
    """Synthetic records can be convolved while preserving sampling rate metadata."""
    first = impulse_train(
        duration_seconds=0.05,
        sampling_rate_hz=100.0,
        impulse_rate_hz=20.0,
    )
    kernel = sine_wave(
        frequency_hz=10.0,
        duration_seconds=0.03,
        sampling_rate_hz=100.0,
    )

    convolved = convolve_signals(first, kernel, mode="full")

    expected = np.convolve(first.values, kernel.values, mode="full")
    np.testing.assert_allclose(convolved.values, expected)
    assert convolved.sampling_rate_hz == first.sampling_rate_hz
    assert convolved.n_samples == first.n_samples + kernel.n_samples - 1
    assert convolved.attributes["operation"] == "convolve"
    assert convolved.attributes["mode"] == "full"


def test_signal_composition_rejects_incompatible_records() -> None:
    """Composition fails clearly when sampling assumptions are incompatible."""
    first = sine_wave(duration_seconds=1.0, sampling_rate_hz=100.0)
    different_length = sine_wave(duration_seconds=2.0, sampling_rate_hz=100.0)
    different_rate = sine_wave(duration_seconds=1.0, sampling_rate_hz=200.0)

    with pytest.raises(ValueError, match="same number of samples"):
        add_signals([first, different_length])

    with pytest.raises(ValueError, match="same sampling_rate_hz"):
        multiply_signals([first, different_rate])

    with pytest.raises(ValueError, match="mode must be"):
        convolve_signals(first, first, mode="unsupported")

    with pytest.raises(ValueError, match="At least one"):
        add_signals([])
