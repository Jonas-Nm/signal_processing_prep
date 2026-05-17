"""Tests for synthetic signal generators."""

import numpy as np

from signal_processing_prep.synthetic import (
    chirp_signal,
    clipped_signal,
    impulse_train,
    make_synthetic_dataset,
    noisy_sine_wave,
    sine_wave,
    transient_burst,
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
    assert record.metadata["frequency_hz"] == 10.0
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
    assert record.metadata["start_frequency_hz"] == 5.0
    assert record.metadata["end_frequency_hz"] == 50.0


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
