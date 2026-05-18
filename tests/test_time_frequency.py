"""Tests for time-frequency analysis helpers."""

import numpy as np
import pytest

from signal_processing_prep.synthetic import sine_wave, transient_burst
from signal_processing_prep.time_frequency import (
    hilbert_analysis,
    morlet_wavelet_scalogram,
    spectrogram_analysis,
    stft_analysis,
)


def test_stft_analysis_detects_known_tone_frequency() -> None:
    """STFT magnitude peaks near the known sine frequency."""
    record = sine_wave(
        frequency_hz=50.0,
        duration_seconds=2.0,
        sampling_rate_hz=1000.0,
    )

    result = stft_analysis(record, window_seconds=0.5, step_seconds=0.25)
    mean_magnitude = np.mean(result.magnitude, axis=1)
    peak_frequency = result.frequencies_hz[int(np.argmax(mean_magnitude))]

    assert result.coefficients.shape == result.magnitude.shape
    assert result.magnitude.shape[0] == result.frequencies_hz.size
    assert result.magnitude.shape[1] == result.times_seconds.size
    assert peak_frequency == pytest.approx(50.0, abs=2.0)


def test_spectrogram_analysis_localizes_transient_energy() -> None:
    """Spectrogram energy increases around a transient burst."""
    record = transient_burst(
        duration_seconds=2.0,
        sampling_rate_hz=1000.0,
        burst_frequency_hz=200.0,
        burst_start_seconds=1.0,
        burst_duration_seconds=0.2,
    )

    result = spectrogram_analysis(record, window_seconds=0.2, step_seconds=0.1)
    energy_by_time = np.sum(result.power, axis=0)
    peak_time = result.times_seconds[int(np.argmax(energy_by_time))]

    assert 0.9 <= peak_time <= 1.2
    assert result.power.shape == (result.frequencies_hz.size, result.times_seconds.size)


def test_hilbert_analysis_returns_envelope_and_instantaneous_frequency() -> None:
    """Hilbert analysis returns expected envelope and frequency for a pure tone."""
    record = sine_wave(
        frequency_hz=25.0,
        duration_seconds=2.0,
        sampling_rate_hz=1000.0,
    )

    result = hilbert_analysis(record)

    assert result.amplitude_envelope.shape == record.values.shape
    assert result.instantaneous_frequency_hz.shape == record.values.shape
    assert np.median(result.amplitude_envelope) == pytest.approx(1.0, rel=1e-2)
    assert np.median(result.instantaneous_frequency_hz[10:-10]) == pytest.approx(
        25.0,
        rel=1e-2,
    )


def test_morlet_wavelet_scalogram_returns_frequency_time_power_grid() -> None:
    """Morlet scalogram returns one row per requested frequency."""
    record = sine_wave(
        frequency_hz=40.0,
        duration_seconds=1.0,
        sampling_rate_hz=500.0,
    )
    frequencies = np.array([20.0, 40.0, 80.0])

    result = morlet_wavelet_scalogram(record, frequencies_hz=frequencies)
    mean_power = np.mean(result.power, axis=1)

    assert result.coefficients.shape == (3, record.n_samples)
    assert result.power.shape == result.coefficients.shape
    assert result.times_seconds.shape == record.values.shape
    assert result.frequencies_hz.tolist() == pytest.approx(frequencies.tolist())
    assert result.frequencies_hz[int(np.argmax(mean_power))] == pytest.approx(40.0)


def test_time_frequency_helpers_reject_invalid_window_arguments() -> None:
    """Invalid time-frequency configuration fails clearly."""
    record = sine_wave(duration_seconds=1.0, sampling_rate_hz=100.0)

    with pytest.raises(ValueError, match="window_seconds must be positive"):
        stft_analysis(record, window_seconds=0.0)

    with pytest.raises(ValueError, match="step_seconds must not exceed"):
        spectrogram_analysis(record, window_seconds=0.1, step_seconds=0.2)

    with pytest.raises(ValueError, match="Nyquist"):
        morlet_wavelet_scalogram(record, frequencies_hz=[60.0])
