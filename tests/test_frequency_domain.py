"""Tests for frequency-domain signal helpers."""

import numpy as np
import pytest

from signal_processing_prep.frequency_domain import (
    band_energy,
    dominant_frequency,
    fft_magnitude,
    psd,
    spectral_bandwidth,
    spectral_centroid,
    spectral_flatness,
    spectral_rolloff,
)
from signal_processing_prep.records import SignalRecord
from signal_processing_prep.synthetic import noisy_sine_wave, sine_wave


def test_fft_magnitude_detects_known_sine_peak() -> None:
    """FFT magnitude has its dominant non-DC bin at the sine frequency."""
    record = sine_wave(
        frequency_hz=50.0,
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
        amplitude=2.0,
    )
    spectrum = fft_magnitude(record)

    assert dominant_frequency(spectrum) == pytest.approx(50.0)
    peak_index = int(np.argmax(spectrum.magnitudes[1:]) + 1)
    assert spectrum.magnitudes[peak_index] == pytest.approx(2.0, rel=1e-3)


def test_psd_detects_known_sine_peak() -> None:
    """Welch PSD peak is close to a known sine-wave frequency."""
    record = sine_wave(
        frequency_hz=80.0,
        duration_seconds=2.0,
        sampling_rate_hz=1000.0,
    )
    power_spectrum = psd(record, nperseg=1000)

    assert dominant_frequency(power_spectrum) == pytest.approx(80.0, abs=1.0)


def test_band_energy_captures_sine_mean_square_power() -> None:
    """Band energy around a sine peak equals its mean-square amplitude."""
    record = sine_wave(
        frequency_hz=50.0,
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
        amplitude=2.0,
    )

    assert band_energy(record, low_hz=45.0, high_hz=55.0) == pytest.approx(2.0, rel=1e-3)
    assert band_energy(record, low_hz=100.0, high_hz=150.0) == pytest.approx(0.0, abs=1e-12)


def test_band_energy_uses_half_open_boundaries_for_adjacent_bands() -> None:
    """Shared band boundaries are not double counted across adjacent bands."""
    record = sine_wave(
        frequency_hz=100.0,
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
        amplitude=2.0,
    )

    assert band_energy(record, low_hz=0.0, high_hz=100.0) == pytest.approx(0.0, abs=1e-12)
    assert band_energy(record, low_hz=100.0, high_hz=200.0) == pytest.approx(2.0, rel=1e-3)


def test_band_energy_rejects_invalid_frequency_bands() -> None:
    """Band energy validates band boundaries against the Nyquist frequency."""
    record = sine_wave(sampling_rate_hz=1000.0)

    with pytest.raises(ValueError, match="Band limits must be finite"):
        band_energy(record, low_hz=np.nan, high_hz=10.0)

    with pytest.raises(ValueError, match="Band limits must be finite"):
        band_energy(record, low_hz=10.0, high_hz=np.inf)

    with pytest.raises(ValueError, match="low_hz must be non-negative"):
        band_energy(record, low_hz=-1.0, high_hz=10.0)

    with pytest.raises(ValueError, match="high_hz must be greater"):
        band_energy(record, low_hz=10.0, high_hz=10.0)

    with pytest.raises(ValueError, match="Nyquist"):
        band_energy(record, low_hz=10.0, high_hz=600.0)


def test_raw_frequency_helpers_require_sampling_rate() -> None:
    """Raw arrays need an explicit sampling rate for frequency-domain helpers."""
    values = np.ones(16)

    with pytest.raises(ValueError, match="sampling_rate_hz is required"):
        fft_magnitude(values)

    with pytest.raises(ValueError, match="sampling_rate_hz is required"):
        psd(values)

    with pytest.raises(ValueError, match="sampling_rate_hz is required"):
        band_energy(values, low_hz=1.0, high_hz=2.0)


def test_frequency_helpers_reject_nonfinite_samples_before_dsp() -> None:
    """Frequency-domain helpers fail clearly for NaN or Inf samples."""
    record = SignalRecord(values=np.array([0.0, np.nan, 1.0]), sampling_rate_hz=10.0)

    with pytest.raises(ValueError, match="non-finite samples"):
        fft_magnitude(record)

    with pytest.raises(ValueError, match="non-finite samples"):
        psd(np.array([0.0, np.inf, 1.0]), sampling_rate_hz=10.0)

    with pytest.raises(ValueError, match="non-finite samples"):
        band_energy(record, low_hz=1.0, high_hz=2.0)


@pytest.mark.parametrize("sampling_rate_hz", [np.nan, np.inf])
def test_frequency_helpers_reject_nonfinite_sampling_rates(sampling_rate_hz: float) -> None:
    """Frequency axes and bands require finite timing."""
    values = np.ones(16)

    with pytest.raises(ValueError, match="sampling_rate_hz must be positive and finite"):
        fft_magnitude(values, sampling_rate_hz=sampling_rate_hz)

    with pytest.raises(ValueError, match="sampling_rate_hz must be positive and finite"):
        psd(values, sampling_rate_hz=sampling_rate_hz)

    with pytest.raises(ValueError, match="sampling_rate_hz must be positive and finite"):
        band_energy(values, low_hz=1.0, high_hz=2.0, sampling_rate_hz=sampling_rate_hz)


def test_frequency_helpers_handle_tiny_signals() -> None:
    """Tiny but valid records return finite, interpretable spectral outputs."""
    record = sine_wave(
        frequency_hz=1.0,
        duration_seconds=0.001,
        sampling_rate_hz=1000.0,
    )

    spectrum = fft_magnitude(record)

    assert spectrum.frequencies_hz.tolist() == [0.0]
    assert dominant_frequency(spectrum) == 0.0
    assert spectral_centroid(spectrum) == 0.0
    assert spectral_bandwidth(spectrum) == 0.0
    assert spectral_rolloff(spectrum) == 0.0
    assert np.isfinite(spectral_flatness(spectrum))


def test_spectral_flatness_is_zero_for_zero_energy_spectra() -> None:
    """Silent or mean-removed constant signals do not look spectrally flat."""
    assert spectral_flatness(np.zeros(16), sampling_rate_hz=1000.0) == 0.0
    assert spectral_flatness(np.ones(16), sampling_rate_hz=1000.0) == 0.0


def test_spectral_descriptors_are_interpretable_for_tone_and_noise() -> None:
    """Spectral descriptors remain bounded and separate tonal from noisy signals."""
    tone = sine_wave(frequency_hz=40.0, duration_seconds=1.0, sampling_rate_hz=1000.0)
    noisy = noisy_sine_wave(
        frequency_hz=40.0,
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
        noise_std=1.0,
        seed=1,
    )

    assert spectral_centroid(tone) == pytest.approx(40.0, abs=1.0)
    assert spectral_bandwidth(tone) < 1.0
    assert 40.0 <= spectral_rolloff(tone, rolloff_fraction=0.85) <= 41.0
    assert 0.0 < spectral_flatness(tone) < spectral_flatness(noisy) < 1.0
