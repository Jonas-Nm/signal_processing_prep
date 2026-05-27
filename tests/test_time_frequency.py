"""Tests for time-frequency analysis helpers."""

import numpy as np
import pytest

from signal_processing_prep.records import SignalRecord
from signal_processing_prep.synthetic import chirp_signal, sine_wave, transient_burst
from signal_processing_prep.time_frequency import (
    hilbert_analysis,
    hilbert_huang_transform,
    morlet_wavelet_scalogram,
    spectrogram_analysis,
    stft_analysis,
    teager_kaiser_demodulation,
    teager_kaiser_energy,
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


def test_hilbert_huang_emd_reconstructs_tone_and_identifies_dominant_frequency() -> None:
    """EMD HHT reconstructs a tone and exposes its dominant Hilbert frequency."""
    record = sine_wave(
        frequency_hz=40.0,
        duration_seconds=1.0,
        sampling_rate_hz=500.0,
        amplitude=2.0,
    )

    result = hilbert_huang_transform(record, max_imfs=4, n_frequency_bins=64)
    reconstructed = np.sum(result.intrinsic_mode_functions, axis=0) + result.residual
    dominant_index = int(np.argmax(np.mean(np.square(result.intrinsic_mode_functions), axis=1)))
    valid = result.valid_frequency_mask[dominant_index]

    assert result.method == "emd"
    assert result.intrinsic_mode_functions.shape[1] == record.n_samples
    assert result.spectrum_power.shape == (64, record.n_samples)
    assert np.allclose(reconstructed, record.values, atol=1e-10)
    assert np.median(result.instantaneous_frequency_hz[dominant_index, valid]) == pytest.approx(
        40.0,
        abs=0.5,
    )


def test_hilbert_huang_spectrum_tracks_controlled_chirp() -> None:
    """The binned Hilbert energy ridge follows a controlled chirp."""
    record = chirp_signal(
        start_frequency_hz=20.0,
        end_frequency_hz=100.0,
        duration_seconds=2.0,
        sampling_rate_hz=500.0,
    )

    result = hilbert_huang_transform(
        record,
        max_imfs=4,
        n_frequency_bins=128,
        max_frequency_hz=150.0,
    )
    peak_frequency = result.spectrum_frequencies_hz[np.argmax(result.spectrum_power, axis=0)]
    expected_frequency = 20.0 + 80.0 * result.time_seconds / 2.0
    interior = (result.time_seconds > 0.1) & (result.time_seconds < 1.9)

    assert np.isfinite(result.spectrum_power).all()
    assert np.median(np.abs(peak_frequency[interior] - expected_frequency[interior])) < 1.0


def test_hilbert_huang_ceemdan_is_reproducible_with_fixed_seed() -> None:
    """Configured serial CEEMDAN produces stable exploratory results."""
    record = sine_wave(
        frequency_hz=40.0,
        duration_seconds=0.4,
        sampling_rate_hz=500.0,
    )
    kwargs = {
        "method": "ceemdan",
        "max_imfs": 3,
        "ceemdan_trials": 3,
        "random_state": 4,
        "n_frequency_bins": 32,
    }

    first = hilbert_huang_transform(record, **kwargs)
    second = hilbert_huang_transform(record, **kwargs)

    assert first.method == "ceemdan"
    assert first.intrinsic_mode_functions.shape[0] <= kwargs["max_imfs"]
    assert first.residual.shape == record.values.shape
    assert first.valid_frequency_mask.shape == first.instantaneous_frequency_hz.shape
    assert np.array_equal(first.intrinsic_mode_functions, second.intrinsic_mode_functions)
    assert np.array_equal(first.spectrum_power, second.spectrum_power)
    assert np.allclose(
        np.sum(first.intrinsic_mode_functions, axis=0) + first.residual,
        record.values,
        atol=1e-10,
    )


def test_hilbert_huang_returns_residual_only_for_nonoscillatory_signals() -> None:
    """Constant or monotonic inputs do not produce interpretable adaptive modes."""
    for values, method in [(np.ones(20), "ceemdan"), (np.linspace(0.0, 1.0, 20), "emd")]:
        result = hilbert_huang_transform(values, sampling_rate_hz=20.0, method=method)

        assert result.intrinsic_mode_functions.shape == (0, values.size)
        assert result.instantaneous_frequency_hz.shape == (0, values.size)
        assert np.array_equal(result.residual, values)
        assert np.all(result.spectrum_power == 0.0)


def test_teager_kaiser_energy_matches_known_tone_value() -> None:
    """TKEO returns the analytical constant energy of a sampled sinusoid."""
    record = sine_wave(
        frequency_hz=25.0,
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
        amplitude=2.0,
    )

    result = teager_kaiser_energy(record)
    expected_energy = 4.0 * np.sin(2.0 * np.pi * 25.0 / 1000.0) ** 2

    assert result.energy.shape == (record.n_samples - 2,)
    assert result.time_seconds[0] == pytest.approx(1.0 / record.sampling_rate_hz)
    assert np.allclose(result.energy, expected_energy, rtol=1e-10, atol=1e-12)


def test_teager_kaiser_energy_localizes_transient_burst() -> None:
    """TKEO energy is elevated in a known active burst interval."""
    record = transient_burst(
        duration_seconds=2.0,
        sampling_rate_hz=1000.0,
        burst_frequency_hz=100.0,
        burst_start_seconds=1.0,
        burst_duration_seconds=0.2,
        amplitude=2.0,
    )

    result = teager_kaiser_energy(record)
    active = (result.time_seconds >= 1.0) & (result.time_seconds < 1.2)
    inactive = result.time_seconds < 0.8

    assert np.max(result.energy[active]) > 1.0
    assert np.max(np.abs(result.energy[inactive])) == pytest.approx(0.0)


def test_teager_kaiser_demodulation_tracks_narrowband_chirp_frequency() -> None:
    """DESA-2 follows a controlled chirp below the quarter-rate ambiguity limit."""
    record = chirp_signal(
        start_frequency_hz=20.0,
        end_frequency_hz=150.0,
        duration_seconds=2.0,
        sampling_rate_hz=2000.0,
    )

    result = teager_kaiser_demodulation(record)
    expected_frequency = 20.0 + (150.0 - 20.0) * result.time_seconds / 2.0
    interior = (result.time_seconds > 0.01) & (result.time_seconds < 1.99)
    valid = result.valid_mask & interior
    absolute_error = np.abs(result.instantaneous_frequency_hz[valid] - expected_frequency[valid])

    assert result.time_seconds.shape == (record.n_samples - 4,)
    assert np.count_nonzero(valid) > 0.98 * np.count_nonzero(interior)
    assert np.median(absolute_error) < 0.1
    assert np.max(absolute_error) < 0.5


def test_teager_kaiser_demodulation_recovers_known_tone_amplitude_and_frequency() -> None:
    """DESA-2 exactly separates amplitude and frequency for an ideal tone."""
    record = sine_wave(
        frequency_hz=80.0,
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
        amplitude=2.5,
    )

    result = teager_kaiser_demodulation(record)

    assert np.all(result.valid_mask)
    assert np.allclose(result.instantaneous_amplitude, 2.5, rtol=1e-10, atol=1e-12)
    assert np.allclose(result.instantaneous_frequency_hz, 80.0, rtol=1e-10, atol=1e-10)


def test_teager_kaiser_demodulation_marks_zero_energy_estimates_invalid() -> None:
    """Flat signals retain energy output but do not produce demodulated values."""
    result = teager_kaiser_demodulation(np.ones(8), sampling_rate_hz=100.0)

    assert not np.any(result.valid_mask)
    assert np.all(np.isnan(result.instantaneous_amplitude))
    assert np.all(np.isnan(result.instantaneous_frequency_hz))
    assert np.all(result.energy == 0.0)


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

    low_rate_record = sine_wave(duration_seconds=1.0, sampling_rate_hz=10.0)
    with pytest.raises(ValueError, match="step_seconds must not exceed"):
        spectrogram_analysis(low_rate_record, window_seconds=0.04, step_seconds=0.05)

    with pytest.raises(ValueError, match="Nyquist"):
        morlet_wavelet_scalogram(record, frequencies_hz=[60.0])

    with pytest.raises(ValueError, match="method"):
        hilbert_huang_transform(record, method="unknown")

    with pytest.raises(ValueError, match="max_imfs"):
        hilbert_huang_transform(record, max_imfs=0)

    with pytest.raises(ValueError, match="n_frequency_bins"):
        hilbert_huang_transform(record, n_frequency_bins=0)

    with pytest.raises(ValueError, match="max_frequency_hz"):
        hilbert_huang_transform(record, max_frequency_hz=60.0)

    with pytest.raises(ValueError, match="ceemdan_trials"):
        hilbert_huang_transform(record, method="ceemdan", ceemdan_trials=0)

    with pytest.raises(ValueError, match="ceemdan_epsilon"):
        hilbert_huang_transform(record, method="ceemdan", ceemdan_epsilon=0.0)

    with pytest.raises(ValueError, match="random_state"):
        hilbert_huang_transform(record, method="ceemdan", random_state=-1)

    with pytest.raises(ValueError, match="random_state"):
        hilbert_huang_transform(record, method="ceemdan", random_state=2**32)

    with pytest.raises(ValueError, match="at least 3 samples"):
        teager_kaiser_energy(np.ones(2), sampling_rate_hz=10.0)

    with pytest.raises(ValueError, match="at least 5 samples"):
        teager_kaiser_demodulation(np.ones(4), sampling_rate_hz=10.0)

    with pytest.raises(ValueError, match="validity_tolerance"):
        teager_kaiser_demodulation(np.ones(5), sampling_rate_hz=10.0, validity_tolerance=-1.0)

    with pytest.raises(ValueError, match="validity_tolerance"):
        teager_kaiser_demodulation(np.ones(5), sampling_rate_hz=10.0, validity_tolerance=np.nan)


def test_raw_teager_kaiser_helpers_require_sampling_rate() -> None:
    """Raw arrays need explicit timing information for TKEO result axes."""
    with pytest.raises(ValueError, match="sampling_rate_hz is required"):
        teager_kaiser_energy(np.ones(3))

    with pytest.raises(ValueError, match="sampling_rate_hz is required"):
        teager_kaiser_demodulation(np.ones(5))

    with pytest.raises(ValueError, match="sampling_rate_hz is required"):
        hilbert_huang_transform(np.ones(5))


def test_time_frequency_helpers_reject_nonfinite_samples_before_dsp() -> None:
    """Time-frequency helpers fail clearly for NaN or Inf samples."""
    record = SignalRecord(values=np.array([0.0, np.nan, 1.0, 0.0]), sampling_rate_hz=10.0)

    with pytest.raises(ValueError, match="non-finite samples"):
        stft_analysis(record, window_seconds=0.2)

    with pytest.raises(ValueError, match="non-finite samples"):
        spectrogram_analysis(record, window_seconds=0.2)

    with pytest.raises(ValueError, match="non-finite samples"):
        hilbert_analysis(record)

    with pytest.raises(ValueError, match="non-finite samples"):
        morlet_wavelet_scalogram(record, frequencies_hz=[1.0])

    with pytest.raises(ValueError, match="non-finite samples"):
        teager_kaiser_energy(record)

    with pytest.raises(ValueError, match="non-finite samples"):
        teager_kaiser_demodulation(record)

    with pytest.raises(ValueError, match="non-finite samples"):
        hilbert_huang_transform(record)


def test_energy_based_helpers_reject_nonfinite_computed_outputs() -> None:
    """Overflow in analytical energy values fails rather than escaping in results."""
    sampling_rate_hz = 1000.0
    times = np.arange(1000, dtype=np.float64) / sampling_rate_hz
    values = 1e155 * np.sin(2.0 * np.pi * 80.0 * times)

    with pytest.raises(ValueError, match="Teager-Kaiser energy calculation"):
        teager_kaiser_energy(values, sampling_rate_hz=sampling_rate_hz)

    with pytest.raises(ValueError, match="Teager-Kaiser demodulation"):
        teager_kaiser_demodulation(values, sampling_rate_hz=sampling_rate_hz)

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="Hilbert-Huang spectrum calculation"):
            hilbert_huang_transform(values, sampling_rate_hz=sampling_rate_hz)
