"""Time-frequency analysis helpers for sampled signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pywt
from PyEMD import CEEMDAN, EMD
from scipy import signal as scipy_signal

from signal_processing_prep._validation import validate_finite_signal_values
from signal_processing_prep.records import SignalRecord


@dataclass(frozen=True)
class StftResult:
    """Short-time Fourier transform result."""

    frequencies_hz: NDArray[np.float64]
    times_seconds: NDArray[np.float64]
    coefficients: NDArray[np.complex128]
    magnitude: NDArray[np.float64]


@dataclass(frozen=True)
class SpectrogramResult:
    """Power spectrogram result."""

    frequencies_hz: NDArray[np.float64]
    times_seconds: NDArray[np.float64]
    power: NDArray[np.float64]


@dataclass(frozen=True)
class HilbertResult:
    """Analytic-signal features from the Hilbert transform."""

    time_seconds: NDArray[np.float64]
    analytic_signal: NDArray[np.complex128]
    amplitude_envelope: NDArray[np.float64]
    instantaneous_phase_radians: NDArray[np.float64]
    instantaneous_frequency_hz: NDArray[np.float64]


@dataclass(frozen=True)
class HilbertHuangResult:
    """Empirical modes and Hilbert spectral analysis for one signal."""

    method: str
    time_seconds: NDArray[np.float64]
    intrinsic_mode_functions: NDArray[np.float64]
    residual: NDArray[np.float64]
    instantaneous_amplitude: NDArray[np.float64]
    instantaneous_frequency_hz: NDArray[np.float64]
    valid_frequency_mask: NDArray[np.bool_]
    spectrum_frequencies_hz: NDArray[np.float64]
    spectrum_power: NDArray[np.float64]


@dataclass(frozen=True)
class TeagerKaiserEnergyResult:
    """Sample-aligned output of the discrete Teager-Kaiser energy operator."""

    time_seconds: NDArray[np.float64]
    energy: NDArray[np.float64]


@dataclass(frozen=True)
class TeagerKaiserDemodulationResult:
    """DESA-2 amplitude and instantaneous-frequency estimates."""

    time_seconds: NDArray[np.float64]
    energy: NDArray[np.float64]
    instantaneous_amplitude: NDArray[np.float64]
    instantaneous_frequency_hz: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class WaveletScalogram:
    """Continuous wavelet transform scalogram."""

    frequencies_hz: NDArray[np.float64]
    times_seconds: NDArray[np.float64]
    coefficients: NDArray[np.complex128]
    power: NDArray[np.float64]
    widths: NDArray[np.float64]
    wavelet: str


def stft_analysis(
    signal: SignalRecord | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
    window_seconds: float = 0.1,
    step_seconds: float | None = None,
    window: str = "hann",
) -> StftResult:
    """Compute a short-time Fourier transform with explicit window controls."""
    values, sampling_rate = _values_and_sampling_rate(signal, sampling_rate_hz)
    nperseg, noverlap = _window_and_overlap_samples(
        window_seconds,
        step_seconds,
        sampling_rate,
        values.size,
    )
    frequencies, times, coefficients = scipy_signal.stft(
        values,
        fs=sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    return StftResult(
        frequencies_hz=np.asarray(frequencies, dtype=np.float64),
        times_seconds=np.asarray(times, dtype=np.float64),
        coefficients=coefficients,
        magnitude=np.abs(coefficients),
    )


def spectrogram_analysis(
    signal: SignalRecord | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
    window_seconds: float = 0.1,
    step_seconds: float | None = None,
    window: str = "hann",
) -> SpectrogramResult:
    """Compute a power spectrogram with explicit window controls."""
    values, sampling_rate = _values_and_sampling_rate(signal, sampling_rate_hz)
    nperseg, noverlap = _window_and_overlap_samples(
        window_seconds,
        step_seconds,
        sampling_rate,
        values.size,
    )
    frequencies, times, power = scipy_signal.spectrogram(
        values,
        fs=sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )
    return SpectrogramResult(
        frequencies_hz=np.asarray(frequencies, dtype=np.float64),
        times_seconds=np.asarray(times, dtype=np.float64),
        power=np.asarray(power, dtype=np.float64),
    )


def hilbert_analysis(
    signal: SignalRecord | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
) -> HilbertResult:
    """Compute analytic signal, envelope, phase, and instantaneous frequency.

    Hilbert envelope and instantaneous frequency are most interpretable for
    narrowband or monocomponent signals; broadband mixtures can produce
    misleading instantaneous-frequency trajectories.
    """
    values, sampling_rate = _values_and_sampling_rate(signal, sampling_rate_hz)
    analytic_signal = np.asarray(scipy_signal.hilbert(values), dtype=np.complex128)
    envelope = np.abs(analytic_signal)
    phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(phase) * sampling_rate / (2.0 * np.pi)
    if instantaneous_frequency.size:
        instantaneous_frequency = np.concatenate(
            [instantaneous_frequency, instantaneous_frequency[-1:]]
        )
    else:
        instantaneous_frequency = np.zeros_like(values)
    times = np.arange(values.size, dtype=np.float64) / sampling_rate
    return HilbertResult(
        time_seconds=times,
        analytic_signal=analytic_signal,
        amplitude_envelope=envelope,
        instantaneous_phase_radians=phase,
        instantaneous_frequency_hz=instantaneous_frequency.astype(np.float64),
    )


def hilbert_huang_transform(
    signal: SignalRecord | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
    method: str = "emd",
    max_imfs: int | None = None,
    n_frequency_bins: int = 128,
    max_frequency_hz: float | None = None,
    ceemdan_trials: int = 20,
    ceemdan_epsilon: float = 0.005,
    random_state: int = 0,
) -> HilbertHuangResult:
    """Compute EMD/CEEMDAN modes followed by Hilbert spectral analysis.

    HHT is data-adaptive and useful for exploratory nonstationary analysis,
    but IMF identities are not guaranteed physical components. Results can
    exhibit mode mixing, edge effects, and noise sensitivity.
    """
    values, sampling_rate = _values_and_sampling_rate(signal, sampling_rate_hz)
    method = method.lower()
    if method not in {"emd", "ceemdan"}:
        raise ValueError("method must be 'emd' or 'ceemdan'.")
    if max_imfs is not None and (not isinstance(max_imfs, int) or max_imfs <= 0):
        raise ValueError("max_imfs must be a positive integer or None.")
    if not isinstance(n_frequency_bins, int) or n_frequency_bins <= 0:
        raise ValueError("n_frequency_bins must be a positive integer.")
    nyquist_hz = sampling_rate / 2.0
    if max_frequency_hz is None:
        max_frequency_hz = nyquist_hz
    if not np.isfinite(max_frequency_hz) or not 0.0 < max_frequency_hz <= nyquist_hz:
        raise ValueError("max_frequency_hz must be positive and not exceed Nyquist.")
    if method == "ceemdan":
        if not isinstance(ceemdan_trials, int) or ceemdan_trials <= 0:
            raise ValueError("ceemdan_trials must be a positive integer.")
        if not np.isfinite(ceemdan_epsilon) or ceemdan_epsilon <= 0:
            raise ValueError("ceemdan_epsilon must be positive and finite.")
        if not isinstance(random_state, int) or not 0 <= random_state <= 2**32 - 1:
            raise ValueError("random_state must be an integer between 0 and 2**32 - 1.")

    times = np.arange(values.size, dtype=np.float64) / sampling_rate
    if _has_no_oscillatory_extrema(values):
        imfs = np.empty((0, values.size), dtype=np.float64)
        residual = values.copy()
    else:
        imfs, residual = _empirical_mode_decomposition(
            values,
            times,
            method=method,
            max_imfs=max_imfs,
            ceemdan_trials=ceemdan_trials,
            ceemdan_epsilon=ceemdan_epsilon,
            random_state=random_state,
        )
    amplitudes, frequencies = _hilbert_modes(imfs, sampling_rate)
    valid_frequency_mask = (
        np.isfinite(frequencies)
        & (frequencies >= 0.0)
        & (frequencies <= nyquist_hz)
    )
    spectrum_frequencies, spectrum_power = _hilbert_energy_spectrum(
        amplitudes,
        frequencies,
        valid_frequency_mask,
        max_frequency_hz=float(max_frequency_hz),
        n_frequency_bins=n_frequency_bins,
    )
    return HilbertHuangResult(
        method=method,
        time_seconds=times,
        intrinsic_mode_functions=imfs,
        residual=residual,
        instantaneous_amplitude=amplitudes,
        instantaneous_frequency_hz=frequencies,
        valid_frequency_mask=valid_frequency_mask,
        spectrum_frequencies_hz=spectrum_frequencies,
        spectrum_power=spectrum_power,
    )


def teager_kaiser_energy(
    signal: SignalRecord | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
) -> TeagerKaiserEnergyResult:
    """Compute the discrete Teager-Kaiser energy operator for a signal.

    The returned energy is aligned to interior signal samples because
    ``x[n]**2 - x[n - 1] * x[n + 1]`` requires one neighbor on each side.
    """
    values, sampling_rate = _values_and_sampling_rate(signal, sampling_rate_hz)
    if values.size < 3:
        raise ValueError("Teager-Kaiser energy requires at least 3 samples.")
    times = np.arange(1, values.size - 1, dtype=np.float64) / sampling_rate
    energy = _discrete_teager_kaiser_energy(values)
    if not np.isfinite(energy).all():
        raise ValueError("Teager-Kaiser energy calculation produced non-finite values.")
    return TeagerKaiserEnergyResult(
        time_seconds=times,
        energy=energy,
    )


def teager_kaiser_demodulation(
    signal: SignalRecord | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
    validity_tolerance: float = 1e-12,
) -> TeagerKaiserDemodulationResult:
    """Estimate instantaneous amplitude and frequency with DESA-2.

    This estimator is intended for monocomponent or explicitly narrowband
    AM-FM signals. Its instantaneous-frequency result is unambiguous only
    below one quarter of the sampling rate. ``energy`` retains the aligned
    TKEO values; invalid demodulated estimates are returned as ``NaN`` and
    identified through ``valid_mask``. That mask records numerical formula
    validity; it cannot detect violations of the narrowband or frequency-range
    assumptions.
    """
    if not np.isfinite(validity_tolerance) or validity_tolerance < 0:
        raise ValueError("validity_tolerance must be finite and non-negative.")
    values, sampling_rate = _values_and_sampling_rate(signal, sampling_rate_hz)
    if values.size < 5:
        raise ValueError("Teager-Kaiser demodulation requires at least 5 samples.")

    energy = _discrete_teager_kaiser_energy(values)[1:-1]
    with np.errstate(over="ignore", invalid="ignore"):
        centered_difference = values[2:] - values[:-2]
    difference_energy = _discrete_teager_kaiser_energy(centered_difference)
    if not np.isfinite(energy).all() or not np.isfinite(difference_energy).all():
        raise ValueError("Teager-Kaiser demodulation produced non-finite energy values.")
    arccos_argument = np.full(energy.shape, np.nan, dtype=np.float64)
    positive_energy = (energy > 0.0) & (difference_energy > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        arccos_argument[positive_energy] = (
            1.0 - difference_energy[positive_energy] / (2.0 * energy[positive_energy])
        )
    valid_mask = (
        positive_energy
        & np.isfinite(arccos_argument)
        & (arccos_argument >= -1.0 - validity_tolerance)
        & (arccos_argument <= 1.0 + validity_tolerance)
    )

    instantaneous_amplitude = np.full(energy.shape, np.nan, dtype=np.float64)
    instantaneous_frequency = np.full(energy.shape, np.nan, dtype=np.float64)
    clipped_argument = np.clip(arccos_argument[valid_mask], -1.0, 1.0)
    angular_frequency = 0.5 * np.arccos(clipped_argument)
    instantaneous_frequency[valid_mask] = angular_frequency * sampling_rate / (2.0 * np.pi)
    with np.errstate(over="ignore", invalid="ignore"):
        instantaneous_amplitude[valid_mask] = (
            2.0 * energy[valid_mask] / np.sqrt(difference_energy[valid_mask])
        )
    if not np.isfinite(instantaneous_amplitude[valid_mask]).all():
        raise ValueError("Teager-Kaiser demodulation produced non-finite amplitude values.")
    times = np.arange(2, values.size - 2, dtype=np.float64) / sampling_rate
    return TeagerKaiserDemodulationResult(
        time_seconds=times,
        energy=energy,
        instantaneous_amplitude=instantaneous_amplitude,
        instantaneous_frequency_hz=instantaneous_frequency,
        valid_mask=valid_mask,
    )


def morlet_wavelet_scalogram(
    signal: SignalRecord | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
    frequencies_hz: ArrayLike | None = None,
    min_frequency_hz: float = 10.0,
    max_frequency_hz: float | None = None,
    n_frequencies: int = 32,
    wavelet: str = "cmor1.5-1.0",
) -> WaveletScalogram:
    """Compute a complex Morlet continuous wavelet scalogram."""
    values, sampling_rate = _values_and_sampling_rate(signal, sampling_rate_hz)
    if frequencies_hz is None:
        if max_frequency_hz is None:
            max_frequency_hz = sampling_rate / 2.0
        if min_frequency_hz <= 0:
            raise ValueError("min_frequency_hz must be positive.")
        if max_frequency_hz <= min_frequency_hz:
            raise ValueError("max_frequency_hz must be greater than min_frequency_hz.")
        if n_frequencies <= 0:
            raise ValueError("n_frequencies must be positive.")
        frequencies = np.geomspace(min_frequency_hz, max_frequency_hz, n_frequencies)
    else:
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequencies_hz must be a non-empty one-dimensional array.")
        if np.any(frequencies <= 0):
            raise ValueError("frequencies_hz values must be positive.")
    if np.any(frequencies > sampling_rate / 2.0):
        raise ValueError("Wavelet frequencies must not exceed the Nyquist frequency.")

    continuous_wavelet = pywt.ContinuousWavelet(wavelet)
    central_frequency = pywt.central_frequency(continuous_wavelet)
    scales = central_frequency * sampling_rate / frequencies
    coefficients, resolved_frequencies = pywt.cwt(
        values,
        scales,
        continuous_wavelet,
        sampling_period=1.0 / sampling_rate,
    )
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    times = np.arange(values.size, dtype=np.float64) / sampling_rate
    return WaveletScalogram(
        frequencies_hz=np.asarray(resolved_frequencies, dtype=np.float64),
        times_seconds=times,
        coefficients=coefficients,
        power=np.square(np.abs(coefficients)),
        widths=scales.astype(np.float64),
        wavelet=wavelet,
    )


def _discrete_teager_kaiser_energy(values: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.square(values[1:-1]) - values[:-2] * values[2:]


def _has_no_oscillatory_extrema(values: NDArray[np.float64]) -> bool:
    maxima = scipy_signal.find_peaks(values)[0]
    minima = scipy_signal.find_peaks(-values)[0]
    return maxima.size == 0 or minima.size == 0


def _empirical_mode_decomposition(
    values: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    method: str,
    max_imfs: int | None,
    ceemdan_trials: int,
    ceemdan_epsilon: float,
    random_state: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    limit = -1 if max_imfs is None else max_imfs
    if method == "emd":
        decomposition = EMD()
        decomposition.emd(values, times, max_imf=limit)
        imfs, residual = decomposition.get_imfs_and_residue()
    else:
        decomposition = CEEMDAN(
            trials=ceemdan_trials,
            epsilon=ceemdan_epsilon,
            parallel=False,
        )
        decomposition.noise_seed(random_state)
        components = decomposition.ceemdan(values, times, max_imf=limit)
        # PyEMD CEEMDAN returns the final residue as its last component.
        imfs, residual = components[:-1], components[-1]
    imfs = np.asarray(imfs, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if not np.isfinite(imfs).all() or not np.isfinite(residual).all():
        raise ValueError("Empirical mode decomposition produced non-finite values.")
    return imfs, residual


def _hilbert_modes(
    imfs: NDArray[np.float64],
    sampling_rate_hz: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    amplitudes = np.empty_like(imfs)
    frequencies = np.empty_like(imfs)
    for index, imf in enumerate(imfs):
        result = hilbert_analysis(imf, sampling_rate_hz=sampling_rate_hz)
        amplitudes[index] = result.amplitude_envelope
        frequencies[index] = result.instantaneous_frequency_hz
    return amplitudes, frequencies


def _hilbert_energy_spectrum(
    amplitudes: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    valid_frequency_mask: NDArray[np.bool_],
    *,
    max_frequency_hz: float,
    n_frequency_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    edges = np.linspace(0.0, max_frequency_hz, n_frequency_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    spectrum_power = np.zeros((n_frequency_bins, amplitudes.shape[1]), dtype=np.float64)
    in_range = valid_frequency_mask & (frequencies <= max_frequency_hz)
    imf_indices, time_indices = np.nonzero(in_range)
    if imf_indices.size:
        bin_indices = np.searchsorted(
            edges,
            frequencies[imf_indices, time_indices],
            side="right",
        ) - 1
        bin_indices = np.clip(bin_indices, 0, n_frequency_bins - 1)
        with np.errstate(over="ignore", invalid="ignore"):
            np.add.at(
                spectrum_power,
                (bin_indices, time_indices),
                np.square(amplitudes[imf_indices, time_indices]),
            )
    if not np.isfinite(spectrum_power).all():
        raise ValueError("Hilbert-Huang spectrum calculation produced non-finite power values.")
    return centers, spectrum_power


def _values_and_sampling_rate(
    signal: SignalRecord | ArrayLike,
    sampling_rate_hz: float | None,
) -> tuple[NDArray[np.float64], float]:
    values = signal.values if isinstance(signal, SignalRecord) else signal
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Signal values must be one-dimensional.")
    if values.size == 0:
        raise ValueError("Signal values must not be empty.")
    validate_finite_signal_values(values)
    resolved_sampling_rate = (
        signal.sampling_rate_hz if isinstance(signal, SignalRecord) else sampling_rate_hz
    )
    if resolved_sampling_rate is None:
        raise ValueError("sampling_rate_hz is required for raw signal arrays.")
    if resolved_sampling_rate <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    return values, float(resolved_sampling_rate)


def _window_and_overlap_samples(
    window_seconds: float,
    step_seconds: float | None,
    sampling_rate_hz: float,
    n_samples: int,
) -> tuple[int, int]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")
    if step_seconds is None:
        step_seconds = window_seconds / 2.0
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive.")
    if step_seconds > window_seconds:
        raise ValueError("step_seconds must not exceed window_seconds.")
    nperseg = max(1, int(round(window_seconds * sampling_rate_hz)))
    step_samples = max(1, int(round(step_seconds * sampling_rate_hz)))
    if nperseg > n_samples:
        raise ValueError("window_seconds must not exceed the signal duration.")
    return nperseg, nperseg - step_samples
