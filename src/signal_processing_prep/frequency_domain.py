"""Frequency-domain helpers for sampled time-series signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal as scipy_signal

from signal_processing_prep.records import SignalRecord


@dataclass(frozen=True)
class Spectrum:
    """Single-sided FFT magnitude spectrum."""

    frequencies_hz: NDArray[np.float64]
    magnitudes: NDArray[np.float64]


@dataclass(frozen=True)
class PowerSpectrum:
    """Single-sided power spectral density estimate."""

    frequencies_hz: NDArray[np.float64]
    power: NDArray[np.float64]


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

    resolved_sampling_rate = (
        signal.sampling_rate_hz if isinstance(signal, SignalRecord) else sampling_rate_hz
    )
    if resolved_sampling_rate is None:
        raise ValueError("sampling_rate_hz is required for raw signal arrays.")
    if resolved_sampling_rate <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    return values, float(resolved_sampling_rate)


def fft_magnitude(
    signal: SignalRecord | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
    remove_mean: bool = True,
) -> Spectrum:
    """Return the single-sided FFT amplitude spectrum for a signal."""
    values, sampling_rate = _values_and_sampling_rate(signal, sampling_rate_hz)
    if remove_mean:
        values = values - np.mean(values)

    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sampling_rate)
    magnitudes = np.abs(np.fft.rfft(values)) / values.size
    if values.size > 1:
        magnitudes[1:-1] *= 2.0
        if values.size % 2 == 1:
            magnitudes[-1] *= 2.0
    return Spectrum(frequencies_hz=frequencies, magnitudes=magnitudes)


def psd(
    signal: SignalRecord | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
    nperseg: int | None = None,
) -> PowerSpectrum:
    """Estimate power spectral density using Welch's method."""
    values, sampling_rate = _values_and_sampling_rate(signal, sampling_rate_hz)
    if nperseg is None:
        nperseg = min(1024, values.size)
    if nperseg <= 0:
        raise ValueError("nperseg must be positive.")

    frequencies, power = scipy_signal.welch(
        values,
        fs=sampling_rate,
        nperseg=min(nperseg, values.size),
        detrend="constant",
        scaling="density",
    )
    return PowerSpectrum(
        frequencies_hz=np.asarray(frequencies, dtype=np.float64),
        power=np.asarray(power, dtype=np.float64),
    )


def dominant_frequency(
    signal: SignalRecord | Spectrum | PowerSpectrum | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
    ignore_dc: bool = True,
) -> float:
    """Return the frequency with the largest spectral magnitude or power."""
    frequencies, weights = _spectral_weights(signal, sampling_rate_hz=sampling_rate_hz)
    if ignore_dc and frequencies.size > 1:
        frequencies = frequencies[1:]
        weights = weights[1:]
    if weights.size == 0 or np.max(weights) == 0.0:
        return 0.0
    return float(frequencies[int(np.argmax(weights))])


def spectral_centroid(
    signal: SignalRecord | Spectrum | PowerSpectrum | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
) -> float:
    """Return the weighted mean frequency of a spectrum."""
    frequencies, weights = _spectral_weights(signal, sampling_rate_hz=sampling_rate_hz)
    total = float(np.sum(weights))
    if total == 0.0:
        return 0.0
    return float(np.sum(frequencies * weights) / total)


def spectral_bandwidth(
    signal: SignalRecord | Spectrum | PowerSpectrum | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
) -> float:
    """Return the weighted standard deviation around the spectral centroid."""
    frequencies, weights = _spectral_weights(signal, sampling_rate_hz=sampling_rate_hz)
    total = float(np.sum(weights))
    if total == 0.0:
        return 0.0
    centroid = np.sum(frequencies * weights) / total
    variance = np.sum(weights * np.square(frequencies - centroid)) / total
    return float(np.sqrt(variance))


def spectral_rolloff(
    signal: SignalRecord | Spectrum | PowerSpectrum | ArrayLike,
    *,
    rolloff_fraction: float = 0.85,
    sampling_rate_hz: float | None = None,
) -> float:
    """Return the frequency below which a fraction of spectral weight lies.

    For ``Spectrum`` inputs and raw signals this uses FFT magnitudes. For
    ``PowerSpectrum`` inputs this uses PSD power values.
    """
    if not 0.0 < rolloff_fraction <= 1.0:
        raise ValueError("rolloff_fraction must be in the interval (0, 1].")
    frequencies, weights = _spectral_weights(signal, sampling_rate_hz=sampling_rate_hz)
    total = float(np.sum(weights))
    if total == 0.0:
        return 0.0
    threshold = rolloff_fraction * total
    index = int(np.searchsorted(np.cumsum(weights), threshold, side="left"))
    return float(frequencies[min(index, frequencies.size - 1)])


def spectral_flatness(
    signal: SignalRecord | Spectrum | PowerSpectrum | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
    epsilon: float = 1e-12,
) -> float:
    """Return geometric mean divided by arithmetic mean of spectral weights."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    _, weights = _spectral_weights(signal, sampling_rate_hz=sampling_rate_hz)
    positive_weights = np.maximum(weights, epsilon)
    arithmetic_mean = float(np.mean(positive_weights))
    if arithmetic_mean == 0.0:
        return 0.0
    geometric_mean = float(np.exp(np.mean(np.log(positive_weights))))
    return geometric_mean / arithmetic_mean


def band_energy(
    signal: SignalRecord | ArrayLike,
    *,
    low_hz: float,
    high_hz: float,
    sampling_rate_hz: float | None = None,
    remove_mean: bool = True,
) -> float:
    """Return mean-square signal energy in a frequency band.

    The single-sided FFT bin energies sum to total signal mean-square amplitude,
    so this is directly comparable across bands for fixed preprocessing choices.
    """
    if low_hz < 0:
        raise ValueError("low_hz must be non-negative.")
    if high_hz <= low_hz:
        raise ValueError("high_hz must be greater than low_hz.")
    values, sampling_rate = _values_and_sampling_rate(signal, sampling_rate_hz)
    if high_hz > sampling_rate / 2.0:
        raise ValueError("high_hz must not exceed the Nyquist frequency.")
    if remove_mean:
        values = values - np.mean(values)

    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sampling_rate)
    coefficients = np.fft.rfft(values)
    energies = np.square(np.abs(coefficients)) / np.square(values.size)
    if values.size > 1:
        energies[1:-1] *= 2.0
        if values.size % 2 == 1:
            energies[-1] *= 2.0

    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    return float(np.sum(energies[band_mask]))


def _spectral_weights(
    signal: SignalRecord | Spectrum | PowerSpectrum | ArrayLike,
    *,
    sampling_rate_hz: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(signal, Spectrum):
        return signal.frequencies_hz, signal.magnitudes
    if isinstance(signal, PowerSpectrum):
        return signal.frequencies_hz, signal.power
    spectrum = fft_magnitude(signal, sampling_rate_hz=sampling_rate_hz)
    return spectrum.frequencies_hz, spectrum.magnitudes
