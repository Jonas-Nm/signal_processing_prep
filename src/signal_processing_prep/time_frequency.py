"""Time-frequency analysis helpers for sampled signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pywt
from scipy import signal as scipy_signal

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
    nperseg = int(round(window_seconds * sampling_rate_hz))
    step_samples = int(round(step_seconds * sampling_rate_hz))
    if nperseg <= 0:
        raise ValueError("window_seconds produces no samples.")
    if step_samples <= 0:
        raise ValueError("step_seconds produces no samples.")
    if nperseg > n_samples:
        raise ValueError("window_seconds must not exceed the signal duration.")
    if step_samples > nperseg:
        raise ValueError("step_seconds must not exceed window_seconds.")
    return nperseg, nperseg - step_samples
