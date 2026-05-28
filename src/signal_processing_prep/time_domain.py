"""Time-domain metrics for one-dimensional sampled signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from signal_processing_prep._validation import validate_finite_signal_values
from signal_processing_prep.records import SignalRecord


def _values_from_signal(signal: SignalRecord | ArrayLike) -> NDArray[np.float64]:
    values = signal.values if isinstance(signal, SignalRecord) else signal
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Signal values must be one-dimensional.")
    if values.size == 0:
        raise ValueError("Signal values must not be empty.")
    validate_finite_signal_values(values)
    return values


def rms(signal: SignalRecord | ArrayLike) -> float:
    """Return the root mean square amplitude of a signal."""
    values = _values_from_signal(signal)
    return float(np.sqrt(np.mean(np.square(values))))


def crest_factor(signal: SignalRecord | ArrayLike) -> float:
    """Return peak absolute amplitude divided by RMS amplitude."""
    values = _values_from_signal(signal)
    signal_rms = rms(values)
    if signal_rms == 0.0:
        return 0.0
    return float(np.max(np.abs(values)) / signal_rms)


def zero_crossing_rate(
    signal: SignalRecord | ArrayLike,
    *,
    sampling_rate_hz: float | None = None,
) -> float:
    """Return zero crossings per second, or per interval without a sampling rate.

    When a ``SignalRecord`` is provided, its sampling rate is used automatically.
    For raw arrays, pass ``sampling_rate_hz`` to obtain crossings per second.
    Without a sampling rate, the result is normalized by the number of sample
    intervals and therefore lies between 0 and 1 for ordinary finite signals.
    """
    values = _values_from_signal(signal)
    if values.size < 2:
        return 0.0

    if isinstance(signal, SignalRecord):
        sampling_rate_hz = signal.sampling_rate_hz
    if sampling_rate_hz is not None and (
        not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0
    ):
        raise ValueError("sampling_rate_hz must be positive and finite.")

    nonzero_values = values[values != 0.0]
    if nonzero_values.size < 2:
        return 0.0
    signs = np.signbit(nonzero_values)
    crossings = np.count_nonzero(signs[1:] != signs[:-1])

    denominator = values.size - 1
    if sampling_rate_hz is None:
        return float(crossings / denominator)
    return float(crossings * sampling_rate_hz / denominator)


def skewness(signal: SignalRecord | ArrayLike) -> float:
    """Return Fisher-Pearson sample skewness of a signal."""
    values = _values_from_signal(signal)
    if np.all(values == values[0]):
        return 0.0
    return float(stats.skew(values, bias=False))


def kurtosis(signal: SignalRecord | ArrayLike) -> float:
    """Return Pearson kurtosis of a signal, where a normal distribution is 3."""
    values = _values_from_signal(signal)
    if np.all(values == values[0]):
        return 0.0
    return float(stats.kurtosis(values, fisher=False, bias=False))
