"""Tests for time-domain signal metrics."""

from collections.abc import Callable

import numpy as np
import pytest

from signal_processing_prep.synthetic import sine_wave
from signal_processing_prep.time_domain import (
    crest_factor,
    kurtosis,
    rms,
    skewness,
    zero_crossing_rate,
)


def test_rms_for_known_sine_wave() -> None:
    """A sine wave RMS equals amplitude divided by sqrt(2)."""
    record = sine_wave(
        frequency_hz=10.0,
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
        amplitude=2.0,
    )

    assert rms(record) == pytest.approx(np.sqrt(2.0), rel=1e-3)


def test_crest_factor_for_known_sine_wave() -> None:
    """A sine wave crest factor is sqrt(2)."""
    record = sine_wave(
        frequency_hz=10.0,
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
        amplitude=2.0,
    )

    assert crest_factor(record) == pytest.approx(np.sqrt(2.0), rel=1e-3)


def test_zero_crossing_rate_for_known_sine_wave() -> None:
    """A sine wave crosses zero twice per cycle."""
    record = sine_wave(
        frequency_hz=5.0,
        duration_seconds=2.0,
        sampling_rate_hz=1000.0,
    )

    assert zero_crossing_rate(record) == pytest.approx(10.0, rel=0.05)


def test_shape_metrics_handle_symmetric_and_constant_signals() -> None:
    """Skewness and kurtosis remain finite for common simple signals."""
    symmetric = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    constant = np.ones(10)

    assert skewness(symmetric) == pytest.approx(0.0)
    assert kurtosis(symmetric) > 1.0
    assert skewness(constant) == 0.0
    assert kurtosis(constant) == 0.0
    assert crest_factor(np.zeros(10)) == 0.0


def test_time_domain_helpers_handle_tiny_signals() -> None:
    """Single-sample signals return finite shape and crossing metrics."""
    values = np.array([2.0])

    assert rms(values) == 2.0
    assert crest_factor(values) == 1.0
    assert zero_crossing_rate(values) == 0.0
    assert skewness(values) == 0.0
    assert kurtosis(values) == 0.0


@pytest.mark.parametrize(
    "metric",
    [rms, crest_factor, skewness, kurtosis, zero_crossing_rate],
)
def test_time_domain_helpers_reject_nonfinite_samples(
    metric: Callable[[np.ndarray], float],
) -> None:
    """Time-domain metrics fail clearly for NaN or Inf samples."""
    values = np.array([0.0, np.nan, 1.0])

    with pytest.raises(ValueError, match="non-finite samples"):
        metric(values)

    values = np.array([0.0, np.inf, 1.0])

    with pytest.raises(ValueError, match="non-finite samples"):
        metric(values)


@pytest.mark.parametrize("sampling_rate_hz", [np.nan, np.inf])
def test_zero_crossing_rate_rejects_nonfinite_sampling_rate(sampling_rate_hz: float) -> None:
    """Zero-crossing rates with physical units require finite timing."""
    with pytest.raises(ValueError, match="sampling_rate_hz must be positive and finite"):
        zero_crossing_rate(np.array([-1.0, 1.0]), sampling_rate_hz=sampling_rate_hz)
