"""Tests for shared signal record structures."""

import numpy as np
import pytest

from signal_processing_prep.records import SignalRecord


def test_signal_record_normalizes_values_and_exposes_shape() -> None:
    """SignalRecord converts values to a float array and exposes basic metadata."""
    record = SignalRecord(
        values=[0, 1, 0, -1],
        sampling_rate_hz=4.0,
        label="normal",
        name="example",
        metadata={"sensor": "accelerometer"},
    )

    assert record.values.dtype == np.float64
    assert record.n_samples == 4
    assert record.duration_seconds == 1.0
    assert record.label == "normal"
    assert record.name == "example"
    assert record.metadata["sensor"] == "accelerometer"
    np.testing.assert_allclose(record.time_seconds, [0.0, 0.25, 0.5, 0.75])


def test_signal_record_rejects_invalid_values() -> None:
    """SignalRecord rejects empty, multidimensional, and invalid sampling-rate data."""
    with pytest.raises(ValueError, match="must not be empty"):
        SignalRecord(values=[], sampling_rate_hz=1.0)

    with pytest.raises(ValueError, match="one-dimensional"):
        SignalRecord(values=[[1.0, 2.0]], sampling_rate_hz=1.0)

    with pytest.raises(ValueError, match="positive"):
        SignalRecord(values=[1.0], sampling_rate_hz=0.0)
