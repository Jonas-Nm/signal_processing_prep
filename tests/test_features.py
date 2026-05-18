"""Tests for localized feature extraction."""

import numpy as np
import pytest

from signal_processing_prep.features import (
    FrequencyBand,
    SlidingWindowConfig,
    sliding_window_features,
)
from signal_processing_prep.records import SignalRecord
from signal_processing_prep.synthetic import sine_wave, transient_burst


def test_sliding_window_features_returns_one_row_per_window() -> None:
    """Sliding features expose explicit timing columns and expected metrics."""
    record = sine_wave(
        frequency_hz=10.0,
        duration_seconds=2.0,
        sampling_rate_hz=100.0,
        amplitude=2.0,
        name="tone",
    )
    config = SlidingWindowConfig(
        window_seconds=0.5,
        step_seconds=0.25,
        frequency_bands=(FrequencyBand("tone_band", 8.0, 12.0),),
    )

    features = sliding_window_features(record, config)

    assert features.shape[0] == 7
    assert features["window_start_seconds"].tolist() == pytest.approx(
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    )
    assert "skewness" in features.columns
    assert "dominant_frequency_hz" in features.columns
    assert "band_energy_tone_band" in features.columns
    assert features["frequency_window"].tolist() == ["hann"] * 7
    assert np.allclose(features["rms"], np.sqrt(2.0), rtol=1e-3)
    assert np.allclose(features["dominant_frequency_hz"], 10.0)


def test_sliding_window_frequency_features_can_disable_taper() -> None:
    """Frequency-domain sliding features can use a rectangular window explicitly."""
    record = sine_wave(frequency_hz=10.0, duration_seconds=1.0, sampling_rate_hz=100.0)
    config = SlidingWindowConfig(
        window_seconds=0.5,
        step_seconds=0.5,
        frequency_window=None,
    )

    features = sliding_window_features(record, config)

    assert features["frequency_window"].isna().all()
    assert features["dominant_frequency_hz"].tolist() == pytest.approx([10.0, 10.0])


def test_sliding_window_features_capture_time_localized_transient() -> None:
    """Localized RMS increases only near a transient burst."""
    record = transient_burst(
        duration_seconds=2.0,
        sampling_rate_hz=1000.0,
        burst_frequency_hz=100.0,
        burst_start_seconds=1.0,
        burst_duration_seconds=0.2,
        amplitude=2.0,
    )
    config = SlidingWindowConfig(
        window_seconds=0.2,
        step_seconds=0.1,
        include_frequency_domain=False,
    )

    features = sliding_window_features(record, config)
    active = features["window_center_seconds"].between(1.0, 1.2)
    inactive = features["window_center_seconds"] < 0.8

    assert features.loc[active, "rms"].max() > 1.0
    assert features.loc[inactive, "rms"].max() == 0.0


def test_sliding_window_features_can_extract_frequency_only() -> None:
    """Feature groups can be selected explicitly."""
    record = sine_wave(frequency_hz=25.0, duration_seconds=1.0, sampling_rate_hz=200.0)
    config = SlidingWindowConfig(
        window_seconds=0.5,
        step_seconds=0.5,
        include_time_domain=False,
        include_frequency_domain=True,
    )

    features = sliding_window_features(record, config)

    assert "rms" not in features.columns
    assert "dominant_frequency_hz" in features.columns
    assert features["dominant_frequency_hz"].tolist() == pytest.approx([25.0, 25.0], abs=1.0)


def test_sliding_window_features_rejects_invalid_config() -> None:
    """Invalid localized-feature settings fail clearly."""
    record = sine_wave(duration_seconds=1.0)

    with pytest.raises(ValueError, match="window_seconds must be positive"):
        sliding_window_features(record, SlidingWindowConfig(0.0, 0.1))

    with pytest.raises(ValueError, match="step_seconds must be positive"):
        sliding_window_features(record, SlidingWindowConfig(0.1, 0.0))

    with pytest.raises(ValueError, match="must not exceed"):
        sliding_window_features(record, SlidingWindowConfig(2.0, 0.1))

    with pytest.raises(ValueError, match="At least one"):
        sliding_window_features(
            record,
            SlidingWindowConfig(
                0.1,
                0.1,
                include_time_domain=False,
                include_frequency_domain=False,
            ),
        )


def test_sliding_window_features_handles_raw_record_metadata() -> None:
    """Record name and missing labels are preserved in each feature row."""
    record = SignalRecord(values=np.ones(100), sampling_rate_hz=100.0, name="constant")
    config = SlidingWindowConfig(window_seconds=0.5, step_seconds=0.5)

    features = sliding_window_features(record, config)

    assert features["record_name"].tolist() == ["constant", "constant"]
    assert features["label"].isna().all()
