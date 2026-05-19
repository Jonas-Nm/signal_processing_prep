"""Tests for localized feature extraction."""

import numpy as np
import pytest

from signal_processing_prep.features import (
    FeatureExtractionConfig,
    FrequencyBand,
    SlidingWindowConfig,
    extract_features,
    frequency_bands_from_mapping,
    sliding_window_features,
)
from signal_processing_prep.records import SignalRecord
from signal_processing_prep.synthetic import sine_wave, transient_burst


def test_extract_features_returns_one_row_per_signal_record() -> None:
    """Dataset feature extraction returns one row per input record."""
    records = [
        sine_wave(
            frequency_hz=10.0,
            duration_seconds=1.0,
            sampling_rate_hz=200.0,
            label="normal",
            name="tone-10",
        ),
        sine_wave(
            frequency_hz=30.0,
            duration_seconds=1.0,
            sampling_rate_hz=200.0,
            label="fault",
            name="tone-30",
        ),
    ]
    config = FeatureExtractionConfig(
        frequency_bands=(FrequencyBand("low", 0.0, 20.0), FrequencyBand("mid", 20.0, 60.0)),
        spectrogram_window_seconds=0.2,
    )

    features = extract_features(records, config)

    assert features.shape[0] == 2
    assert features["record_name"].tolist() == ["tone-10", "tone-30"]
    assert features["label"].tolist() == ["normal", "fault"]
    assert features["dominant_frequency_hz"].tolist() == pytest.approx([10.0, 30.0])


def test_extract_features_includes_expected_phase_five_columns() -> None:
    """Record-level features include time, frequency, and time-frequency summaries."""
    record = sine_wave(
        frequency_hz=25.0,
        duration_seconds=1.0,
        sampling_rate_hz=200.0,
        amplitude=2.0,
    )

    features = extract_features(
        [record],
        FeatureExtractionConfig(frequency_bands=(FrequencyBand("tone", 20.0, 30.0),)),
    )

    expected_columns = {
        "mean",
        "std",
        "rms",
        "min",
        "max",
        "peak_to_peak",
        "crest_factor",
        "skewness",
        "kurtosis",
        "zero_crossing_rate",
        "dominant_frequency_hz",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_rolloff_85_hz",
        "spectral_flatness",
        "spectral_entropy",
        "band_energy_tone",
        "mean_spectrogram_energy",
        "max_spectrogram_energy",
        "high_frequency_transient_energy",
    }
    assert expected_columns.issubset(features.columns)
    assert features.loc[0, "rms"] == pytest.approx(np.sqrt(2.0), rel=1e-3)
    assert features.loc[0, "band_energy_tone"] > 1.5


def test_extract_features_preserves_missing_labels_for_small_dataset() -> None:
    """Small unlabeled datasets are handled without special casing."""
    record = SignalRecord(values=np.ones(16), sampling_rate_hz=16.0, name="constant")

    features = extract_features([record])

    assert features.shape[0] == 1
    assert features.loc[0, "record_name"] == "constant"
    assert features["label"].isna().all()
    assert features.loc[0, "duration_seconds"] == 1.0


def test_extract_features_rejects_nonfinite_samples_before_dsp() -> None:
    """Feature extraction fails clearly instead of propagating NaN/Inf values."""
    record = SignalRecord(values=np.array([0.0, np.nan, 1.0]), sampling_rate_hz=10.0)

    with pytest.raises(ValueError, match="non-finite samples"):
        extract_features([record])


def test_extract_features_handles_very_low_rate_short_records() -> None:
    """Tiny valid records keep explicit feature columns instead of failing indirectly."""
    record = SignalRecord(values=np.ones(5), sampling_rate_hz=5.0, name="tiny")

    features = extract_features([record])

    assert features.shape[0] == 1
    assert features.loc[0, "record_name"] == "tiny"
    assert np.isfinite(features.loc[0, "mean_spectrogram_energy"])


def test_extract_features_keeps_out_of_range_band_column() -> None:
    """Generic high-frequency bands remain explicit when above Nyquist."""
    record = sine_wave(frequency_hz=5.0, duration_seconds=1.0, sampling_rate_hz=50.0)

    features = extract_features(
        [record],
        FeatureExtractionConfig(frequency_bands=(FrequencyBand("too_high", 30.0, 40.0),)),
    )

    assert "band_energy_too_high" in features.columns
    assert np.isnan(features.loc[0, "band_energy_too_high"])


def test_extract_features_rejects_invalid_config() -> None:
    """Invalid record-level feature settings fail clearly."""
    record = sine_wave(duration_seconds=1.0)

    with pytest.raises(ValueError, match="spectrogram_window_seconds must be positive"):
        extract_features(
            [record],
            FeatureExtractionConfig(spectrogram_window_seconds=0.0),
        )

    with pytest.raises(ValueError, match="spectrogram_step_seconds must be positive"):
        extract_features(
            [record],
            FeatureExtractionConfig(spectrogram_step_seconds=0.0),
        )

    with pytest.raises(ValueError, match="high_frequency_cutoff_hz must be non-negative"):
        extract_features(
            [record],
            FeatureExtractionConfig(high_frequency_cutoff_hz=-1.0),
        )

    with pytest.raises(ValueError, match="low_hz must be non-negative"):
        extract_features(
            [record],
            FeatureExtractionConfig(frequency_bands=(FrequencyBand("bad", -1.0, 10.0),)),
        )

    with pytest.raises(ValueError, match="high_hz must be greater"):
        extract_features(
            [record],
            FeatureExtractionConfig(frequency_bands=(FrequencyBand("bad", 10.0, 10.0),)),
        )


def test_extract_features_reports_nan_when_valid_step_exceeds_short_record_window() -> None:
    """Valid global spectrogram settings remain usable for very short records."""
    record = sine_wave(duration_seconds=0.05, sampling_rate_hz=1000.0)

    features = extract_features(
        [record],
        FeatureExtractionConfig(
            spectrogram_window_seconds=0.1,
            spectrogram_step_seconds=0.075,
        ),
    )

    assert np.isnan(features.loc[0, "mean_spectrogram_energy"])
    assert np.isnan(features.loc[0, "max_spectrogram_energy"])
    assert np.isnan(features.loc[0, "high_frequency_transient_energy"])


def test_frequency_bands_from_mapping_converts_config_shape() -> None:
    """Config frequency-band mappings can feed feature extraction directly."""
    bands = frequency_bands_from_mapping({"bearing": (100.0, 300.0)})

    assert bands == (FrequencyBand("bearing", 100.0, 300.0),)


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

    low_rate_record = sine_wave(duration_seconds=1.0, sampling_rate_hz=10.0)
    with pytest.raises(ValueError, match="step_seconds must not exceed"):
        sliding_window_features(low_rate_record, SlidingWindowConfig(0.04, 0.05))

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
