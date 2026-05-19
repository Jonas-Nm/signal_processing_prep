"""Feature extraction helpers for signal records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from signal_processing_prep.frequency_domain import (
    band_energy,
    dominant_frequency,
    fft_magnitude,
    spectral_bandwidth,
    spectral_centroid,
    spectral_flatness,
    spectral_rolloff,
)
from signal_processing_prep.records import SignalRecord
from signal_processing_prep.time_frequency import spectrogram_analysis
from signal_processing_prep.time_domain import (
    crest_factor,
    kurtosis,
    rms,
    skewness,
    zero_crossing_rate,
)


@dataclass(frozen=True)
class FrequencyBand:
    """Named frequency range for band-energy features."""

    name: str
    low_hz: float
    high_hz: float


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Configuration for one-row-per-record feature extraction."""

    frequency_bands: Sequence[FrequencyBand] = field(default_factory=tuple)
    spectrogram_window_seconds: float = 0.1
    spectrogram_step_seconds: float | None = None
    high_frequency_cutoff_hz: float | None = None


@dataclass(frozen=True)
class SlidingWindowConfig:
    """Configuration for time-localized feature extraction."""

    window_seconds: float
    step_seconds: float
    frequency_bands: Sequence[FrequencyBand] = field(default_factory=tuple)
    include_time_domain: bool = True
    include_frequency_domain: bool = True
    frequency_window: str | tuple[str, float] | None = "hann"
    normalize_frequency_window_power: bool = True


def extract_features(
    records: Sequence[SignalRecord],
    config: FeatureExtractionConfig | None = None,
) -> pd.DataFrame:
    """Extract one row of interpretable features per signal record.

    Frequency bands are always represented as columns. Bands that sit entirely
    above a record's Nyquist frequency are reported as ``NaN`` for that record,
    which keeps generic configurations usable across mixed sampling rates.
    """
    if config is None:
        config = FeatureExtractionConfig()
    _validate_feature_config(config)

    for record in records:
        _validate_feature_record(record)
    rows = [_record_features(record, config) for record in records]
    return pd.DataFrame(rows)


def sliding_window_features(
    record: SignalRecord,
    config: SlidingWindowConfig,
) -> pd.DataFrame:
    """Extract interpretable features over sliding time windows.

    The returned DataFrame has one row per window and includes explicit timing
    columns so feature trajectories can be plotted or aligned with events.
    """
    _validate_feature_record(record)
    window_samples = _seconds_to_sample_count(
        config.window_seconds,
        record.sampling_rate_hz,
        field_name="window_seconds",
    )
    step_samples = _seconds_to_sample_count(
        config.step_seconds,
        record.sampling_rate_hz,
        field_name="step_seconds",
    )
    if config.step_seconds > config.window_seconds:
        raise ValueError("step_seconds must not exceed window_seconds.")
    if window_samples > record.n_samples:
        raise ValueError("window_seconds must not exceed the record duration.")
    if not config.include_time_domain and not config.include_frequency_domain:
        raise ValueError("At least one feature group must be enabled.")

    rows: list[dict[str, float | str | None]] = []
    for start_index in range(0, record.n_samples - window_samples + 1, step_samples):
        end_index = start_index + window_samples
        window_values = record.values[start_index:end_index]
        start_seconds = start_index / record.sampling_rate_hz
        end_seconds = end_index / record.sampling_rate_hz
        row: dict[str, float | str | None] = {
            "record_name": record.name,
            "label": record.label,
            "window_start_seconds": start_seconds,
            "window_end_seconds": end_seconds,
            "window_center_seconds": 0.5 * (start_seconds + end_seconds),
            "window_n_samples": float(window_samples),
            "frequency_window": _window_name(config.frequency_window),
        }

        if config.include_time_domain:
            row.update(_time_domain_features(window_values, record.sampling_rate_hz))
        if config.include_frequency_domain:
            row.update(
                _frequency_domain_features(
                    window_values,
                    record.sampling_rate_hz,
                    config.frequency_bands,
                    config.frequency_window,
                    config.normalize_frequency_window_power,
                )
            )
        rows.append(row)

    return pd.DataFrame(rows)


def frequency_bands_from_mapping(
    bands_hz: dict[str, tuple[float, float]],
) -> tuple[FrequencyBand, ...]:
    """Convert a configuration mapping into ``FrequencyBand`` objects."""
    return tuple(
        FrequencyBand(name=name, low_hz=low_hz, high_hz=high_hz)
        for name, (low_hz, high_hz) in bands_hz.items()
    )


def _record_features(
    record: SignalRecord,
    config: FeatureExtractionConfig,
) -> dict[str, float | str | None]:
    row: dict[str, float | str | None] = {
        "record_name": record.name,
        "label": record.label,
        "sampling_rate_hz": record.sampling_rate_hz,
        "n_samples": float(record.n_samples),
        "duration_seconds": record.duration_seconds,
    }
    row.update(_time_domain_features(record.values, record.sampling_rate_hz))
    row.update(
        _frequency_domain_features(
            record.values,
            record.sampling_rate_hz,
            config.frequency_bands,
            frequency_window=None,
            normalize_window_power=False,
        )
    )
    row["spectral_entropy"] = _spectral_entropy(record)
    row.update(_time_frequency_features(record, config))
    return row


def _time_domain_features(
    values: np.ndarray,
    sampling_rate_hz: float,
) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "peak_to_peak": float(np.ptp(values)),
        "rms": rms(values),
        "crest_factor": crest_factor(values),
        "zero_crossing_rate": zero_crossing_rate(values, sampling_rate_hz=sampling_rate_hz),
        "skewness": skewness(values),
        "kurtosis": kurtosis(values),
    }


def _frequency_domain_features(
    values: np.ndarray,
    sampling_rate_hz: float,
    frequency_bands: Sequence[FrequencyBand],
    frequency_window: str | tuple[str, float] | None,
    normalize_window_power: bool,
) -> dict[str, float]:
    frequency_values = _apply_frequency_window(
        values,
        frequency_window,
        normalize_power=normalize_window_power,
    )
    spectrum = fft_magnitude(frequency_values, sampling_rate_hz=sampling_rate_hz)
    features = {
        "dominant_frequency_hz": dominant_frequency(spectrum),
        "spectral_centroid_hz": spectral_centroid(spectrum),
        "spectral_bandwidth_hz": spectral_bandwidth(spectrum),
        "spectral_rolloff_85_hz": spectral_rolloff(spectrum, rolloff_fraction=0.85),
        "spectral_flatness": spectral_flatness(spectrum),
    }
    for band in frequency_bands:
        features[f"band_energy_{band.name}"] = _safe_band_energy(
            frequency_values,
            sampling_rate_hz,
            band,
        )
    return features


def _safe_band_energy(
    values: np.ndarray,
    sampling_rate_hz: float,
    band: FrequencyBand,
) -> float:
    nyquist_hz = sampling_rate_hz / 2.0
    if band.low_hz >= nyquist_hz:
        return float("nan")
    return band_energy(
        values,
        low_hz=band.low_hz,
        high_hz=min(band.high_hz, nyquist_hz),
        sampling_rate_hz=sampling_rate_hz,
    )


def _spectral_entropy(record: SignalRecord) -> float:
    spectrum = fft_magnitude(record)
    weights = np.square(spectrum.magnitudes)
    total = float(np.sum(weights))
    if total == 0.0:
        return 0.0
    probabilities = weights / total
    positive_probabilities = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(positive_probabilities * np.log2(positive_probabilities)))
    max_entropy = np.log2(weights.size) if weights.size > 1 else 1.0
    return float(entropy / max_entropy)


def _time_frequency_features(
    record: SignalRecord,
    config: FeatureExtractionConfig,
) -> dict[str, float]:
    window_seconds = min(config.spectrogram_window_seconds, record.duration_seconds)
    if _spectrogram_step_exceeds_window(config.spectrogram_step_seconds, window_seconds):
        return {
            "mean_spectrogram_energy": float("nan"),
            "max_spectrogram_energy": float("nan"),
            "high_frequency_transient_energy": float("nan"),
        }
    spectrogram = spectrogram_analysis(
        record,
        window_seconds=window_seconds,
        step_seconds=config.spectrogram_step_seconds,
    )

    power = spectrogram.power
    high_frequency_cutoff = config.high_frequency_cutoff_hz
    if high_frequency_cutoff is None:
        high_frequency_cutoff = 0.75 * (record.sampling_rate_hz / 2.0)
    high_frequency_mask = spectrogram.frequencies_hz >= high_frequency_cutoff
    if np.any(high_frequency_mask):
        high_frequency_energy = float(np.max(np.sum(power[high_frequency_mask, :], axis=0)))
    else:
        high_frequency_energy = 0.0

    return {
        "mean_spectrogram_energy": float(np.mean(power)),
        "max_spectrogram_energy": float(np.max(power)),
        "high_frequency_transient_energy": high_frequency_energy,
    }


def _validate_feature_config(config: FeatureExtractionConfig) -> None:
    if config.spectrogram_window_seconds <= 0:
        raise ValueError("spectrogram_window_seconds must be positive.")
    if config.spectrogram_step_seconds is not None and config.spectrogram_step_seconds <= 0:
        raise ValueError("spectrogram_step_seconds must be positive.")
    if config.high_frequency_cutoff_hz is not None and config.high_frequency_cutoff_hz < 0:
        raise ValueError("high_frequency_cutoff_hz must be non-negative.")

    for band in config.frequency_bands:
        if band.low_hz < 0:
            raise ValueError(f"Frequency band '{band.name}' low_hz must be non-negative.")
        if band.high_hz <= band.low_hz:
            raise ValueError(
                f"Frequency band '{band.name}' high_hz must be greater than low_hz."
            )


def _validate_feature_record(record: SignalRecord) -> None:
    if not np.isfinite(record.values).all():
        raise ValueError(
            f"Record {record.name or '<unnamed>'} contains non-finite samples; "
            "run quality checks and clean, impute, or skip the record before feature extraction."
        )


def _spectrogram_step_exceeds_window(
    step_seconds: float | None,
    window_seconds: float,
) -> bool:
    return step_seconds is not None and step_seconds > window_seconds


def _apply_frequency_window(
    values: np.ndarray,
    window: str | tuple[str, float] | None,
    *,
    normalize_power: bool,
) -> np.ndarray:
    if window is None:
        return values
    weights = scipy_signal.get_window(window, values.size, fftbins=True).astype(np.float64)
    if normalize_power:
        mean_square = float(np.mean(np.square(weights)))
        if mean_square > 0.0:
            weights = weights / np.sqrt(mean_square)
    return values * weights


def _window_name(window: str | tuple[str, float] | None) -> str | None:
    if window is None:
        return None
    if isinstance(window, tuple):
        return str(window[0])
    return window


def _seconds_to_sample_count(
    seconds: float,
    sampling_rate_hz: float,
    *,
    field_name: str,
) -> int:
    if seconds <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return max(1, int(round(seconds * sampling_rate_hz)))
