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
class SlidingWindowConfig:
    """Configuration for time-localized feature extraction."""

    window_seconds: float
    step_seconds: float
    frequency_bands: Sequence[FrequencyBand] = field(default_factory=tuple)
    include_time_domain: bool = True
    include_frequency_domain: bool = True
    frequency_window: str | tuple[str, float] | None = "hann"
    normalize_frequency_window_power: bool = True


def sliding_window_features(
    record: SignalRecord,
    config: SlidingWindowConfig,
) -> pd.DataFrame:
    """Extract interpretable features over sliding time windows.

    The returned DataFrame has one row per window and includes explicit timing
    columns so feature trajectories can be plotted or aligned with events.
    """
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
        features[f"band_energy_{band.name}"] = band_energy(
            frequency_values,
            low_hz=band.low_hz,
            high_hz=band.high_hz,
            sampling_rate_hz=sampling_rate_hz,
        )
    return features


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
    sample_count = int(round(seconds * sampling_rate_hz))
    if sample_count <= 0:
        raise ValueError(f"{field_name} produces no samples.")
    return sample_count
