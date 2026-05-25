"""Signal quality checks for sampled time-series records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from signal_processing_prep.artifacts import QualityTable
from signal_processing_prep.config import QualityConfig
from signal_processing_prep.records import SignalRecord


@dataclass(frozen=True)
class QualityReport:
    """Interpretable quality summary for one signal record."""

    record_name: str | None
    label: str | None
    n_samples: int
    sampling_rate_hz: float
    duration_seconds: float
    missing_fraction: float
    has_missing_values: bool
    max_abs_amplitude: float
    mean: float
    std: float
    rms: float
    clipping_fraction: float
    is_clipped: bool
    is_near_constant: bool
    has_large_amplitude: bool
    stationarity_mean_drift: float
    stationarity_std_cv: float
    is_likely_nonstationary: bool
    time_step_jitter_fraction: float
    time_gap_count: int
    has_time_axis_irregularity: bool
    sampling_rate_mismatch_fraction: float
    has_sampling_rate_mismatch: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        """Return the report as a flat dictionary suitable for tabulation."""
        return {
            "record_name": self.record_name,
            "label": self.label,
            "n_samples": self.n_samples,
            "sampling_rate_hz": self.sampling_rate_hz,
            "duration_seconds": self.duration_seconds,
            "missing_fraction": self.missing_fraction,
            "has_missing_values": self.has_missing_values,
            "max_abs_amplitude": self.max_abs_amplitude,
            "mean": self.mean,
            "std": self.std,
            "rms": self.rms,
            "clipping_fraction": self.clipping_fraction,
            "is_clipped": self.is_clipped,
            "is_near_constant": self.is_near_constant,
            "has_large_amplitude": self.has_large_amplitude,
            "stationarity_mean_drift": self.stationarity_mean_drift,
            "stationarity_std_cv": self.stationarity_std_cv,
            "is_likely_nonstationary": self.is_likely_nonstationary,
            "time_step_jitter_fraction": self.time_step_jitter_fraction,
            "time_gap_count": self.time_gap_count,
            "has_time_axis_irregularity": self.has_time_axis_irregularity,
            "sampling_rate_mismatch_fraction": self.sampling_rate_mismatch_fraction,
            "has_sampling_rate_mismatch": self.has_sampling_rate_mismatch,
            "issues": "; ".join(self.issues),
        }


def assess_signal_quality(
    record: SignalRecord,
    config: QualityConfig | None = None,
) -> QualityReport:
    """Assess duration, missing values, clipping, scaling, and stationarity."""
    if config is None:
        config = QualityConfig()

    values = record.values
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    missing_fraction = float(1.0 - np.count_nonzero(finite_mask) / values.size)
    has_missing_values = missing_fraction > 0.0

    if finite_values.size == 0:
        mean = std = rms = max_abs = float("nan")
        clipping_fraction = 0.0
        is_near_constant = False
        has_large_amplitude = False
        stationarity_mean_drift = float("nan")
        stationarity_std_cv = float("nan")
        is_likely_nonstationary = False
    else:
        mean = float(np.mean(finite_values))
        std = float(np.std(finite_values))
        rms = float(np.sqrt(np.mean(np.square(finite_values))))
        max_abs = float(np.max(np.abs(finite_values)))
        clipping_fraction = _clipping_fraction(finite_values, config.clipping_tolerance)
        is_near_constant = std <= config.near_constant_std_threshold
        has_large_amplitude = (
            config.large_amplitude_threshold is not None
            and max_abs > config.large_amplitude_threshold
        )
        stationarity_mean_drift, stationarity_std_cv = _stationarity_indicators(
            values,
            record.sampling_rate_hz,
            config.stationarity_window_seconds,
        )
        is_likely_nonstationary = (
            stationarity_mean_drift > config.stationarity_mean_drift_threshold
            or stationarity_std_cv > config.stationarity_std_cv_threshold
        )

    time_step_jitter_fraction, time_gap_count, has_time_axis_irregularity = (
        _time_axis_indicators(record, config)
    )
    sampling_rate_mismatch_fraction, has_sampling_rate_mismatch = (
        _sampling_rate_mismatch_indicators(record, config)
    )
    is_clipped = clipping_fraction >= config.clipping_fraction_threshold
    issues = _quality_issues(
        record,
        config,
        has_missing_values=has_missing_values,
        is_clipped=is_clipped,
        is_near_constant=is_near_constant,
        has_large_amplitude=has_large_amplitude,
        is_likely_nonstationary=is_likely_nonstationary,
        has_time_axis_irregularity=has_time_axis_irregularity,
        has_sampling_rate_mismatch=has_sampling_rate_mismatch,
        all_values_missing=finite_values.size == 0,
    )

    return QualityReport(
        record_name=record.name,
        label=record.label,
        n_samples=record.n_samples,
        sampling_rate_hz=record.sampling_rate_hz,
        duration_seconds=record.duration_seconds,
        missing_fraction=missing_fraction,
        has_missing_values=has_missing_values,
        max_abs_amplitude=max_abs,
        mean=mean,
        std=std,
        rms=rms,
        clipping_fraction=clipping_fraction,
        is_clipped=is_clipped,
        is_near_constant=is_near_constant,
        has_large_amplitude=has_large_amplitude,
        stationarity_mean_drift=stationarity_mean_drift,
        stationarity_std_cv=stationarity_std_cv,
        is_likely_nonstationary=is_likely_nonstationary,
        time_step_jitter_fraction=time_step_jitter_fraction,
        time_gap_count=time_gap_count,
        has_time_axis_irregularity=has_time_axis_irregularity,
        sampling_rate_mismatch_fraction=sampling_rate_mismatch_fraction,
        has_sampling_rate_mismatch=has_sampling_rate_mismatch,
        issues=tuple(issues),
    )


def assess_dataset_quality(
    records: Sequence[SignalRecord],
    config: QualityConfig | None = None,
) -> QualityTable:
    """Return typed quality observations for signal records."""
    columns = list(_quality_report_columns())
    if len(records) == 0:
        return QualityTable(pd.DataFrame(columns=columns))
    return QualityTable(
        pd.DataFrame(
            [assess_signal_quality(record, config).to_dict() for record in records],
            columns=columns,
        )
    )


@dataclass(frozen=True)
class QualityAssessor:
    """Pipeline collaborator producing typed quality observations."""

    config: QualityConfig = QualityConfig()

    def assess(self, records: Sequence[SignalRecord]) -> QualityTable:
        """Assess all supplied records through the configured thresholds."""
        return assess_dataset_quality(records, self.config)


def _quality_report_columns() -> tuple[str, ...]:
    return (
        "record_name",
        "label",
        "n_samples",
        "sampling_rate_hz",
        "duration_seconds",
        "missing_fraction",
        "has_missing_values",
        "max_abs_amplitude",
        "mean",
        "std",
        "rms",
        "clipping_fraction",
        "is_clipped",
        "is_near_constant",
        "has_large_amplitude",
        "stationarity_mean_drift",
        "stationarity_std_cv",
        "is_likely_nonstationary",
        "time_step_jitter_fraction",
        "time_gap_count",
        "has_time_axis_irregularity",
        "sampling_rate_mismatch_fraction",
        "has_sampling_rate_mismatch",
        "issues",
    )


def _clipping_fraction(values: np.ndarray, tolerance: float) -> float:
    max_value = float(np.max(values))
    min_value = float(np.min(values))
    if np.isclose(max_value, min_value, atol=tolerance, rtol=0.0):
        return 0.0
    upper = np.isclose(values, max_value, atol=tolerance, rtol=0.0)
    lower = np.isclose(values, min_value, atol=tolerance, rtol=0.0)
    clipped_mask = _plateau_mask(upper) | _plateau_mask(lower)
    return float(np.count_nonzero(clipped_mask) / values.size)


def _stationarity_indicators(
    values: np.ndarray,
    sampling_rate_hz: float,
    window_seconds: float | None,
) -> tuple[float, float]:
    if window_seconds is None:
        window_samples = max(1, values.size // 4)
    else:
        window_samples = int(round(window_seconds * sampling_rate_hz))
    if window_samples <= 0 or values.size < 2 * window_samples:
        return 0.0, 0.0

    means: list[float] = []
    stds: list[float] = []
    for start in range(0, values.size - window_samples + 1, window_samples):
        segment = values[start : start + window_samples]
        finite_segment = segment[np.isfinite(segment)]
        if finite_segment.size == 0:
            continue
        means.append(float(np.mean(finite_segment)))
        stds.append(float(np.std(finite_segment)))

    if len(means) < 2:
        return 0.0, 0.0
    finite_values = values[np.isfinite(values)]
    global_std = float(np.std(finite_values)) if finite_values.size else 0.0
    mean_drift = float((max(means) - min(means)) / global_std) if global_std > 0 else 0.0
    mean_std = float(np.mean(stds))
    std_cv = float(np.std(stds) / mean_std) if mean_std > 0 else 0.0
    return mean_drift, std_cv


def _time_axis_indicators(
    record: SignalRecord,
    config: QualityConfig,
) -> tuple[float, int, bool]:
    jitter = record.acquisition.time_step_jitter_fraction
    gap_count = record.acquisition.time_gap_count
    if jitter is None:
        jitter = 0.0
    if gap_count is None:
        gap_count = 0
    irregular = (
        jitter > config.time_step_jitter_threshold
        or gap_count > config.time_gap_count_threshold
        or record.acquisition.time_axis_valid is False
    )
    return jitter, gap_count, irregular


def _sampling_rate_mismatch_indicators(
    record: SignalRecord,
    config: QualityConfig,
) -> tuple[float, bool]:
    mismatch_fraction = record.acquisition.sampling_rate_mismatch_fraction
    if mismatch_fraction is None:
        mismatch_fraction = 0.0
    return mismatch_fraction, mismatch_fraction > config.sampling_rate_mismatch_threshold


def _plateau_mask(mask: np.ndarray) -> np.ndarray:
    plateau = np.zeros_like(mask, dtype=bool)
    start: int | None = None
    for index, is_active in enumerate(mask):
        if is_active and start is None:
            start = index
        if start is not None and (not is_active or index == mask.size - 1):
            end = index + 1 if is_active and index == mask.size - 1 else index
            if end - start >= 2:
                plateau[start:end] = True
            start = None
    return plateau


def _quality_issues(
    record: SignalRecord,
    config: QualityConfig,
    *,
    has_missing_values: bool,
    is_clipped: bool,
    is_near_constant: bool,
    has_large_amplitude: bool,
    is_likely_nonstationary: bool,
    has_time_axis_irregularity: bool,
    has_sampling_rate_mismatch: bool,
    all_values_missing: bool,
) -> list[str]:
    issues: list[str] = []
    if config.min_duration_seconds is not None and (
        record.duration_seconds < config.min_duration_seconds
    ):
        issues.append("duration_below_minimum")
    if has_missing_values:
        issues.append("missing_values")
    if all_values_missing:
        issues.append("all_values_missing")
    if is_clipped:
        issues.append("possible_clipping")
    if is_near_constant:
        issues.append("near_constant")
    if has_large_amplitude:
        issues.append("large_amplitude")
    if is_likely_nonstationary:
        issues.append("nonstationarity_indicator")
    if has_time_axis_irregularity:
        issues.append("time_axis_irregularity")
    if has_sampling_rate_mismatch:
        issues.append("sampling_rate_mismatch")
    return issues
