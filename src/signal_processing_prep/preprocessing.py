"""Optional preprocessing utilities for signal records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

from signal_processing_prep.config import FilteringConfig
from signal_processing_prep.errors import ConfigurationError, RecordDataError
from signal_processing_prep.records import ProcessingStep, SignalRecord


@dataclass(frozen=True)
class FilterSpec:
    """Digital filter configuration."""

    kind: str
    low_cut_hz: float | None = None
    high_cut_hz: float | None = None
    order: int = 4
    zero_phase: bool = True
    allow_causal_fallback: bool = False

    def __post_init__(self) -> None:
        """Validate static filter fields when a specialist constructs a filter."""
        FilteringConfig(
            enabled=True,
            kind=self.kind,
            low_cut_hz=self.low_cut_hz,
            high_cut_hz=self.high_cut_hz,
            order=self.order,
            zero_phase=self.zero_phase,
            allow_causal_fallback=self.allow_causal_fallback,
        )


def apply_filter(record: SignalRecord, spec: FilterSpec) -> SignalRecord:
    """Apply an optional low-pass, high-pass, or band-pass Butterworth filter."""
    _validate_filter_spec(spec, record.sampling_rate_hz)
    kind = spec.kind.lower()
    nyquist_hz = record.sampling_rate_hz / 2.0

    if kind == "lowpass":
        cutoff: float | tuple[float, float] = spec.high_cut_hz / nyquist_hz  # type: ignore[operator]
        btype = "lowpass"
    elif kind == "highpass":
        cutoff = spec.low_cut_hz / nyquist_hz  # type: ignore[operator]
        btype = "highpass"
    else:
        cutoff = (spec.low_cut_hz / nyquist_hz, spec.high_cut_hz / nyquist_hz)  # type: ignore[operator]
        btype = "bandpass"

    sos = scipy_signal.butter(spec.order, cutoff, btype=btype, output="sos")
    if spec.zero_phase:
        try:
            filtered_values = scipy_signal.sosfiltfilt(sos, record.values)
            phase_mode = "zero_phase"
        except ValueError as error:
            if not spec.allow_causal_fallback:
                raise RecordDataError(
                    "Record is too short for zero-phase filtering. Set "
                    "zero_phase=False or allow_causal_fallback=True to use causal filtering."
                ) from error
            filtered_values = scipy_signal.sosfilt(sos, record.values)
            phase_mode = "causal_fallback"
    else:
        filtered_values = scipy_signal.sosfilt(sos, record.values)
        phase_mode = "causal"

    parameters = {
        "filter_kind": kind,
        "low_cut_hz": spec.low_cut_hz,
        "high_cut_hz": spec.high_cut_hz,
        "order": spec.order,
        "phase_mode": phase_mode,
        "allow_causal_fallback": spec.allow_causal_fallback,
    }
    return record.with_values(
        np.asarray(filtered_values, dtype=np.float64),
        processing_step=ProcessingStep("filter", parameters),
    )


def apply_configured_filter(
    record: SignalRecord,
    config: FilteringConfig,
) -> SignalRecord:
    """Apply a loaded filtering config, or return the record unchanged if disabled."""
    if not config.enabled:
        return record
    if config.kind is None:
        raise ValueError("Filtering kind is required when filtering is enabled.")
    return apply_filter(
        record,
        FilterSpec(
            kind=config.kind,
            low_cut_hz=config.low_cut_hz,
            high_cut_hz=config.high_cut_hz,
            order=config.order,
            zero_phase=config.zero_phase,
            allow_causal_fallback=config.allow_causal_fallback,
        ),
    )


def interpolate_missing_values(
    record: SignalRecord,
    *,
    method: str = "linear",
    max_missing_fraction: float = 0.01,
) -> SignalRecord:
    """Interpolate isolated non-finite samples in a signal record.

    This helper is intended for small, explicit repairs before frequency-domain
    analysis. Larger gaps should usually be segmented, skipped, or investigated
    rather than silently filled.
    """
    if method != "linear":
        raise ConfigurationError("Only linear interpolation is currently supported.")
    if not 0.0 <= max_missing_fraction <= 1.0:
        raise ConfigurationError("max_missing_fraction must be between 0 and 1.")

    values = record.values
    finite_mask = np.isfinite(values)
    missing_count = int(values.size - np.count_nonzero(finite_mask))
    missing_fraction = float(missing_count / values.size)
    if missing_count == 0:
        return record
    if missing_fraction > max_missing_fraction:
        raise RecordDataError(
            "Missing fraction exceeds max_missing_fraction; skip, segment, or "
            "choose a more explicit repair strategy."
        )
    if not np.any(finite_mask):
        raise RecordDataError("Cannot interpolate a record with no finite samples.")

    sample_indices = np.arange(values.size, dtype=np.float64)
    interpolated_values = np.interp(
        sample_indices,
        sample_indices[finite_mask],
        values[finite_mask],
    )

    parameters: dict[str, object] = {
        "method": method,
        "missing_count": missing_count,
        "missing_fraction": missing_fraction,
        "max_missing_fraction": max_missing_fraction,
    }
    return record.with_values(
        np.asarray(interpolated_values, dtype=np.float64),
        processing_step=ProcessingStep("missing_value_interpolation", parameters),
    )


def apply_window(
    record: SignalRecord,
    *,
    window: str | tuple[str, float] = "hann",
    normalize_power: bool = False,
) -> SignalRecord:
    """Multiply a signal by a named scipy window."""
    weights = scipy_signal.get_window(window, record.n_samples, fftbins=True).astype(np.float64)
    if normalize_power:
        mean_square = float(np.mean(np.square(weights)))
        if mean_square > 0.0:
            weights = weights / np.sqrt(mean_square)
    windowed_values = record.values * weights
    window_parameters = {
        "name": _window_name(window),
        "normalize_power": normalize_power,
    }
    return record.with_values(
        windowed_values,
        processing_step=ProcessingStep("window", window_parameters),
    )


def segment_signal(
    record: SignalRecord,
    *,
    window_seconds: float,
    step_seconds: float | None = None,
    include_partial: bool = False,
) -> list[SignalRecord]:
    """Split a record into explicit fixed-duration windows."""
    window_samples = _seconds_to_samples(
        window_seconds,
        record.sampling_rate_hz,
        field_name="window_seconds",
    )
    if step_seconds is None:
        step_seconds = window_seconds
    step_samples = _seconds_to_samples(
        step_seconds,
        record.sampling_rate_hz,
        field_name="step_seconds",
    )
    if step_seconds > window_seconds:
        raise ConfigurationError("step_seconds must not exceed window_seconds.")
    if window_samples > record.n_samples and not include_partial:
        raise RecordDataError("window_seconds must not exceed the record duration.")

    windows: list[SignalRecord] = []
    for start in range(0, record.n_samples, step_samples):
        end = start + window_samples
        if end > record.n_samples:
            if not include_partial:
                break
            end = record.n_samples
        if end <= start:
            break
        windows.append(record.segment(start, end, len(windows)))
        if end == record.n_samples:
            break
    return windows


def segment_dataset(
    records: Sequence[SignalRecord],
    *,
    window_seconds: float,
    step_seconds: float | None = None,
    include_partial: bool = False,
) -> list[SignalRecord]:
    """Segment multiple records and return a flat list of windows."""
    segments: list[SignalRecord] = []
    for record in records:
        segments.extend(
            segment_signal(
                record,
                window_seconds=window_seconds,
                step_seconds=step_seconds,
                include_partial=include_partial,
            )
        )
    return segments


def _validate_filter_spec(spec: FilterSpec, sampling_rate_hz: float) -> None:
    kind = spec.kind.lower()
    if kind not in {"lowpass", "highpass", "bandpass"}:
        raise ConfigurationError("Filter kind must be 'lowpass', 'highpass', or 'bandpass'.")
    if spec.order <= 0:
        raise ConfigurationError("Filter order must be positive.")
    nyquist_hz = sampling_rate_hz / 2.0
    if kind in {"highpass", "bandpass"}:
        _validate_cutoff(spec.low_cut_hz, nyquist_hz, "low_cut_hz")
    if kind in {"lowpass", "bandpass"}:
        _validate_cutoff(spec.high_cut_hz, nyquist_hz, "high_cut_hz")
    if kind == "bandpass" and spec.low_cut_hz >= spec.high_cut_hz:  # type: ignore[operator]
        raise ConfigurationError("low_cut_hz must be less than high_cut_hz for bandpass filters.")


def _validate_cutoff(cutoff_hz: float | None, nyquist_hz: float, name: str) -> None:
    if cutoff_hz is None:
        raise ConfigurationError(f"{name} is required for this filter kind.")
    if cutoff_hz <= 0:
        raise ConfigurationError(f"{name} must be positive.")
    if cutoff_hz >= nyquist_hz:
        raise RecordDataError(f"{name} must be below the Nyquist frequency.")


def _seconds_to_samples(seconds: float, sampling_rate_hz: float, *, field_name: str) -> int:
    if seconds <= 0:
        raise ConfigurationError(f"{field_name} must be positive.")
    return max(1, int(round(seconds * sampling_rate_hz)))


def _window_name(window: str | tuple[str, float]) -> str:
    if isinstance(window, tuple):
        return str(window[0])
    return window


@dataclass(frozen=True)
class SignalPreprocessor:
    """Configured pipeline collaborator applying explicit preprocessing steps."""

    filtering: FilteringConfig = FilteringConfig()

    def process(self, record: SignalRecord) -> SignalRecord:
        """Apply configured preprocessing to one record."""
        return apply_configured_filter(record, self.filtering)
