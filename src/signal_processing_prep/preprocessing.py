"""Optional preprocessing utilities for signal records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

from signal_processing_prep.config import FilteringConfig
from signal_processing_prep.records import SignalRecord


@dataclass(frozen=True)
class FilterSpec:
    """Digital filter configuration."""

    kind: str
    low_cut_hz: float | None = None
    high_cut_hz: float | None = None
    order: int = 4
    zero_phase: bool = True
    allow_causal_fallback: bool = False


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
                raise ValueError(
                    "Record is too short for zero-phase filtering. Set "
                    "zero_phase=False or allow_causal_fallback=True to use causal filtering."
                ) from error
            filtered_values = scipy_signal.sosfilt(sos, record.values)
            phase_mode = "causal_fallback"
    else:
        filtered_values = scipy_signal.sosfilt(sos, record.values)
        phase_mode = "causal"

    return SignalRecord(
        values=np.asarray(filtered_values, dtype=np.float64),
        sampling_rate_hz=record.sampling_rate_hz,
        label=record.label,
        name=record.name,
        metadata={
            **record.metadata,
            "preprocessing": {
                "filter_kind": kind,
                "low_cut_hz": spec.low_cut_hz,
                "high_cut_hz": spec.high_cut_hz,
                "order": spec.order,
                "phase_mode": phase_mode,
                "allow_causal_fallback": spec.allow_causal_fallback,
            },
        },
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
        ),
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
    return SignalRecord(
        values=windowed_values,
        sampling_rate_hz=record.sampling_rate_hz,
        label=record.label,
        name=record.name,
        metadata={
            **record.metadata,
            "window": {
                "name": _window_name(window),
                "normalize_power": normalize_power,
            },
        },
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
        raise ValueError("step_seconds must not exceed window_seconds.")
    if window_samples > record.n_samples and not include_partial:
        raise ValueError("window_seconds must not exceed the record duration.")

    windows: list[SignalRecord] = []
    for start in range(0, record.n_samples, step_samples):
        end = start + window_samples
        if end > record.n_samples:
            if not include_partial:
                break
            end = record.n_samples
        if end <= start:
            break
        windows.append(_window_record(record, start, end, len(windows)))
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
        raise ValueError("Filter kind must be 'lowpass', 'highpass', or 'bandpass'.")
    if spec.order <= 0:
        raise ValueError("Filter order must be positive.")
    nyquist_hz = sampling_rate_hz / 2.0
    if kind in {"highpass", "bandpass"}:
        _validate_cutoff(spec.low_cut_hz, nyquist_hz, "low_cut_hz")
    if kind in {"lowpass", "bandpass"}:
        _validate_cutoff(spec.high_cut_hz, nyquist_hz, "high_cut_hz")
    if kind == "bandpass" and spec.low_cut_hz >= spec.high_cut_hz:  # type: ignore[operator]
        raise ValueError("low_cut_hz must be less than high_cut_hz for bandpass filters.")


def _validate_cutoff(cutoff_hz: float | None, nyquist_hz: float, name: str) -> None:
    if cutoff_hz is None:
        raise ValueError(f"{name} is required for this filter kind.")
    if cutoff_hz <= 0:
        raise ValueError(f"{name} must be positive.")
    if cutoff_hz >= nyquist_hz:
        raise ValueError(f"{name} must be below the Nyquist frequency.")

def _seconds_to_samples(seconds: float, sampling_rate_hz: float, *, field_name: str) -> int:
    if seconds <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return max(1, int(round(seconds * sampling_rate_hz)))


def _window_record(
    record: SignalRecord,
    start_index: int,
    end_index: int,
    window_index: int,
) -> SignalRecord:
    start_seconds = start_index / record.sampling_rate_hz
    end_seconds = end_index / record.sampling_rate_hz
    base_name = record.name or "record"
    return SignalRecord(
        values=record.values[start_index:end_index].copy(),
        sampling_rate_hz=record.sampling_rate_hz,
        label=record.label,
        name=f"{base_name}_window_{window_index}",
        metadata={
            **record.metadata,
            "source_name": record.name,
            "window_index": window_index,
            "window_start_seconds": start_seconds,
            "window_end_seconds": end_seconds,
            "window_start_sample": start_index,
            "window_end_sample": end_index,
        },
    )


def _window_name(window: str | tuple[str, float]) -> str:
    if isinstance(window, tuple):
        return str(window[0])
    return window
