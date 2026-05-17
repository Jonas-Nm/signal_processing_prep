"""Synthetic signal generators for examples and tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from signal_processing_prep.records import SignalRecord


def _time_axis(duration_seconds: float, sampling_rate_hz: float) -> NDArray[np.float64]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")

    n_samples = int(round(duration_seconds * sampling_rate_hz))
    if n_samples <= 0:
        raise ValueError("duration_seconds and sampling_rate_hz produce no samples.")
    return np.arange(n_samples, dtype=np.float64) / sampling_rate_hz


def sine_wave(
    *,
    frequency_hz: float = 50.0,
    duration_seconds: float = 1.0,
    sampling_rate_hz: float = 1000.0,
    amplitude: float = 1.0,
    phase_radians: float = 0.0,
    label: str | None = "sine",
    name: str | None = "sine_wave",
) -> SignalRecord:
    """Generate a pure sine-wave signal."""
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive.")

    time = _time_axis(duration_seconds, sampling_rate_hz)
    values = amplitude * np.sin(2.0 * np.pi * frequency_hz * time + phase_radians)
    return SignalRecord(
        values=values,
        sampling_rate_hz=sampling_rate_hz,
        label=label,
        name=name,
        metadata={"frequency_hz": frequency_hz, "amplitude": amplitude},
    )


def noisy_sine_wave(
    *,
    frequency_hz: float = 50.0,
    duration_seconds: float = 1.0,
    sampling_rate_hz: float = 1000.0,
    amplitude: float = 1.0,
    noise_std: float = 0.1,
    seed: int | None = 0,
    label: str | None = "noisy_sine",
    name: str | None = "noisy_sine_wave",
) -> SignalRecord:
    """Generate a sine wave with additive Gaussian noise."""
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    clean = sine_wave(
        frequency_hz=frequency_hz,
        duration_seconds=duration_seconds,
        sampling_rate_hz=sampling_rate_hz,
        amplitude=amplitude,
        label=label,
        name=name,
    )
    rng = np.random.default_rng(seed)
    values = clean.values + rng.normal(0.0, noise_std, size=clean.n_samples)
    metadata = {**clean.metadata, "noise_std": noise_std, "seed": seed}
    return SignalRecord(values, sampling_rate_hz, label=label, name=name, metadata=metadata)


def impulse_train(
    *,
    duration_seconds: float = 1.0,
    sampling_rate_hz: float = 1000.0,
    impulse_rate_hz: float = 20.0,
    amplitude: float = 1.0,
    label: str | None = "impulse_train",
    name: str | None = "impulse_train",
) -> SignalRecord:
    """Generate a sparse impulse-train signal."""
    if impulse_rate_hz <= 0:
        raise ValueError("impulse_rate_hz must be positive.")

    time = _time_axis(duration_seconds, sampling_rate_hz)
    values = np.zeros_like(time)
    spacing = max(1, int(round(sampling_rate_hz / impulse_rate_hz)))
    values[::spacing] = amplitude
    return SignalRecord(
        values,
        sampling_rate_hz,
        label=label,
        name=name,
        metadata={"impulse_rate_hz": impulse_rate_hz, "amplitude": amplitude},
    )


def chirp_signal(
    *,
    start_frequency_hz: float = 20.0,
    end_frequency_hz: float = 200.0,
    duration_seconds: float = 1.0,
    sampling_rate_hz: float = 1000.0,
    amplitude: float = 1.0,
    label: str | None = "chirp",
    name: str | None = "chirp_signal",
) -> SignalRecord:
    """Generate a linear chirp signal."""
    if start_frequency_hz <= 0 or end_frequency_hz <= 0:
        raise ValueError("start_frequency_hz and end_frequency_hz must be positive.")

    time = _time_axis(duration_seconds, sampling_rate_hz)
    frequency_slope = (end_frequency_hz - start_frequency_hz) / duration_seconds
    phase = 2.0 * np.pi * (
        start_frequency_hz * time + 0.5 * frequency_slope * np.square(time)
    )
    values = amplitude * np.sin(phase)
    return SignalRecord(
        values,
        sampling_rate_hz,
        label=label,
        name=name,
        metadata={
            "start_frequency_hz": start_frequency_hz,
            "end_frequency_hz": end_frequency_hz,
            "amplitude": amplitude,
        },
    )


def clipped_signal(
    *,
    frequency_hz: float = 50.0,
    duration_seconds: float = 1.0,
    sampling_rate_hz: float = 1000.0,
    amplitude: float = 1.5,
    clip_limit: float = 1.0,
    label: str | None = "clipped",
    name: str | None = "clipped_signal",
) -> SignalRecord:
    """Generate a sine wave clipped to a fixed absolute limit."""
    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive.")

    base = sine_wave(
        frequency_hz=frequency_hz,
        duration_seconds=duration_seconds,
        sampling_rate_hz=sampling_rate_hz,
        amplitude=amplitude,
        label=label,
        name=name,
    )
    values = np.clip(base.values, -clip_limit, clip_limit)
    metadata = {**base.metadata, "clip_limit": clip_limit}
    return SignalRecord(values, sampling_rate_hz, label=label, name=name, metadata=metadata)


def transient_burst(
    *,
    duration_seconds: float = 1.0,
    sampling_rate_hz: float = 2000.0,
    burst_frequency_hz: float = 500.0,
    burst_start_seconds: float = 0.4,
    burst_duration_seconds: float = 0.05,
    amplitude: float = 1.0,
    label: str | None = "transient_burst",
    name: str | None = "transient_burst",
) -> SignalRecord:
    """Generate a signal with a short high-frequency burst."""
    if burst_frequency_hz <= 0:
        raise ValueError("burst_frequency_hz must be positive.")
    if burst_start_seconds < 0 or burst_duration_seconds <= 0:
        raise ValueError("Burst start must be non-negative and duration must be positive.")

    time = _time_axis(duration_seconds, sampling_rate_hz)
    values = np.zeros_like(time)
    start_index = int(round(burst_start_seconds * sampling_rate_hz))
    end_index = int(round((burst_start_seconds + burst_duration_seconds) * sampling_rate_hz))
    start_index = min(start_index, values.size)
    end_index = min(max(end_index, start_index), values.size)
    burst_time = time[start_index:end_index]
    values[start_index:end_index] = amplitude * np.sin(
        2.0 * np.pi * burst_frequency_hz * burst_time
    )
    return SignalRecord(
        values,
        sampling_rate_hz,
        label=label,
        name=name,
        metadata={
            "burst_frequency_hz": burst_frequency_hz,
            "burst_start_seconds": burst_start_seconds,
            "burst_duration_seconds": burst_duration_seconds,
            "amplitude": amplitude,
        },
    )


def make_synthetic_dataset() -> list[SignalRecord]:
    """Return a small representative synthetic dataset."""
    return [
        sine_wave(),
        noisy_sine_wave(),
        impulse_train(),
        chirp_signal(),
        clipped_signal(),
        transient_burst(),
    ]
