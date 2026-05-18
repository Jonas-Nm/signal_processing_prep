"""Synthetic signal generators for examples and tests."""

from __future__ import annotations

from collections.abc import Sequence

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


def window_signal(
    *,
    duration_seconds: float = 1.0,
    sampling_rate_hz: float = 1000.0,
    window_start_seconds: float = 0.0,
    window_duration_seconds: float | None = None,
    window_type: str = "rectangular",
    amplitude: float = 1.0,
    label: str | None = "window",
    name: str | None = None,
) -> SignalRecord:
    """Generate a window embedded in a zero-valued signal record.

    Supported window types are ``rectangular``, ``hann``, ``hamming``,
    ``blackman``, and ``bartlett``. ``duration_seconds`` is the total record
    duration; ``window_start_seconds`` and ``window_duration_seconds`` define
    where the active window lies inside that record.
    """
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative.")
    if window_start_seconds < 0:
        raise ValueError("window_start_seconds must be non-negative.")

    time = _time_axis(duration_seconds, sampling_rate_hz)
    if window_duration_seconds is None:
        window_duration_seconds = duration_seconds - window_start_seconds
    if window_duration_seconds <= 0:
        raise ValueError("window_duration_seconds must be positive.")
    if window_start_seconds + window_duration_seconds > duration_seconds:
        raise ValueError("Window must fit within duration_seconds.")

    start_index = int(round(window_start_seconds * sampling_rate_hz))
    end_index = int(round((window_start_seconds + window_duration_seconds) * sampling_rate_hz))
    start_index = min(start_index, time.size)
    end_index = min(max(end_index, start_index), time.size)
    window_size = end_index - start_index
    if window_size <= 0:
        raise ValueError("Window parameters produce no active samples.")

    window_type = window_type.lower()
    if window_type in {"rectangular", "rectangle", "boxcar"}:
        window_values = np.ones(window_size, dtype=np.float64)
        canonical_type = "rectangular"
    elif window_type == "hann":
        window_values = np.hanning(window_size)
        canonical_type = "hann"
    elif window_type == "hamming":
        window_values = np.hamming(window_size)
        canonical_type = "hamming"
    elif window_type == "blackman":
        window_values = np.blackman(window_size)
        canonical_type = "blackman"
    elif window_type == "bartlett":
        window_values = np.bartlett(window_size)
        canonical_type = "bartlett"
    else:
        raise ValueError(
            "window_type must be one of 'rectangular', 'hann', 'hamming', "
            "'blackman', or 'bartlett'."
        )

    values = np.zeros(time.size, dtype=np.float64)
    values[start_index:end_index] = amplitude * window_values
    return SignalRecord(
        values,
        sampling_rate_hz,
        label=label,
        name=name or f"{canonical_type}_window",
        metadata={
            "window_type": canonical_type,
            "window_start_seconds": window_start_seconds,
            "window_duration_seconds": window_duration_seconds,
            "amplitude": amplitude,
        },
    )


def sinc_signal(
    *,
    duration_seconds: float = 1.0,
    sampling_rate_hz: float = 1000.0,
    bandwidth_hz: float = 50.0,
    center_seconds: float | None = None,
    amplitude: float = 1.0,
    label: str | None = "sinc",
    name: str | None = "sinc_signal",
) -> SignalRecord:
    """Generate a normalized sinc signal centered in time by default.

    The generated values follow ``amplitude * sinc(2 * bandwidth_hz * t)``,
    where ``t`` is measured relative to ``center_seconds``.
    """
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive.")

    time = _time_axis(duration_seconds, sampling_rate_hz)
    if center_seconds is None:
        center_seconds = 0.5 * (time[0] + time[-1])
    if center_seconds < 0 or center_seconds > duration_seconds:
        raise ValueError("center_seconds must be within the signal duration.")

    centered_time = time - center_seconds
    values = amplitude * np.sinc(2.0 * bandwidth_hz * centered_time)
    return SignalRecord(
        values,
        sampling_rate_hz,
        label=label,
        name=name,
        metadata={
            "bandwidth_hz": bandwidth_hz,
            "center_seconds": center_seconds,
            "amplitude": amplitude,
        },
    )


def add_signals(
    records: Sequence[SignalRecord],
    *,
    label: str | None = "synthetic_sum",
    name: str | None = "synthetic_sum",
) -> SignalRecord:
    """Add same-length synthetic signals sample by sample."""
    _validate_compatible_records(records, require_same_length=True)
    first = records[0]
    values = np.sum([record.values for record in records], axis=0)
    return SignalRecord(
        values=values,
        sampling_rate_hz=first.sampling_rate_hz,
        label=label,
        name=name,
        metadata=_composition_metadata("add", records),
    )


def multiply_signals(
    records: Sequence[SignalRecord],
    *,
    label: str | None = "synthetic_product",
    name: str | None = "synthetic_product",
) -> SignalRecord:
    """Multiply same-length synthetic signals sample by sample."""
    _validate_compatible_records(records, require_same_length=True)
    first = records[0]
    values = np.prod([record.values for record in records], axis=0)
    return SignalRecord(
        values=values,
        sampling_rate_hz=first.sampling_rate_hz,
        label=label,
        name=name,
        metadata=_composition_metadata("multiply", records),
    )


def convolve_signals(
    first: SignalRecord,
    second: SignalRecord,
    *,
    mode: str = "same",
    label: str | None = "synthetic_convolution",
    name: str | None = "synthetic_convolution",
) -> SignalRecord:
    """Convolve two synthetic signals using ``numpy.convolve`` modes."""
    _validate_compatible_records([first, second], require_same_length=False)
    if mode not in {"full", "same", "valid"}:
        raise ValueError("mode must be one of 'full', 'same', or 'valid'.")

    values = np.convolve(first.values, second.values, mode=mode)
    return SignalRecord(
        values=values,
        sampling_rate_hz=first.sampling_rate_hz,
        label=label,
        name=name,
        metadata={
            **_composition_metadata("convolve", [first, second]),
            "mode": mode,
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


def _validate_compatible_records(
    records: Sequence[SignalRecord],
    *,
    require_same_length: bool,
) -> None:
    if len(records) == 0:
        raise ValueError("At least one SignalRecord is required.")

    sampling_rate_hz = records[0].sampling_rate_hz
    for record in records:
        if not np.isclose(record.sampling_rate_hz, sampling_rate_hz):
            raise ValueError("All records must have the same sampling_rate_hz.")

    if require_same_length:
        n_samples = records[0].n_samples
        if any(record.n_samples != n_samples for record in records):
            raise ValueError("All records must have the same number of samples.")


def _composition_metadata(
    operation: str,
    records: Sequence[SignalRecord],
) -> dict[str, object]:
    return {
        "operation": operation,
        "source_names": [record.name for record in records],
        "source_labels": [record.label for record in records],
        "source_sample_counts": [record.n_samples for record in records],
    }


# create a main function for testing

if __name__ == "__main__":
    time = _time_axis(duration_seconds=2.0, sampling_rate_hz=1000.0)
    print(time[:10])
    
