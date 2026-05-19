"""Plotting helpers for signal exploration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider

from signal_processing_prep.frequency_domain import fft_magnitude, psd
from signal_processing_prep.records import SignalRecord
from signal_processing_prep.time_frequency import spectrogram_analysis


def plot_time_signal(
    record: SignalRecord,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    max_points: int | None = None,
    downsample_method: str = "envelope",
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot one signal record in the time domain."""
    time, values = _windowed_data(
        record,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        max_points=max_points,
        downsample_method=downsample_method,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(time, values, linewidth=1.0, label=record.label or record.name or "signal")
    ax.set_title(_time_plot_title(record, start_seconds, duration_seconds))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    if record.label is not None:
        ax.legend(loc="best")
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


@dataclass
class TimeSignalNavigator:
    """Interactive matplotlib controls for exploring one time-domain signal."""

    record: SignalRecord
    fig: Figure
    ax: Axes
    slider: Slider
    previous_button: Button
    next_button: Button
    window_seconds: float
    max_points: int | None
    downsample_method: str
    line: object

    def set_start_seconds(self, start_seconds: float) -> None:
        """Move the visible window to a new start time."""
        start = _clamp_start(start_seconds, self.record.duration_seconds, self.window_seconds)
        self.slider.set_val(start)

    def step_previous(self, _event: object | None = None) -> None:
        """Move the visible window one window length backward."""
        self.set_start_seconds(float(self.slider.val) - self.window_seconds)

    def step_next(self, _event: object | None = None) -> None:
        """Move the visible window one window length forward."""
        self.set_start_seconds(float(self.slider.val) + self.window_seconds)

    def _update(self, start_seconds: float) -> None:
        time, values = _windowed_data(
            self.record,
            start_seconds=start_seconds,
            duration_seconds=self.window_seconds,
            max_points=self.max_points,
            downsample_method=self.downsample_method,
        )
        self.line.set_data(time, values)
        self.ax.set_xlim(time[0], time[-1])
        if values.size > 0:
            y_min = float(np.nanmin(values))
            y_max = float(np.nanmax(values))
            padding = max((y_max - y_min) * 0.05, 1e-9)
            self.ax.set_ylim(y_min - padding, y_max + padding)
        self.fig.canvas.draw_idle()


def plot_time_signal_navigator(
    record: SignalRecord,
    *,
    window_seconds: float = 1.0,
    start_seconds: float = 0.0,
    max_points: int | None = None,
    downsample_method: str = "envelope",
    show: bool = False,
) -> TimeSignalNavigator:
    """Create an interactive time-domain navigator for one signal record."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points must be positive.")

    window_seconds = min(window_seconds, record.duration_seconds)
    start_seconds = _clamp_start(start_seconds, record.duration_seconds, window_seconds)
    time, values = _windowed_data(
        record,
        start_seconds=start_seconds,
        duration_seconds=window_seconds,
        max_points=max_points,
        downsample_method=downsample_method,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.25)
    (line,) = ax.plot(time, values, linewidth=1.0)
    ax.set_title(f"{record.name or 'Signal'} - interactive time navigator")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time[0], time[-1])

    slider_ax = fig.add_axes([0.18, 0.10, 0.64, 0.03])
    previous_ax = fig.add_axes([0.18, 0.04, 0.12, 0.04])
    next_ax = fig.add_axes([0.70, 0.04, 0.12, 0.04])

    max_start = max(record.duration_seconds - window_seconds, 0.0)
    slider = Slider(
        ax=slider_ax,
        label="Start [s]",
        valmin=0.0,
        valmax=max_start,
        valinit=start_seconds,
    )
    previous_button = Button(previous_ax, "Previous")
    next_button = Button(next_ax, "Next")

    navigator = TimeSignalNavigator(
        record=record,
        fig=fig,
        ax=ax,
        slider=slider,
        previous_button=previous_button,
        next_button=next_button,
        window_seconds=window_seconds,
        max_points=max_points,
        downsample_method=downsample_method,
        line=line,
    )
    slider.on_changed(navigator._update)
    previous_button.on_clicked(navigator.step_previous)
    next_button.on_clicked(navigator.step_next)

    if show:
        plt.show()
    return navigator


def plot_frequency_spectrum(
    record: SignalRecord,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    spectrum_type: str = "fft",
    max_frequency_hz: float | None = None,
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot FFT magnitude or PSD for one signal record window."""
    frequencies, values, ylabel = _windowed_spectrum(
        record,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        spectrum_type=spectrum_type,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(frequencies, values, linewidth=1.0, label=record.label or record.name or "signal")
    ax.set_title(_frequency_plot_title(record, start_seconds, duration_seconds, spectrum_type))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if max_frequency_hz is not None:
        if max_frequency_hz <= 0:
            raise ValueError("max_frequency_hz must be positive.")
        ax.set_xlim(0.0, max_frequency_hz)
    if record.label is not None:
        ax.legend(loc="best")
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def plot_frequency_spectra(
    records: list[SignalRecord],
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    spectrum_type: str = "fft",
    max_frequency_hz: float | None = None,
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot FFT magnitude or PSD for multiple signal records."""
    if len(records) == 0:
        raise ValueError("At least one SignalRecord is required.")
    if max_frequency_hz is not None and max_frequency_hz <= 0:
        raise ValueError("max_frequency_hz must be positive.")

    spectrum_type = _validate_spectrum_type(spectrum_type)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ylabel = "Magnitude" if spectrum_type == "fft" else "PSD [amplitude^2 / Hz]"
    for record in records:
        frequencies, values, ylabel = _windowed_spectrum(
            record,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            spectrum_type=spectrum_type,
        )
        ax.plot(frequencies, values, linewidth=1.0, label=record.name or record.label or "signal")

    spectrum_name = "FFT magnitude" if spectrum_type == "fft" else "PSD"
    ax.set_title(_frequency_comparison_title(spectrum_name, start_seconds, duration_seconds))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if max_frequency_hz is not None:
        ax.set_xlim(0.0, max_frequency_hz)
    ax.legend(loc="best")
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def plot_spectrogram(
    record: SignalRecord,
    *,
    window_seconds: float = 0.1,
    step_seconds: float | None = None,
    max_frequency_hz: float | None = None,
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot a power spectrogram for one signal record."""
    if max_frequency_hz is not None and max_frequency_hz <= 0:
        raise ValueError("max_frequency_hz must be positive.")
    result = spectrogram_analysis(
        record,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )
    frequencies = result.frequencies_hz
    power = result.power
    if max_frequency_hz is not None:
        mask = frequencies <= max_frequency_hz
        frequencies = frequencies[mask]
        power = power[mask, :]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(
        result.times_seconds,
        frequencies,
        10.0 * np.log10(np.maximum(power, 1e-24)),
        shading="auto",
    )
    ax.set_title(f"{record.name or 'Signal'} - spectrogram")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("Power [dB]")
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def plot_feature_distribution(
    features: pd.DataFrame,
    feature: str,
    *,
    label_column: str | None = "label",
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot one feature distribution, optionally grouped by label."""
    if feature not in features.columns:
        raise ValueError(f"Feature column not found: {feature}")
    if label_column is not None and label_column not in features.columns:
        raise ValueError(f"Label column not found: {label_column}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    if label_column is None or features[label_column].isna().all():
        values = pd.to_numeric(features[feature], errors="coerce").dropna()
        ax.hist(values, bins=min(20, max(5, values.size)), alpha=0.75)
    else:
        for label, group in features.groupby(label_column, dropna=True):
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            if not values.empty:
                ax.hist(values, bins=min(20, max(5, values.size)), alpha=0.55, label=str(label))
        ax.legend(loc="best")

    ax.set_title(f"{feature} distribution")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    *,
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot a labeled confusion matrix."""
    matrix = np.asarray(matrix)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("matrix shape must match the number of labels.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    threshold = float(np.max(matrix)) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > threshold else "black"
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def plot_feature_importance(
    importances: pd.Series | Mapping[str, float],
    *,
    top_n: int = 20,
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot feature importances sorted by absolute importance."""
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    series = pd.Series(importances, dtype=float).dropna()
    if series.empty:
        raise ValueError("At least one feature importance value is required.")
    series = series.reindex(series.abs().sort_values(ascending=False).index).head(top_n)
    series = series.sort_values()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(series))))
    else:
        fig = ax.figure

    ax.barh(series.index.astype(str), series.values)
    ax.set_title("Feature importance")
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def save_figure(
    fig: Figure,
    figures_dir: str | Path = "reports/figures",
    *,
    stem: str,
    run_date: date | str | None = None,
    dpi: int = 150,
) -> Path:
    """Save a figure under a date-stamped reports/figures directory."""
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    if run_date is None:
        date_part = date.today().isoformat()
    elif isinstance(run_date, date):
        date_part = run_date.isoformat()
    else:
        date_part = run_date

    output_dir = Path(figures_dir) / date_part
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_filename(stem)}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def _time_plot_title(
    record: SignalRecord,
    start_seconds: float | None,
    duration_seconds: float | None,
) -> str:
    title = record.name or "Signal"
    if record.label is not None:
        title = f"{title} ({record.label})"
    title = f"{title} - {record.sampling_rate_hz:g} Hz"
    if start_seconds is not None or duration_seconds is not None:
        start = 0.0 if start_seconds is None else start_seconds
        title = f"{title}, window from {start:g} s"
    return title


def _frequency_plot_title(
    record: SignalRecord,
    start_seconds: float | None,
    duration_seconds: float | None,
    spectrum_type: str,
) -> str:
    spectrum_name = "FFT magnitude" if spectrum_type == "fft" else "PSD"
    title = f"{record.name or 'Signal'} - {spectrum_name}"
    if start_seconds is not None or duration_seconds is not None:
        start = 0.0 if start_seconds is None else start_seconds
        title = f"{title}, window from {start:g} s"
    return title


def _frequency_comparison_title(
    spectrum_name: str,
    start_seconds: float | None,
    duration_seconds: float | None,
) -> str:
    title = f"{spectrum_name} comparison"
    if start_seconds is not None or duration_seconds is not None:
        start = 0.0 if start_seconds is None else start_seconds
        title = f"{title}, window from {start:g} s"
    return title


def _windowed_spectrum(
    record: SignalRecord,
    *,
    start_seconds: float | None,
    duration_seconds: float | None,
    spectrum_type: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    _, values = _windowed_data(
        record,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        max_points=None,
    )
    spectrum_type = _validate_spectrum_type(spectrum_type)
    if spectrum_type == "fft":
        spectrum = fft_magnitude(values, sampling_rate_hz=record.sampling_rate_hz)
        return spectrum.frequencies_hz, spectrum.magnitudes, "Magnitude"

    power_spectrum = psd(values, sampling_rate_hz=record.sampling_rate_hz)
    return power_spectrum.frequencies_hz, power_spectrum.power, "PSD [amplitude^2 / Hz]"


def _validate_spectrum_type(spectrum_type: str) -> str:
    if spectrum_type not in {"fft", "psd"}:
        raise ValueError("spectrum_type must be 'fft' or 'psd'.")
    return spectrum_type


def _windowed_data(
    record: SignalRecord,
    *,
    start_seconds: float | None,
    duration_seconds: float | None,
    max_points: int | None,
    downsample_method: str = "envelope",
) -> tuple[np.ndarray, np.ndarray]:
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points must be positive.")

    start_index = 0 if start_seconds is None else _seconds_to_index(record, start_seconds)
    if duration_seconds is None:
        end_index = record.n_samples
    else:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive.")
        end_index = start_index + max(1, int(round(duration_seconds * record.sampling_rate_hz)))
    end_index = min(max(end_index, start_index + 1), record.n_samples)

    time = record.time_seconds[start_index:end_index]
    values = record.values[start_index:end_index]
    if max_points is None:
        return time, values
    return _downsample_for_plot(time, values, max_points, method=downsample_method)


def _downsample_for_plot(
    time: np.ndarray,
    values: np.ndarray,
    max_points: int,
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if values.size <= max_points:
        return time, values
    if method == "stride":
        stride = max(1, int(np.ceil(values.size / max_points)))
        return time[::stride], values[::stride]
    if method != "envelope":
        raise ValueError("downsample_method must be 'envelope' or 'stride'.")

    n_bins = max(1, max_points // 2)
    edges = np.linspace(0, values.size, n_bins + 1, dtype=int)
    plot_time: list[float] = []
    plot_values: list[float] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        segment = values[start:end]
        segment_time = time[start:end]
        min_index = int(np.nanargmin(segment))
        max_index = int(np.nanargmax(segment))
        for index in sorted({min_index, max_index}):
            plot_time.append(float(segment_time[index]))
            plot_values.append(float(segment[index]))

    return np.asarray(plot_time), np.asarray(plot_values)


def _seconds_to_index(record: SignalRecord, seconds: float) -> int:
    if seconds < 0:
        raise ValueError("start_seconds must be non-negative.")
    index = int(round(seconds * record.sampling_rate_hz))
    return min(index, record.n_samples - 1)


def _clamp_start(start_seconds: float, duration_seconds: float, window_seconds: float) -> float:
    return min(max(start_seconds, 0.0), max(duration_seconds - window_seconds, 0.0))


def _safe_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    safe = safe.strip("_")
    if not safe:
        raise ValueError("stem must contain at least one filename-safe character.")
    return safe
