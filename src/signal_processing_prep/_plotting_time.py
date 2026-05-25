"""Time-domain plotting and interactive navigation."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider

from signal_processing_prep.records import SignalRecord


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
    time, values = windowed_data(
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


@dataclass(eq=False)
class _AdaptiveTimeSignalPlot:
    record: SignalRecord
    fig: Figure
    ax: Axes
    line: object
    max_points: int
    downsample_method: str
    callback_id: int | None = None
    is_updating: bool = False

    def update_to_xlim(self, _axes: Axes | None = None) -> None:
        if self.is_updating:
            return
        start_seconds, duration_seconds = _visible_window_from_xlim(self.record, self.ax.get_xlim())
        time, values = windowed_data(
            self.record,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            max_points=self.max_points,
            downsample_method=self.downsample_method,
        )
        self.is_updating = True
        try:
            self.line.set_data(time, values)
            _set_padded_ylim(self.ax, values)
            self.fig.canvas.draw_idle()
        finally:
            self.is_updating = False


def plot_time_signal_adaptive(
    record: SignalRecord,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    max_points: int = 2000,
    downsample_method: str = "envelope",
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot a signal with zoom-aware display downsampling."""
    if max_points <= 0:
        raise ValueError("max_points must be positive.")
    time, values = windowed_data(
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
    (line,) = ax.plot(time, values, linewidth=1.0, label=record.label or record.name or "signal")
    ax.set_title(_time_plot_title(record, start_seconds, duration_seconds))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time[0], time[-1])
    _set_padded_ylim(ax, values)
    if record.label is not None:
        ax.legend(loc="best")
    existing = getattr(ax, "_signal_processing_prep_adaptive_time_plot", None)
    if existing is not None and existing.callback_id is not None:
        ax.callbacks.disconnect(existing.callback_id)
    controller = _AdaptiveTimeSignalPlot(
        record=record,
        fig=fig,
        ax=ax,
        line=line,
        max_points=max_points,
        downsample_method=downsample_method,
    )
    controller.callback_id = ax.callbacks.connect("xlim_changed", controller.update_to_xlim)
    setattr(ax, "_signal_processing_prep_adaptive_time_plot", controller)
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
        self.slider.set_val(_clamp_start(start_seconds, self.record.duration_seconds, self.window_seconds))

    def step_previous(self, _event: object | None = None) -> None:
        """Move the visible window one window length backward."""
        self.set_start_seconds(float(self.slider.val) - self.window_seconds)

    def step_next(self, _event: object | None = None) -> None:
        """Move the visible window one window length forward."""
        self.set_start_seconds(float(self.slider.val) + self.window_seconds)

    def _update(self, start_seconds: float) -> None:
        time, values = windowed_data(
            self.record,
            start_seconds=start_seconds,
            duration_seconds=self.window_seconds,
            max_points=self.max_points,
            downsample_method=self.downsample_method,
        )
        self.line.set_data(time, values)
        self.ax.set_xlim(time[0], time[-1])
        _set_padded_ylim(self.ax, values)
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
    time, values = windowed_data(
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
    slider = Slider(slider_ax, "Start [s]", 0.0, max_start, valinit=start_seconds)
    previous_button = Button(previous_ax, "Previous")
    next_button = Button(next_ax, "Next")
    navigator = TimeSignalNavigator(
        record, fig, ax, slider, previous_button, next_button, window_seconds, max_points, downsample_method, line
    )
    slider.on_changed(navigator._update)
    previous_button.on_clicked(navigator.step_previous)
    next_button.on_clicked(navigator.step_next)
    if show:
        plt.show()
    return navigator


def windowed_data(
    record: SignalRecord,
    *,
    start_seconds: float | None,
    duration_seconds: float | None,
    max_points: int | None,
    downsample_method: str = "envelope",
) -> tuple[np.ndarray, np.ndarray]:
    """Select a display window and optionally reduce only plotted samples."""
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


def _time_plot_title(record: SignalRecord, start_seconds: float | None, duration_seconds: float | None) -> str:
    title = record.name or "Signal"
    if record.label is not None:
        title = f"{title} ({record.label})"
    title = f"{title} - {record.sampling_rate_hz:g} Hz"
    if start_seconds is not None or duration_seconds is not None:
        start = 0.0 if start_seconds is None else start_seconds
        title = f"{title}, window from {start:g} s"
    return title


def _visible_window_from_xlim(record: SignalRecord, xlim: tuple[float, float]) -> tuple[float, float]:
    x_min, x_max = sorted(float(limit) for limit in xlim)
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return 0.0, record.duration_seconds
    start_seconds = min(max(x_min, 0.0), record.duration_seconds)
    end_seconds = min(max(x_max, 0.0), record.duration_seconds)
    sample_period_seconds = 1.0 / record.sampling_rate_hz
    if end_seconds <= start_seconds:
        end_seconds = min(start_seconds + sample_period_seconds, record.duration_seconds)
        start_seconds = min(start_seconds, max(record.duration_seconds - sample_period_seconds, 0.0))
    return start_seconds, max(end_seconds - start_seconds, sample_period_seconds)


def _set_padded_ylim(ax: Axes, values: np.ndarray) -> None:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return
    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    padding = max((y_max - y_min) * 0.05, 1e-9)
    ax.set_ylim(y_min - padding, y_max + padding)


def _downsample_for_plot(
    time: np.ndarray, values: np.ndarray, max_points: int, *, method: str
) -> tuple[np.ndarray, np.ndarray]:
    if values.size <= max_points:
        return time, values
    if method == "stride":
        stride = max(1, int(np.ceil(values.size / max_points)))
        return time[::stride], values[::stride]
    if method != "envelope":
        raise ValueError("downsample_method must be 'envelope' or 'stride'.")
    edges = np.linspace(0, values.size, max(1, max_points // 2) + 1, dtype=int)
    plot_time: list[float] = []
    plot_values: list[float] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        segment = values[start:end]
        segment_time = time[start:end]
        for index in sorted({int(np.nanargmin(segment)), int(np.nanargmax(segment))}):
            plot_time.append(float(segment_time[index]))
            plot_values.append(float(segment[index]))
    return np.asarray(plot_time), np.asarray(plot_values)


def _seconds_to_index(record: SignalRecord, seconds: float) -> int:
    if seconds < 0:
        raise ValueError("start_seconds must be non-negative.")
    return min(int(round(seconds * record.sampling_rate_hz)), record.n_samples - 1)


def _clamp_start(start_seconds: float, duration_seconds: float, window_seconds: float) -> float:
    return min(max(start_seconds, 0.0), max(duration_seconds - window_seconds, 0.0))
