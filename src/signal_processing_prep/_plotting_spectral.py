"""Frequency and time-frequency visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

from signal_processing_prep._plotting_time import windowed_data
from signal_processing_prep.frequency_domain import fft_magnitude, psd
from signal_processing_prep.records import SignalRecord
from signal_processing_prep.time_frequency import (
    HilbertHuangResult,
    morlet_wavelet_scalogram,
    spectrogram_analysis,
    teager_kaiser_demodulation,
    teager_kaiser_energy,
)


def plot_frequency_spectrum(
    record: SignalRecord,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    spectrum_type: str = "fft",
    nperseg: int | None = None,
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
        nperseg=nperseg,
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
    nperseg: int | None = None,
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
            nperseg=nperseg,
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
    vmin_db: float | None = None,
    vmax_db: float | None = None,
    frequency_scale: str = "linear",
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot a power spectrogram for one signal record."""
    if max_frequency_hz is not None and max_frequency_hz <= 0:
        raise ValueError("max_frequency_hz must be positive.")
    result = spectrogram_analysis(record, window_seconds=window_seconds, step_seconds=step_seconds)
    frequencies, power = _limited_power(
        result.frequencies_hz, result.power, max_frequency_hz, frequency_scale
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    mesh = ax.pcolormesh(
        result.times_seconds,
        frequencies,
        10.0 * np.log10(np.maximum(power, 1e-24)),
        shading="auto",
        vmin=vmin_db,
        vmax=vmax_db,
    )
    _configure_time_frequency_axes(
        fig,
        ax,
        mesh,
        record.name or "Signal",
        frequency_scale,
        "Power [dB]",
        "spectrogram",
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_teager_kaiser_energy(
    record: SignalRecord,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot the sample-aligned Teager-Kaiser energy trace for one record."""
    result = teager_kaiser_energy(record)
    visible_mask = _analysis_window_mask(
        record,
        result.time_seconds,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    ax.plot(
        result.time_seconds[visible_mask],
        result.energy[visible_mask],
        linewidth=1.0,
        label=record.label or record.name or "signal",
    )
    ax.set_title(f"{record.name or 'Signal'} - Teager-Kaiser energy")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("TKEO energy [amplitude^2]")
    ax.grid(True, alpha=0.3)
    if record.label is not None:
        ax.legend(loc="best")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_teager_kaiser_frequency(
    record: SignalRecord,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot valid DESA-2 instantaneous-frequency estimates for one record."""
    result = teager_kaiser_demodulation(record)
    visible_mask = _analysis_window_mask(
        record,
        result.time_seconds,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )
    frequency_hz = result.instantaneous_frequency_hz.copy()
    frequency_hz[~result.valid_mask] = np.nan
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    ax.plot(
        result.time_seconds[visible_mask],
        frequency_hz[visible_mask],
        linewidth=1.0,
        label=record.label or record.name or "signal",
    )
    ax.set_title(f"{record.name or 'Signal'} - Teager-Kaiser instantaneous frequency")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Instantaneous frequency [Hz]")
    ax.grid(True, alpha=0.3)
    if record.label is not None:
        ax.legend(loc="best")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


@dataclass(eq=False)
class _DynamicRangePlot:
    fig: Figure
    mesh: object
    colorbar: object
    min_slider: Slider
    max_slider: Slider
    minimum_separation_db: float
    is_updating: bool = False

    def update_clim(self, _value: float) -> None:
        """Apply slider values to the image color limits."""
        if self.is_updating:
            return
        vmin = float(self.min_slider.val)
        vmax = float(self.max_slider.val)
        self.is_updating = True
        try:
            if vmin >= vmax:
                if vmin + self.minimum_separation_db <= self.max_slider.valmax:
                    vmax = vmin + self.minimum_separation_db
                    self.max_slider.set_val(vmax)
                else:
                    vmin = vmax - self.minimum_separation_db
                    self.min_slider.set_val(vmin)
            self.mesh.set_clim(vmin, vmax)
            self.colorbar.update_normal(self.mesh)
            self.fig.canvas.draw_idle()
        finally:
            self.is_updating = False


def plot_spectrogram_dynamic_range(
    record: SignalRecord,
    *,
    window_seconds: float = 0.1,
    step_seconds: float | None = None,
    max_frequency_hz: float | None = None,
    vmin_db: float | None = None,
    vmax_db: float | None = None,
    dynamic_range_db: float = 80.0,
    frequency_scale: str = "linear",
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot a spectrogram with interactive dB color-range controls."""
    if dynamic_range_db <= 0:
        raise ValueError("dynamic_range_db must be positive.")
    if max_frequency_hz is not None and max_frequency_hz <= 0:
        raise ValueError("max_frequency_hz must be positive.")
    result = spectrogram_analysis(record, window_seconds=window_seconds, step_seconds=step_seconds)
    frequencies, power = _limited_power(
        result.frequencies_hz, result.power, max_frequency_hz, frequency_scale
    )
    return _dynamic_power_plot(
        record.name or "Signal",
        result.times_seconds,
        frequencies,
        power,
        title_suffix="spectrogram",
        colorbar_label="Power [dB]",
        frequency_scale=frequency_scale,
        vmin_db=vmin_db,
        vmax_db=vmax_db,
        dynamic_range_db=dynamic_range_db,
        ax=ax,
        show=show,
        controller_attribute="_signal_processing_prep_spectrogram_dynamic_range",
    )


def plot_wavelet_scalogram(
    record: SignalRecord,
    *,
    min_frequency_hz: float = 20.0,
    max_frequency_hz: float | None = None,
    n_frequencies: int = 64,
    wavelet: str = "cmor1.5-1.0",
    frequency_scale: str = "log",
    vmin_db: float | None = None,
    vmax_db: float | None = None,
    dynamic_range_db: float = 80.0,
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot a Morlet continuous-wavelet scalogram for one signal record."""
    if dynamic_range_db <= 0:
        raise ValueError("dynamic_range_db must be positive.")
    result = morlet_wavelet_scalogram(
        record,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        n_frequencies=n_frequencies,
        wavelet=wavelet,
    )
    frequencies, power = _apply_frequency_scale(
        result.frequencies_hz, result.power, frequency_scale
    )
    return _dynamic_power_plot(
        record.name or "Signal",
        result.times_seconds,
        frequencies,
        power,
        title_suffix="Morlet wavelet scalogram",
        colorbar_label="Wavelet power [dB]",
        frequency_scale=frequency_scale,
        vmin_db=vmin_db,
        vmax_db=vmax_db,
        dynamic_range_db=dynamic_range_db,
        ax=ax,
        show=show,
        controller_attribute="_signal_processing_prep_wavelet_dynamic_range",
    )


def plot_hilbert_huang_imfs(
    result: HilbertHuangResult,
    *,
    include_residual: bool = True,
    max_imfs: int | None = None,
    show: bool = False,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot HHT intrinsic mode functions and optional residual as stacked traces."""
    if max_imfs is not None and (not isinstance(max_imfs, int) or max_imfs <= 0):
        raise ValueError("max_imfs must be a positive integer or None.")
    n_imfs = result.intrinsic_mode_functions.shape[0]
    n_shown = n_imfs if max_imfs is None else min(n_imfs, max_imfs)
    n_panels = max(n_shown + int(include_residual), 1)
    fig, raw_axes = plt.subplots(n_panels, 1, figsize=(10, 2.2 * n_panels), sharex=True)
    axes = np.atleast_1d(raw_axes)
    for index in range(n_shown):
        axes[index].plot(result.time_seconds, result.intrinsic_mode_functions[index], linewidth=0.9)
        axes[index].set_ylabel(f"IMF {index + 1}")
        axes[index].grid(True, alpha=0.3)
    if include_residual:
        residual_axis = axes[n_shown]
        residual_axis.plot(result.time_seconds, result.residual, linewidth=0.9)
        residual_axis.set_ylabel("Residual")
        residual_axis.grid(True, alpha=0.3)
    if n_shown == 0 and not include_residual:
        axes[0].text(0.5, 0.5, "No oscillatory IMFs extracted", ha="center", va="center")
        axes[0].set_yticks([])
    axes[0].set_title(f"{result.method.upper()} Hilbert-Huang decomposition")
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_hilbert_huang_spectrum(
    result: HilbertHuangResult,
    *,
    frequency_scale: str = "linear",
    dynamic_range_db: float = 80.0,
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot a binned Hilbert energy spectrum from an existing HHT result."""
    if dynamic_range_db <= 0:
        raise ValueError("dynamic_range_db must be positive.")
    frequencies, power = _apply_frequency_scale(
        result.spectrum_frequencies_hz,
        result.spectrum_power,
        frequency_scale,
    )
    return _dynamic_power_plot(
        f"{result.method.upper()} HHT",
        result.time_seconds,
        frequencies,
        power,
        title_suffix="Hilbert energy spectrum",
        colorbar_label="Hilbert energy [dB]",
        frequency_scale=frequency_scale,
        vmin_db=None,
        vmax_db=None,
        dynamic_range_db=dynamic_range_db,
        ax=ax,
        show=show,
        controller_attribute="_signal_processing_prep_hht_dynamic_range",
    )


def _dynamic_power_plot(
    display_name: str,
    times_seconds: np.ndarray,
    frequencies: np.ndarray,
    power: np.ndarray,
    *,
    title_suffix: str,
    colorbar_label: str,
    frequency_scale: str,
    vmin_db: float | None,
    vmax_db: float | None,
    dynamic_range_db: float,
    ax: Axes | None,
    show: bool,
    controller_attribute: str,
) -> tuple[Figure, Axes]:
    power_db = 10.0 * np.log10(np.maximum(power, 1e-24))
    initial_vmin, initial_vmax = _spectrogram_color_limits(
        power_db, vmin_db=vmin_db, vmax_db=vmax_db, dynamic_range_db=dynamic_range_db
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.subplots_adjust(bottom=0.25)
    else:
        fig = ax.figure
    mesh = ax.pcolormesh(
        times_seconds, frequencies, power_db, shading="auto", vmin=initial_vmin, vmax=initial_vmax
    )
    colorbar = _configure_time_frequency_axes(
        fig, ax, mesh, display_name, frequency_scale, colorbar_label, title_suffix
    )
    slider_min, slider_max = _spectrogram_slider_bounds(power_db, initial_vmin, initial_vmax)
    separation = max((slider_max - slider_min) * 1e-6, 1e-6)
    min_slider = Slider(fig.add_axes([0.18, 0.09, 0.64, 0.03]), "Min dB", slider_min, slider_max, valinit=initial_vmin)
    max_slider = Slider(fig.add_axes([0.18, 0.04, 0.64, 0.03]), "Max dB", slider_min, slider_max, valinit=initial_vmax)
    controller = _DynamicRangePlot(fig, mesh, colorbar, min_slider, max_slider, separation)
    min_slider.on_changed(controller.update_clim)
    max_slider.on_changed(controller.update_clim)
    setattr(ax, controller_attribute, controller)
    if show:
        plt.show()
    return fig, ax


def _configure_time_frequency_axes(
    fig: Figure,
    ax: Axes,
    mesh: object,
    display_name: str,
    frequency_scale: str,
    colorbar_label: str,
    title_suffix: str,
) -> object:
    ax.set_title(f"{display_name} - {title_suffix}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_yscale(frequency_scale)
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(colorbar_label)
    return colorbar


def _windowed_spectrum(
    record: SignalRecord,
    *,
    start_seconds: float | None,
    duration_seconds: float | None,
    spectrum_type: str,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    _, values = windowed_data(
        record, start_seconds=start_seconds, duration_seconds=duration_seconds, max_points=None
    )
    spectrum_type = _validate_spectrum_type(spectrum_type)
    if spectrum_type == "fft":
        spectrum = fft_magnitude(values, sampling_rate_hz=record.sampling_rate_hz)
        return spectrum.frequencies_hz, spectrum.magnitudes, "Magnitude"
    power_spectrum = psd(values, sampling_rate_hz=record.sampling_rate_hz, nperseg=nperseg)
    return power_spectrum.frequencies_hz, power_spectrum.power, "PSD [amplitude^2 / Hz]"


def _analysis_window_mask(
    record: SignalRecord,
    time_seconds: np.ndarray,
    *,
    start_seconds: float | None,
    duration_seconds: float | None,
) -> np.ndarray:
    selected_time, _ = windowed_data(
        record,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        max_points=None,
    )
    return (time_seconds >= selected_time[0]) & (time_seconds <= selected_time[-1])


def _validate_spectrum_type(spectrum_type: str) -> str:
    if spectrum_type not in {"fft", "psd"}:
        raise ValueError("spectrum_type must be 'fft' or 'psd'.")
    return spectrum_type


def _frequency_plot_title(
    record: SignalRecord, start_seconds: float | None, duration_seconds: float | None, spectrum_type: str
) -> str:
    spectrum_name = "FFT magnitude" if spectrum_type == "fft" else "PSD"
    title = f"{record.name or 'Signal'} - {spectrum_name}"
    if start_seconds is not None or duration_seconds is not None:
        start = 0.0 if start_seconds is None else start_seconds
        title = f"{title}, window from {start:g} s"
    return title


def _frequency_comparison_title(
    spectrum_name: str, start_seconds: float | None, duration_seconds: float | None
) -> str:
    title = f"{spectrum_name} comparison"
    if start_seconds is not None or duration_seconds is not None:
        start = 0.0 if start_seconds is None else start_seconds
        title = f"{title}, window from {start:g} s"
    return title


def _limited_power(
    frequencies: np.ndarray,
    power: np.ndarray,
    max_frequency_hz: float | None,
    frequency_scale: str,
) -> tuple[np.ndarray, np.ndarray]:
    if max_frequency_hz is not None:
        mask = frequencies <= max_frequency_hz
        frequencies = frequencies[mask]
        power = power[mask, :]
    return _apply_frequency_scale(frequencies, power, frequency_scale)


def _apply_frequency_scale(
    frequencies: np.ndarray, values_by_frequency: np.ndarray, frequency_scale: str
) -> tuple[np.ndarray, np.ndarray]:
    if frequency_scale not in {"linear", "log"}:
        raise ValueError("frequency_scale must be 'linear' or 'log'.")
    if frequency_scale == "linear":
        return frequencies, values_by_frequency
    positive_frequency_mask = frequencies > 0.0
    if not np.any(positive_frequency_mask):
        raise ValueError("frequency_scale='log' requires at least one positive frequency bin.")
    return frequencies[positive_frequency_mask], values_by_frequency[positive_frequency_mask, :]


def _spectrogram_color_limits(
    power_db: np.ndarray, *, vmin_db: float | None, vmax_db: float | None, dynamic_range_db: float
) -> tuple[float, float]:
    finite_power = power_db[np.isfinite(power_db)]
    data_max = 0.0 if finite_power.size == 0 else float(np.max(finite_power))
    vmax = data_max if vmax_db is None else float(vmax_db)
    vmin = vmax - dynamic_range_db if vmin_db is None else float(vmin_db)
    if vmin >= vmax:
        raise ValueError("vmin_db must be less than vmax_db.")
    return vmin, vmax


def _spectrogram_slider_bounds(
    power_db: np.ndarray, initial_vmin: float, initial_vmax: float
) -> tuple[float, float]:
    finite_power = power_db[np.isfinite(power_db)]
    if finite_power.size == 0:
        data_min, data_max = initial_vmin, initial_vmax
    else:
        data_min, data_max = float(np.min(finite_power)), float(np.max(finite_power))
    slider_min = min(data_min, initial_vmin)
    slider_max = max(data_max, initial_vmax)
    return (slider_min, slider_min + 1.0) if slider_min >= slider_max else (slider_min, slider_max)
