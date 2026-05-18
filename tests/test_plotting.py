"""Tests for time-domain plotting helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from signal_processing_prep.plotting import (
    plot_frequency_spectra,
    plot_frequency_spectrum,
    plot_time_signal,
    plot_time_signal_navigator,
)
from signal_processing_prep.synthetic import sine_wave


def test_plot_time_signal_returns_labeled_figure() -> None:
    """A time-domain plot returns a labeled matplotlib figure and axes."""
    record = sine_wave(
        frequency_hz=5.0,
        duration_seconds=1.0,
        sampling_rate_hz=100.0,
        label="normal",
        name="demo",
    )

    fig, ax = plot_time_signal(record)

    assert fig is ax.figure
    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Amplitude"
    assert "demo" in ax.get_title()
    assert "100 Hz" in ax.get_title()
    assert len(ax.lines) == 1
    plt.close(fig)


def test_plot_time_signal_applies_window_without_implicit_downsampling() -> None:
    """Windowing keeps all samples unless max_points is explicitly set."""
    record = sine_wave(duration_seconds=10.0, sampling_rate_hz=1000.0)

    fig, ax = plot_time_signal(
        record,
        start_seconds=2.0,
        duration_seconds=3.0,
    )
    x_data = ax.lines[0].get_xdata()

    assert len(x_data) == 3000
    assert x_data[0] >= 2.0
    assert x_data[-1] < 5.0
    plt.close(fig)


def test_plot_time_signal_downsamples_when_max_points_is_set() -> None:
    """Explicit downsampling keeps long signal plots bounded."""
    record = sine_wave(duration_seconds=10.0, sampling_rate_hz=1000.0)

    fig, ax = plot_time_signal(
        record,
        start_seconds=2.0,
        duration_seconds=3.0,
        max_points=100,
    )
    x_data = ax.lines[0].get_xdata()

    assert len(x_data) <= 100
    assert x_data[0] >= 2.0
    assert x_data[-1] < 5.0
    plt.close(fig)


def test_plot_time_signal_envelope_downsampling_preserves_impulse() -> None:
    """Envelope downsampling keeps sparse impulses visible in long signals."""
    record = sine_wave(duration_seconds=10.0, sampling_rate_hz=1000.0, amplitude=0.0)
    values = record.values.copy()
    values[4321] = 10.0
    impulse_record = type(record)(
        values=values,
        sampling_rate_hz=record.sampling_rate_hz,
        label=record.label,
        name=record.name,
        metadata=record.metadata,
    )

    fig, ax = plot_time_signal(impulse_record, max_points=100)
    y_data = ax.lines[0].get_ydata()

    assert np.max(y_data) == 10.0
    assert len(y_data) <= 100
    plt.close(fig)


def test_plot_time_signal_rejects_invalid_window_arguments() -> None:
    """Invalid plot window arguments fail clearly."""
    record = sine_wave()

    with pytest.raises(ValueError, match="start_seconds must be non-negative"):
        plot_time_signal(record, start_seconds=-1.0)

    with pytest.raises(ValueError, match="duration_seconds must be positive"):
        plot_time_signal(record, duration_seconds=0.0)

    with pytest.raises(ValueError, match="max_points must be positive"):
        plot_time_signal(record, max_points=0)

    with pytest.raises(ValueError, match="downsample_method"):
        plot_time_signal(record, max_points=1, downsample_method="unknown")


def test_plot_time_signal_navigator_updates_visible_window() -> None:
    """The navigator slider updates the plotted x-data window."""
    record = sine_wave(duration_seconds=5.0, sampling_rate_hz=100.0)

    navigator = plot_time_signal_navigator(
        record,
        window_seconds=1.0,
        start_seconds=0.0,
        max_points=200,
    )
    navigator.set_start_seconds(2.0)
    x_data = navigator.line.get_xdata()

    assert np.isclose(navigator.slider.val, 2.0)
    assert x_data[0] >= 2.0
    assert x_data[-1] < 3.0
    plt.close(navigator.fig)


def test_plot_time_signal_navigator_buttons_step_by_window() -> None:
    """Previous and next controls move by one window length."""
    record = sine_wave(duration_seconds=5.0, sampling_rate_hz=100.0)

    navigator = plot_time_signal_navigator(
        record,
        window_seconds=1.0,
        start_seconds=1.0,
    )
    navigator.step_next()
    assert np.isclose(navigator.slider.val, 2.0)

    navigator.step_previous()
    assert np.isclose(navigator.slider.val, 1.0)
    plt.close(navigator.fig)


def test_plot_frequency_spectrum_returns_labeled_figure() -> None:
    """A frequency-domain plot returns labeled matplotlib objects."""
    record = sine_wave(
        frequency_hz=20.0,
        duration_seconds=1.0,
        sampling_rate_hz=200.0,
        name="tone",
    )

    fig, ax = plot_frequency_spectrum(record, spectrum_type="fft", max_frequency_hz=80.0)

    assert fig is ax.figure
    assert ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_ylabel() == "Magnitude"
    assert "tone" in ax.get_title()
    assert "FFT magnitude" in ax.get_title()
    assert len(ax.lines) == 1
    assert ax.get_xlim()[1] == pytest.approx(80.0)
    plt.close(fig)


def test_plot_frequency_spectra_compares_multiple_records() -> None:
    """Multiple records can be compared in one frequency-domain figure."""
    first = sine_wave(
        frequency_hz=20.0,
        duration_seconds=1.0,
        sampling_rate_hz=200.0,
        name="20_hz",
    )
    second = sine_wave(
        frequency_hz=40.0,
        duration_seconds=1.0,
        sampling_rate_hz=200.0,
        name="40_hz",
    )

    fig, ax = plot_frequency_spectra(
        [first, second],
        spectrum_type="fft",
        max_frequency_hz=80.0,
    )

    assert fig is ax.figure
    assert ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_ylabel() == "Magnitude"
    assert "FFT magnitude comparison" in ax.get_title()
    assert len(ax.lines) == 2
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["20_hz", "40_hz"]
    plt.close(fig)


def test_plot_frequency_spectrum_rejects_invalid_arguments() -> None:
    """Invalid frequency plotting arguments fail clearly."""
    record = sine_wave()

    with pytest.raises(ValueError, match="spectrum_type"):
        plot_frequency_spectrum(record, spectrum_type="unknown")

    with pytest.raises(ValueError, match="max_frequency_hz"):
        plot_frequency_spectrum(record, max_frequency_hz=0.0)

    with pytest.raises(ValueError, match="At least one"):
        plot_frequency_spectra([])
