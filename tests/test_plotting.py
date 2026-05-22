"""Tests for time-domain plotting helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from signal_processing_prep.plotting import (
    plot_anomaly_scores,
    plot_confusion_matrix,
    plot_feature_distribution,
    plot_feature_importance,
    plot_frequency_spectra,
    plot_frequency_spectrum,
    plot_spectrogram,
    plot_spectrogram_dynamic_range,
    plot_time_signal,
    plot_time_signal_adaptive,
    plot_time_signal_navigator,
    plot_wavelet_scalogram,
    save_figure,
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


def test_plot_time_signal_adaptive_updates_data_when_zooming() -> None:
    """Adaptive time plots recompute displayed samples from the visible x-range."""
    record = sine_wave(duration_seconds=10.0, sampling_rate_hz=1000.0)

    fig, ax = plot_time_signal_adaptive(record, max_points=100)
    initial_x_data = ax.lines[0].get_xdata()

    ax.set_xlim(2.0, 2.05)
    zoom_x_data = ax.lines[0].get_xdata()

    assert len(initial_x_data) <= 100
    assert len(zoom_x_data) == 50
    assert zoom_x_data[0] >= 2.0
    assert zoom_x_data[-1] < 2.05
    plt.close(fig)


def test_plot_time_signal_adaptive_rejects_invalid_arguments() -> None:
    """Adaptive time plots validate their display and window arguments."""
    record = sine_wave()

    with pytest.raises(ValueError, match="max_points must be positive"):
        plot_time_signal_adaptive(record, max_points=0)

    with pytest.raises(ValueError, match="start_seconds must be non-negative"):
        plot_time_signal_adaptive(record, start_seconds=-1.0)

    with pytest.raises(ValueError, match="duration_seconds must be positive"):
        plot_time_signal_adaptive(record, duration_seconds=0.0)

    with pytest.raises(ValueError, match="downsample_method"):
        plot_time_signal_adaptive(record, max_points=1, downsample_method="unknown")


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


def test_plot_spectrogram_returns_labeled_figure() -> None:
    """A spectrogram plot returns labeled axes and a colorbar."""
    record = sine_wave(frequency_hz=20.0, duration_seconds=1.0, sampling_rate_hz=200.0)

    fig, ax = plot_spectrogram(record, window_seconds=0.2, step_seconds=0.1)

    assert fig is ax.figure
    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Frequency [Hz]"
    assert ax.get_yscale() == "linear"
    assert "spectrogram" in ax.get_title()
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_spectrogram_supports_log_frequency_scale() -> None:
    """Log-frequency spectrograms drop the zero-Hz bin and use a log y-axis."""
    record = sine_wave(frequency_hz=20.0, duration_seconds=1.0, sampling_rate_hz=200.0)

    fig, ax = plot_spectrogram(
        record,
        window_seconds=0.2,
        step_seconds=0.1,
        frequency_scale="log",
    )
    y_coordinates = ax.collections[0].get_coordinates()[:, :, 1]

    assert ax.get_yscale() == "log"
    assert np.nanmin(y_coordinates) > 0.0
    plt.close(fig)


def test_plot_spectrogram_dynamic_range_adds_sliders() -> None:
    """Dynamic spectrogram plots expose colorbar and dB range sliders."""
    record = sine_wave(frequency_hz=20.0, duration_seconds=1.0, sampling_rate_hz=200.0)

    fig, ax = plot_spectrogram_dynamic_range(record, window_seconds=0.2, step_seconds=0.1)

    assert fig is ax.figure
    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Frequency [Hz]"
    assert len(fig.axes) == 4
    assert hasattr(ax, "_signal_processing_prep_spectrogram_dynamic_range")
    plt.close(fig)


def test_plot_spectrogram_dynamic_range_supports_log_frequency_scale() -> None:
    """Dynamic spectrograms support the same log-frequency view."""
    record = sine_wave(frequency_hz=20.0, duration_seconds=1.0, sampling_rate_hz=200.0)

    fig, ax = plot_spectrogram_dynamic_range(
        record,
        window_seconds=0.2,
        step_seconds=0.1,
        frequency_scale="log",
    )
    y_coordinates = ax.collections[0].get_coordinates()[:, :, 1]

    assert ax.get_yscale() == "log"
    assert np.nanmin(y_coordinates) > 0.0
    plt.close(fig)


def test_plot_spectrogram_dynamic_range_sliders_update_color_limits() -> None:
    """Changing the dB sliders updates the spectrogram mesh color limits."""
    record = sine_wave(frequency_hz=20.0, duration_seconds=1.0, sampling_rate_hz=200.0)

    fig, ax = plot_spectrogram_dynamic_range(
        record,
        window_seconds=0.2,
        step_seconds=0.1,
        dynamic_range_db=40.0,
    )
    controller = ax._signal_processing_prep_spectrogram_dynamic_range
    mesh = ax.collections[0]
    original_vmin, original_vmax = mesh.get_clim()

    controller.min_slider.set_val(original_vmin + 5.0)
    controller.max_slider.set_val(original_vmax - 5.0)

    assert mesh.get_clim() == pytest.approx((original_vmin + 5.0, original_vmax - 5.0))
    plt.close(fig)


def test_plot_spectrogram_dynamic_range_rejects_invalid_arguments() -> None:
    """Invalid dynamic spectrogram arguments fail clearly."""
    record = sine_wave()

    with pytest.raises(ValueError, match="max_frequency_hz"):
        plot_spectrogram_dynamic_range(record, max_frequency_hz=0.0)

    with pytest.raises(ValueError, match="dynamic_range_db"):
        plot_spectrogram_dynamic_range(record, dynamic_range_db=0.0)

    with pytest.raises(ValueError, match="vmin_db"):
        plot_spectrogram_dynamic_range(record, vmin_db=-20.0, vmax_db=-20.0)

    with pytest.raises(ValueError, match="frequency_scale"):
        plot_spectrogram(record, frequency_scale="symlog")

    with pytest.raises(ValueError, match="frequency_scale"):
        plot_spectrogram_dynamic_range(record, frequency_scale="symlog")


def test_plot_wavelet_scalogram_returns_labeled_log_figure() -> None:
    """Wavelet scalograms return labeled axes, a colorbar, and dB range sliders."""
    record = sine_wave(frequency_hz=20.0, duration_seconds=1.0, sampling_rate_hz=200.0)

    fig, ax = plot_wavelet_scalogram(
        record,
        min_frequency_hz=5.0,
        max_frequency_hz=80.0,
        n_frequencies=16,
    )

    assert fig is ax.figure
    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Frequency [Hz]"
    assert ax.get_yscale() == "log"
    assert "wavelet scalogram" in ax.get_title()
    assert len(fig.axes) == 4
    assert hasattr(ax, "_signal_processing_prep_wavelet_dynamic_range")
    plt.close(fig)


def test_plot_wavelet_scalogram_sliders_update_color_limits() -> None:
    """Changing the wavelet dB sliders updates the scalogram mesh color limits."""
    record = sine_wave(frequency_hz=20.0, duration_seconds=1.0, sampling_rate_hz=200.0)

    fig, ax = plot_wavelet_scalogram(
        record,
        min_frequency_hz=5.0,
        max_frequency_hz=80.0,
        n_frequencies=16,
        dynamic_range_db=40.0,
    )
    controller = ax._signal_processing_prep_wavelet_dynamic_range
    mesh = ax.collections[0]
    original_vmin, original_vmax = mesh.get_clim()

    controller.min_slider.set_val(original_vmin + 5.0)
    controller.max_slider.set_val(original_vmax - 5.0)

    assert mesh.get_clim() == pytest.approx((original_vmin + 5.0, original_vmax - 5.0))
    plt.close(fig)


def test_plot_wavelet_scalogram_rejects_invalid_arguments() -> None:
    """Invalid wavelet plotting arguments fail clearly."""
    record = sine_wave(frequency_hz=20.0, duration_seconds=1.0, sampling_rate_hz=200.0)

    with pytest.raises(ValueError, match="max_frequency_hz"):
        plot_wavelet_scalogram(record, min_frequency_hz=80.0, max_frequency_hz=20.0)

    with pytest.raises(ValueError, match="n_frequencies"):
        plot_wavelet_scalogram(record, n_frequencies=0)

    with pytest.raises(ValueError, match="frequency_scale"):
        plot_wavelet_scalogram(record, frequency_scale="symlog")

    with pytest.raises(ValueError, match="dynamic_range_db"):
        plot_wavelet_scalogram(record, dynamic_range_db=0.0)

    with pytest.raises(ValueError, match="vmin_db"):
        plot_wavelet_scalogram(record, vmin_db=-20.0, vmax_db=-20.0)


def test_plot_feature_distribution_groups_by_label() -> None:
    """Feature distributions can compare labels."""
    import pandas as pd

    features = pd.DataFrame(
        {
            "rms": [1.0, 1.2, 3.0, 3.2],
            "label": ["normal", "normal", "fault", "fault"],
        }
    )

    fig, ax = plot_feature_distribution(features, "rms")

    assert ax.get_xlabel() == "rms"
    assert ax.get_ylabel() == "Count"
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_confusion_matrix_and_feature_importance() -> None:
    """Model diagnostic plots expose expected labels."""
    import pandas as pd

    matrix_fig, matrix_ax = plot_confusion_matrix(np.array([[2, 1], [0, 3]]), ["a", "b"])
    assert matrix_ax.get_xlabel() == "Predicted label"
    assert len(matrix_ax.texts) == 4

    importance_fig, importance_ax = plot_feature_importance(pd.Series({"rms": 0.8, "zcr": 0.2}))
    assert importance_ax.get_xlabel() == "Importance"
    assert len(importance_ax.patches) == 2
    plt.close(matrix_fig)
    plt.close(importance_fig)


def test_plot_anomaly_scores_ranks_scores() -> None:
    """Anomaly-score plots show ranked inspection candidates."""
    import pandas as pd

    predictions = pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "record_name": ["a", "b", "c"],
            "anomaly_score": [0.1, 2.0, 0.5],
        }
    )

    fig, ax = plot_anomaly_scores(predictions, top_n=2)

    assert ax.get_xlabel() == "Anomaly score"
    assert ax.get_ylabel() == "Record or row"
    assert len(ax.patches) == 2
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["b", "c"]
    plt.close(fig)


def test_plot_anomaly_scores_falls_back_to_dataframe_index() -> None:
    """Generic score tables do not need row_index metadata."""
    import pandas as pd

    predictions = pd.DataFrame(
        {"anomaly_score": [0.1, 2.0, 0.5]},
        index=["first", "second", "third"],
    )

    fig, ax = plot_anomaly_scores(predictions, top_n=2)

    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["second", "third"]
    plt.close(fig)


def test_save_figure_writes_date_stamped_png(tmp_path) -> None:
    """Figures are saved under reports/figures/date-style directories."""
    record = sine_wave()
    fig, _ = plot_time_signal(record)

    path = save_figure(fig, tmp_path, stem="Raw Signal", run_date="2026-05-19")

    assert path.exists()
    assert path.parent == tmp_path / "2026-05-19"
    assert path.name == "Raw_Signal.png"
    plt.close(fig)
