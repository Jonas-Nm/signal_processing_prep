"""Public facade for plotting signal-analysis artifacts."""

from signal_processing_prep._plotting_results import (
    plot_anomaly_scores,
    plot_confusion_matrix,
    plot_feature_distribution,
    plot_feature_importance,
    save_figure,
)
from signal_processing_prep._plotting_spectral import (
    plot_frequency_spectra,
    plot_frequency_spectrum,
    plot_hilbert_huang_imfs,
    plot_hilbert_huang_spectrum,
    plot_spectrogram,
    plot_spectrogram_dynamic_range,
    plot_teager_kaiser_energy,
    plot_teager_kaiser_frequency,
    plot_wavelet_scalogram,
)
from signal_processing_prep._plotting_time import (
    TimeSignalNavigator,
    plot_time_signal,
    plot_time_signal_adaptive,
    plot_time_signal_navigator,
)

__all__ = [
    "TimeSignalNavigator",
    "plot_anomaly_scores",
    "plot_confusion_matrix",
    "plot_feature_distribution",
    "plot_feature_importance",
    "plot_frequency_spectra",
    "plot_frequency_spectrum",
    "plot_hilbert_huang_imfs",
    "plot_hilbert_huang_spectrum",
    "plot_spectrogram",
    "plot_spectrogram_dynamic_range",
    "plot_teager_kaiser_energy",
    "plot_teager_kaiser_frequency",
    "plot_time_signal",
    "plot_time_signal_adaptive",
    "plot_time_signal_navigator",
    "plot_wavelet_scalogram",
    "save_figure",
]
