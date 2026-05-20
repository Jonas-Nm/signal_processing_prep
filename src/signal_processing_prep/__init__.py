"""Signal-processing analysis toolkit for exploratory time-series tasks."""

from signal_processing_prep.config import ProjectConfig, load_config
from signal_processing_prep.data_loading import (
    load_signal_dataset,
    load_signal_file,
    load_signal_file_channels,
)
from signal_processing_prep.features import (
    FeatureExtractionConfig,
    FrequencyBand,
    SlidingWindowConfig,
    extract_features,
    frequency_bands_from_mapping,
    sliding_window_features,
)
from signal_processing_prep.modeling import (
    ModelEvaluation,
    anomaly_summary_text,
    run_isolation_forest,
    run_supervised_baselines,
    top_anomalies,
)
from signal_processing_prep.plotting import (
    plot_anomaly_scores,
    plot_confusion_matrix,
    plot_feature_distribution,
    plot_feature_importance,
    plot_frequency_spectra,
    plot_frequency_spectrum,
    plot_spectrogram,
    plot_time_signal,
    plot_time_signal_navigator,
    save_figure,
)
from signal_processing_prep.pipeline import (
    AnalysisPipelineConfig,
    AnalysisPipelineResult,
    analyze_records,
    run_synthetic_analysis,
)
from signal_processing_prep.preprocessing import (
    FilterSpec,
    apply_configured_filter,
    apply_filter,
    apply_window,
    interpolate_missing_values,
    segment_dataset,
    segment_signal,
)
from signal_processing_prep.quality import (
    QualityCheckConfig,
    QualityReport,
    assess_dataset_quality,
    assess_signal_quality,
)
from signal_processing_prep.reporting import (
    MarkdownReport,
    dataset_overview_from_records,
    generate_markdown_summary,
    save_markdown_summary,
)
from signal_processing_prep.records import SignalRecord

__version__ = "0.1.0"

__all__ = [
    "FeatureExtractionConfig",
    "FilterSpec",
    "FrequencyBand",
    "SlidingWindowConfig",
    "AnalysisPipelineConfig",
    "AnalysisPipelineResult",
    "MarkdownReport",
    "ModelEvaluation",
    "ProjectConfig",
    "QualityCheckConfig",
    "QualityReport",
    "SignalRecord",
    "__version__",
    "apply_configured_filter",
    "apply_filter",
    "apply_window",
    "analyze_records",
    "anomaly_summary_text",
    "assess_dataset_quality",
    "assess_signal_quality",
    "extract_features",
    "sliding_window_features",
    "frequency_bands_from_mapping",
    "dataset_overview_from_records",
    "generate_markdown_summary",
    "interpolate_missing_values",
    "load_signal_dataset",
    "load_config",
    "load_signal_file",
    "load_signal_file_channels",
    "plot_confusion_matrix",
    "plot_anomaly_scores",
    "plot_feature_distribution",
    "plot_feature_importance",
    "plot_frequency_spectra",
    "plot_frequency_spectrum",
    "plot_spectrogram",
    "plot_time_signal",
    "plot_time_signal_navigator",
    "run_isolation_forest",
    "run_supervised_baselines",
    "run_synthetic_analysis",
    "save_figure",
    "save_markdown_summary",
    "segment_dataset",
    "segment_signal",
    "top_anomalies",
]
