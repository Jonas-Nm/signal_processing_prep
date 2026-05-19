"""Signal-processing analysis toolkit for exploratory time-series tasks."""

from signal_processing_prep.config import ProjectConfig, load_config
from signal_processing_prep.data_loading import load_signal_file
from signal_processing_prep.features import (
    FeatureExtractionConfig,
    FrequencyBand,
    extract_features,
    frequency_bands_from_mapping,
)
from signal_processing_prep.modeling import (
    ModelEvaluation,
    run_isolation_forest,
    run_supervised_baselines,
)
from signal_processing_prep.plotting import (
    plot_confusion_matrix,
    plot_feature_distribution,
    plot_feature_importance,
    plot_spectrogram,
    save_figure,
)
from signal_processing_prep.preprocessing import (
    FilterSpec,
    apply_configured_filter,
    apply_filter,
    apply_window,
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
    "assess_dataset_quality",
    "assess_signal_quality",
    "extract_features",
    "frequency_bands_from_mapping",
    "dataset_overview_from_records",
    "generate_markdown_summary",
    "load_config",
    "load_signal_file",
    "plot_confusion_matrix",
    "plot_feature_distribution",
    "plot_feature_importance",
    "plot_spectrogram",
    "run_isolation_forest",
    "run_supervised_baselines",
    "save_figure",
    "save_markdown_summary",
    "segment_dataset",
    "segment_signal",
]
