"""Signal-processing analysis toolkit for exploratory time-series tasks."""

from signal_processing_prep.config import ProjectConfig, load_config
from signal_processing_prep.data_loading import load_signal_file
from signal_processing_prep.features import (
    FeatureExtractionConfig,
    FrequencyBand,
    extract_features,
    frequency_bands_from_mapping,
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
from signal_processing_prep.records import SignalRecord

__version__ = "0.1.0"

__all__ = [
    "FeatureExtractionConfig",
    "FilterSpec",
    "FrequencyBand",
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
    "load_config",
    "load_signal_file",
    "segment_dataset",
    "segment_signal",
]
