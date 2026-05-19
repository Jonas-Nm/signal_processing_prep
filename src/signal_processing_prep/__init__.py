"""Signal-processing analysis toolkit for exploratory time-series tasks."""

from signal_processing_prep.config import ProjectConfig, load_config
from signal_processing_prep.data_loading import load_signal_file
from signal_processing_prep.features import (
    FeatureExtractionConfig,
    FrequencyBand,
    extract_features,
    frequency_bands_from_mapping,
)
from signal_processing_prep.records import SignalRecord

__version__ = "0.1.0"

__all__ = [
    "FeatureExtractionConfig",
    "FrequencyBand",
    "ProjectConfig",
    "SignalRecord",
    "__version__",
    "extract_features",
    "frequency_bands_from_mapping",
    "load_config",
    "load_signal_file",
]
