"""Signal-processing analysis toolkit for exploratory time-series tasks."""

from signal_processing_prep.config import ProjectConfig, load_config
from signal_processing_prep.data_loading import load_signal_file
from signal_processing_prep.records import SignalRecord

__version__ = "0.1.0"

__all__ = [
    "ProjectConfig",
    "SignalRecord",
    "__version__",
    "load_config",
    "load_signal_file",
]
