"""Primary public workflow and domain API for signal-processing analyses."""

from signal_processing_prep.config import AnalysisConfig, ProjectConfig, load_config
from signal_processing_prep.data_loading import SignalDatasetLoader
from signal_processing_prep.pipeline import AnalysisResult, SignalAnalysisPipeline, run_synthetic_analysis
from signal_processing_prep.records import SignalDataset, SignalRecord

__version__ = "0.2.0"

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "ProjectConfig",
    "SignalAnalysisPipeline",
    "SignalDataset",
    "SignalDatasetLoader",
    "SignalRecord",
    "__version__",
    "load_config",
    "run_synthetic_analysis",
]
