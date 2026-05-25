"""Smoke tests for the primary package surface."""


def test_package_exports_only_primary_workflow_concepts() -> None:
    """The package root remains a compact navigation starting point."""
    import signal_processing_prep as package

    assert package.__version__ == "0.2.0"
    assert set(package.__all__) == {
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
    }
