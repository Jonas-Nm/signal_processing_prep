"""Smoke tests for the package skeleton."""


def test_package_imports() -> None:
    """The package can be imported from the src layout."""
    import signal_processing_prep

    assert signal_processing_prep.__version__ == "0.1.0"
    assert callable(signal_processing_prep.run_robust_mahalanobis_distance)
    assert callable(signal_processing_prep.analyze_dataset)
    assert signal_processing_prep.SignalProvenance(source_name="record").source_name == "record"
