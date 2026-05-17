"""Smoke tests for the package skeleton."""


def test_package_imports() -> None:
    """The package can be imported from the src layout."""
    import signal_processing_prep

    assert signal_processing_prep.__version__ == "0.1.0"
