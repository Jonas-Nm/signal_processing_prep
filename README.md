# Signal Processing Prep

Reusable Python toolkit for exploratory analysis of vibration, acoustic, and other high-frequency time-series signals.

The project is intentionally built in small phases. See `docs/roadmap.md` for the implementation roadmap and acceptance criteria.

## Setup

Create a local virtual environment and install the package in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For notebook-based exploration with interactive matplotlib widgets, install the notebook extra as well:

```powershell
python -m pip install -e ".[dev,notebook]"
```

Run the test suite:

```powershell
pytest
```

The `.venv/` directory is local developer state and is ignored by Git.

## Walkthrough

The main interview-facing walkthrough is `notebooks/01_signal_analysis_walkthrough.ipynb`.
It starts with synthetic records, then calls the package modules for quality checks, plots,
features, optional modeling, anomaly scoring, and Markdown reporting.

## Loading Real Datasets

Use `load_signal_dataset` with a config file to discover all matching files under
`paths.data_dir`. CSV time columns such as `time`, `timestamp`, or `time_seconds`
are used to infer sampling rate and preserve acquisition checks such as timing
jitter and gaps in record metadata.

```python
from signal_processing_prep import load_signal_dataset

records = load_signal_dataset(
    "configs/default.yaml",
    metadata_table="data/raw/metadata.csv",  # optional
    split_channels=True,
)
```

Multi-channel WAV, NPY, TXT, and multi-signal CSV files are split into explicit
single-channel `SignalRecord`s. Supervised baselines group train/test splits by
`source_name` or `record_name` when available, which keeps windows from the same
recording on one side of the evaluation split.

## Quick Synthetic Plot

After setup, generate and inspect a synthetic signal:

```python
from signal_processing_prep.plotting import plot_time_signal_navigator
from signal_processing_prep.synthetic import transient_burst

record = transient_burst(duration_seconds=5.0)
navigator = plot_time_signal_navigator(record, window_seconds=1.0, show=True)
```

Use the matplotlib toolbar for zoom and pan. The slider and previous/next buttons move through longer records by time window.

By default, plotting keeps all samples in the selected window. For very long windows, set `max_points` to explicitly enable envelope downsampling:

```python
navigator = plot_time_signal_navigator(record, window_seconds=10.0, max_points=5000, show=True)
```
