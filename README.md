# Signal Processing Prep

Reusable Python toolkit for exploratory analysis of vibration, acoustic, and other high-frequency time-series signals.

Current usable version: `0.1.0`.

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

## Version 0.1 Workflow

Version `0.1.0` is the first practical release of the toolkit. The intended
workflow is deliberately simple and interpretable:

1. Start with the synthetic pipeline to verify the environment.
2. Load real signal files into `SignalRecord` objects.
3. Inspect acquisition metadata and signal quality before filtering.
4. Plot raw time-domain examples across labels or conditions.
5. Analyze FFT, PSD, and spectrogram behavior.
6. Apply filtering only when there is a clear engineering reason.
7. Extract interpretable time, frequency, and time-frequency features.
8. Compare feature distributions.
9. Train simple baselines only when the dataset and labels support it.
10. Save a short Markdown summary of observations, limitations, and next steps.

The main notebook walkthrough is
`notebooks/01_signal_analysis_walkthrough.ipynb`. It starts with synthetic
records, then calls the package modules for quality checks, plots, features,
optional modeling, anomaly scoring, and Markdown reporting.

### 1. Smoke Test With Synthetic Data

Run the built-in pipeline first. This exercises quality checks, feature
extraction, optional modeling, anomaly scoring, and summary generation without
requiring private data.

```python
from signal_processing_prep import __version__, run_synthetic_analysis

print(__version__)  # 0.1.0

result = run_synthetic_analysis()

print(result.quality.head())
print(result.features.head())
print(result.markdown_summary)
```

### 2. Load Your Own Dataset

Put raw files under `data/raw/` or edit `configs/default.yaml` to point at a
different folder. Supported first-version formats are CSV, TXT, NPY, and WAV.

```python
from signal_processing_prep import load_signal_dataset

records = load_signal_dataset(
    "configs/default.yaml",
    metadata_table="data/raw/metadata.csv",  # optional
    split_channels=True,
)

print(len(records))
print(records[0])
```

If CSV files contain a time-like column such as `time`, `timestamp`, or
`time_seconds`, the loader infers the sampling rate and stores timing jitter,
gap counts, and sampling-rate mismatch indicators in `record.acquisition`.
Legacy metadata keys remain available for compatibility.

For timestamped files with noticeable jitter or gaps, treat the inferred
sampling rate as an acquisition assumption to review before FFT, PSD, or
spectrogram analysis. The toolkit intentionally keeps irregular-time support
lightweight in version `0.1.0`: it reports timing diagnostics but does not
silently resample irregular records.

### 3. Check Quality Before Processing

Quality checks are meant to reveal assumptions before they become modeling
errors.

```python
from signal_processing_prep import assess_dataset_quality

quality = assess_dataset_quality(records)

print(quality[[
    "record_name",
    "sampling_rate_hz",
    "duration_seconds",
    "missing_fraction",
    "is_clipped",
    "has_time_axis_irregularity",
    "has_sampling_rate_mismatch",
    "issues",
]])
```

Treat `issues` as a review checklist. For a first pass, skip or fix records
with non-finite samples, severe clipping, very short duration, or suspicious
time-axis metadata.

Frequency-domain and time-frequency helpers reject `NaN` and `Inf` samples
before computing FFT, PSD, STFT, spectrograms, Hilbert features, or wavelet
scalograms. For small isolated gaps, repair values explicitly and keep that
choice visible in metadata:

```python
from signal_processing_prep import interpolate_missing_values

clean_record = interpolate_missing_values(
    record,
    method="linear",
    max_missing_fraction=0.01,
)
```

Use interpolation only when the missing fraction is small and the assumption is
defensible. For long gaps, severe timestamp irregularity, or heavily repaired
signals, segment or skip the affected region rather than making spectral
claims from silently filled data.

### 4. Plot And Inspect Signals

Use plots to understand the data before extracting features or fitting models.

```python
from signal_processing_prep import (
    plot_frequency_spectrum,
    plot_spectrogram,
    plot_time_signal_navigator,
)

record = records[0]

plot_time_signal_navigator(record, window_seconds=1.0, show=True)
plot_frequency_spectrum(record)
plot_spectrogram(record, window_seconds=0.1, step_seconds=0.05)
```

### 5. Run The Analysis Pipeline

For a compact first report driven by YAML settings, run the configured workflow.
This maps configured frequency bands and filtering into the analysis pipeline.

```python
from signal_processing_prep import AnalysisPipelineConfig, analyze_dataset, load_config

project_config = load_config("configs/default.yaml")
result = analyze_dataset(
    project_config,
    metadata_table="data/raw/metadata.csv",  # optional
    pipeline_config=AnalysisPipelineConfig.from_project_config(
        project_config,
        run_modeling=True,
        run_anomaly_when_unlabeled=True,
        invalid_record_policy="skip",
    ),
)

print(result.processing_notes)
print(result.modeling_notes)
print(result.markdown_summary)
```

The pipeline keeps modeling conservative. If labels exist, supervised baselines
use grouped splits by source or record when available. If labels are missing or
supervised modeling is not defensible, the pipeline falls back to exploratory
Isolation Forest anomaly scoring when enough numeric features exist.

The configured `analysis.window` values are available for explicit segmentation
or sliding-window analysis; whole-record analysis does not segment records
silently.

### 6. Save A Markdown Summary

```python
from signal_processing_prep import save_markdown_summary

output_path = save_markdown_summary(
    result.markdown_summary,
    "reports/summaries",
    stem="v0_1_first_pass",
)

print(output_path)
```

Use the summary as a starting point for a short technical presentation: what
data was provided, what was observed, which DSP methods were applied, which
patterns were found, what the baseline could detect, and what limitations
remain.

## Data Loading Notes

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
