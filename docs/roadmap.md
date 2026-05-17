# Signal Processing Pipeline Roadmap

## Summary

This repository will grow in small, reviewable slices. The target is a lightweight, reusable Python toolkit for exploratory time-series signal analysis, with tested source modules and a notebook walkthrough for interview-style presentation.

The project should remain modular and easy to explain. Core analysis logic belongs in importable Python modules, while notebooks and reports should orchestrate or present results without owning important implementation details.

## Target Project Structure

```text
signal_processing_prep/
|-- AGENTS.md
|-- README.md
|-- pyproject.toml
|-- configs/
|   |-- default.yaml
|   `-- synthetic.yaml
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- docs/
|   `-- roadmap.md
|-- notebooks/
|   `-- 01_signal_analysis_walkthrough.ipynb
|-- reports/
|   |-- figures/
|   `-- summaries/
|-- src/
|   `-- signal_processing_prep/
|       |-- __init__.py
|       |-- config.py
|       |-- data_loading.py
|       |-- records.py
|       |-- synthetic.py
|       |-- quality.py
|       |-- preprocessing.py
|       |-- time_domain.py
|       |-- frequency_domain.py
|       |-- time_frequency.py
|       |-- features.py
|       |-- modeling.py
|       |-- plotting.py
|       |-- reporting.py
|       `-- pipeline.py
`-- tests/
    |-- test_records.py
    |-- test_data_loading.py
    |-- test_synthetic.py
    |-- test_frequency_domain.py
    |-- test_features.py
    `-- test_pipeline.py
```

## Module Responsibilities

- `config.py`: load and validate YAML configuration into typed structures.
- `data_loading.py`: load CSV, TXT, NPY, and WAV files into a shared signal record model.
- `records.py`: define shared dataclasses such as `SignalRecord`.
- `synthetic.py`: generate example signals for demos and tests.
- `quality.py`: check duration, missing values, clipping, scaling, and stationarity indicators.
- `preprocessing.py`: provide optional filtering and windowing utilities.
- `time_domain.py`: compute time-domain metrics.
- `frequency_domain.py`: compute FFT, PSD, dominant frequency, spectral features, and band energies.
- `time_frequency.py`: compute STFT, spectrogram, transient energy, and related features.
- `features.py`: combine interpretable features into a pandas DataFrame.
- `modeling.py`: provide simple supervised and unsupervised baselines.
- `plotting.py`: create presentation-ready matplotlib figures.
- `reporting.py`: generate concise Markdown summaries.
- `pipeline.py`: orchestrate the preferred workflow while delegating implementation to focused modules.

## Phase 0: Planning Docs And AGENTS Cleanup

- Create `docs/roadmap.md`.
- Store the full project structure outline, phased implementation plan, module responsibilities, and acceptance criteria here.
- Rework `AGENTS.md` into a concise repo guidance file:
  - Keep project purpose.
  - Keep core design philosophy.
  - Keep coding standards.
  - Keep important constraints.
  - Keep the high-level preferred workflow.
  - Remove or shorten detailed feature, plotting, modeling, reporting, synthetic data, and testing lists.
  - Add a pointer to `docs/roadmap.md` for detailed implementation scope.
- Do not add executable package code yet.

Acceptance criteria:

- Future sessions can recover the project direction from `docs/roadmap.md`.
- `AGENTS.md` is shorter and stable as an agent instruction file.
- No implementation logic is introduced.

## Phase 1: Project Skeleton

- Add `pyproject.toml` with Python 3.11+, package metadata, dependencies, and pytest config.
- Create the `src/signal_processing_prep/` package with `__init__.py`.
- Create minimal folders: `configs/`, `tests/`, `notebooks/`, `reports/figures/`, `reports/summaries/`.
- Add a small `configs/default.yaml`.
- Add a smoke test proving the package imports correctly.

Acceptance criteria:

- `pytest` runs successfully.
- The package can be imported.
- No real DSP logic yet.

## Phase 2: Core Data Model And Synthetic Signals

- Add a `SignalRecord` dataclass for signal values, sampling rate, label, name, and metadata.
- Add synthetic generators for sine, noisy sine, impulse train, chirp, clipped signal, and transient burst.
- Add tests for signal shape, sampling rate, labels, and basic generated properties.

Acceptance criteria:

- Synthetic data can drive future tests.
- No external dataset is needed.

## Phase 3: Basic Loading And Config

- Implement YAML config loading.
- Implement simple loaders for CSV, TXT, NPY, and WAV.
- Normalize loaded data into `SignalRecord`.
- Add tests using temporary files.

Acceptance criteria:

- Small local test files load correctly.
- Missing labels and variable lengths are handled explicitly.

## Phase 4: DSP Foundations

- Implement time-domain helpers: RMS, crest factor, zero-crossing rate, skewness, kurtosis.
- Implement frequency-domain helpers: FFT magnitude, PSD, dominant frequency, spectral centroid, spectral bandwidth, spectral rolloff, spectral flatness, and band energy.
- Add focused tests using known synthetic signals.

Acceptance criteria:

- Known sine wave peak is detected correctly.
- RMS, crest factor, and band energy tests pass.

## Phase 5: Feature Extraction

- Combine time-domain, frequency-domain, and straightforward time-frequency features into a pandas DataFrame.
- Add configurable frequency bands.
- Include features such as mean, standard deviation, RMS, min, max, peak-to-peak amplitude, crest factor, skewness, kurtosis, zero-crossing rate, dominant frequency, spectral features, band energies, mean spectrogram energy, max spectrogram energy, high-frequency transient energy, and spectral entropy where practical.
- Add feature DataFrame shape and column-name tests.

Acceptance criteria:

- One row per signal record.
- Expected feature columns are present.
- Works for small datasets.

## Phase 6: Quality Checks And Preprocessing

- Add quality checks for duration, missing values, clipping, scaling, and simple stationarity indicators.
- Add optional low-pass, high-pass, and band-pass filtering.
- Add windowing utilities for trying different window sizes.

Acceptance criteria:

- Quality output is explicit and interpretable.
- Filtering is optional and parameter-driven.

## Phase 7: Plotting And Reporting

- Add plotting functions for raw signal, zoomed signal, FFT magnitude, PSD, spectrogram, feature distributions, confusion matrix, and feature importance where applicable.
- Ensure plots include titles, axis labels, units where known, and legends when comparing conditions.
- Save figures to `reports/figures/[date]/`.
- Add Markdown summary generation with these sections:
  1. Dataset overview
  2. Acquisition assumptions
  3. Signal quality observations
  4. Time-domain findings
  5. Frequency-domain findings
  6. Time-frequency findings
  7. Feature/model findings
  8. Limitations
  9. Recommended next steps
- Save summaries to `reports/summaries/`.

Acceptance criteria:

- Figures are presentation-ready.
- Reports are factual and avoid overclaiming.

## Phase 8: Simple Baseline Models

- Add supervised baselines for logistic regression and random forest when labels exist.
- Add unsupervised anomaly scoring with Isolation Forest when labels are absent.
- Consider PCA-based reconstruction error as an optional unsupervised baseline.
- Report accuracy, precision, recall, F1-score, and confusion matrix for supervised runs.
- Include feature importance for random forest where appropriate.

Acceptance criteria:

- Modeling is clearly optional.
- Dataset size and split strategy are reported.
- Anomaly scores are presented as scores, not overclaimed classifications.

## Phase 9: Notebook Walkthrough

- Create `notebooks/01_signal_analysis_walkthrough.ipynb`.
- Use synthetic data first.
- Walk through loading, metadata inspection, quality checks, time-domain plots, FFT/PSD, spectrograms, optional filtering, feature extraction, optional modeling, and Markdown reporting.
- Keep notebook logic thin by calling package functions.

Acceptance criteria:

- The notebook is the main interview-facing artifact.
- Core behavior remains tested in source modules.

## Synthetic Signal Examples

Use synthetic signals for tests and demos so the pipeline does not depend on private datasets:

- Pure sine wave.
- Sine wave plus noise.
- Impulse train.
- Chirp signal.
- Signal with clipping.
- Signal with high-frequency transient burst.

## Testing Expectations

Add unit tests for:

- Loading simple synthetic signals.
- FFT peak detection on a known sine wave.
- RMS calculation.
- Crest factor calculation.
- Band energy calculation.
- Feature DataFrame shape and column names.

Synthetic signals should be preferred in tests wherever possible.

## Assumptions

- Each phase should be implemented and verified before starting the next.
- The first implementation task is Phase 0 only.
- `AGENTS.md` should stay concise and durable.
- The project should stay lightweight, reusable, and easy to explain.
