# Signal Processing Prep

Reusable Python toolkit for exploratory analysis of vibration, acoustic, and other high-frequency time-series signals.

Current version: `0.2.0`.

The package is designed for interpretable signal analysis: inspect acquisition assumptions and signal quality first, apply DSP methods explicitly, and use simple models only where their limitations are clear. See `docs/architecture.md` for boundary rules and `docs/roadmap.md` for implementation scope.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
pytest
```

Install `.[deep-learning]` only when running the optional spectrogram autoencoder experiment.

## Primary Workflow

`SignalAnalysisPipeline` is the user-facing workflow. It coordinates a typed `SignalDataset`, quality assessment, explicit preprocessing, feature extraction, optional modeling, and report construction.

```python
from signal_processing_prep import SignalAnalysisPipeline

pipeline = SignalAnalysisPipeline.from_config("configs/default.yaml")
result = pipeline.run_dataset(metadata_table="data/raw/metadata.csv")

print(result.quality_frame.head())
print(result.feature_frame.head())
print(result.markdown_summary)
result.report.save("reports/summaries", stem="first_pass")
```

For an environment smoke test without private files:

```python
from signal_processing_prep import run_synthetic_analysis

result = run_synthetic_analysis()
print(result.feature_frame.head())
```

## Typed Boundaries

- `SignalRecord` owns immutable values, source `provenance`, acquisition diagnostics, typed annotations, optional `segment_span`, processing history, and user `attributes`.
- `SignalDatasetLoader` loads CSV, TXT, NPY, and WAV data into a `SignalDataset`; joined label, split, sensor, and anomaly interval facts become typed annotations.
- `FeatureExtractor` and `QualityAssessor` return validated `FeatureTable` and `QualityTable` artifacts. Use `to_dataframe()` for notebook inspection.
- Modeling is strategy-based: use scorer objects such as `IsolationForestScorer` or `RobustZScoreScorer`, and `SupervisedBaselineSuite` for labeled baselines.
- `AnalysisReport` owns Markdown rendering and saving.

Dictionary compatibility metadata and package-root convenience exports from the earlier workflow are intentionally removed. New analysis code should not read processing state, segment positions, or annotations from free-form mappings.

## Focused Exploration

Notebook workflows can use specialist components without invoking the complete pipeline:

```python
from signal_processing_prep import SignalDatasetLoader, load_config
from signal_processing_prep.features import FeatureExtractor, SlidingWindowConfig, frequency_bands_from_mapping
from signal_processing_prep.modeling import RobustZScoreScorer, top_anomalies

config = load_config("configs/default.yaml")
record = SignalDatasetLoader().load_file("data/raw/example.csv", signal_column="voltage")

feature_table = FeatureExtractor().extract_windows(
    record,
    SlidingWindowConfig(
        window_seconds=0.25,
        step_seconds=0.05,
        frequency_bands=frequency_bands_from_mapping(dict(config.analysis.frequency_bands_hz)),
    ),
)
evaluation = RobustZScoreScorer().score(feature_table)
print(top_anomalies(evaluation))
```

The calculation-focused modules `time_domain`, `frequency_domain`, and `time_frequency` retain function APIs for direct DSP study and testing.

## Workflow Guidance

1. Load records and inspect typed acquisition diagnostics.
2. Review quality findings before cleaning, filtering, or modeling.
3. Plot time-domain and frequency-domain behavior.
4. Extract interpretable whole-record or window features.
5. Use grouped evaluation for labeled records; treat anomaly scores as candidates for inspection.
6. Record assumptions and limitations in the resulting report.

Multi-channel inputs are represented as separate records with explicit channel provenance. The library reports inferred timing irregularities but does not silently resample or truncate data.

## Notebooks

Recommended interview-preparation sequence:

1. `notebooks/02_dsp_foundations_examples.ipynb`: controlled DSP concept laboratory with expected results, aliasing, windowing, localization, and envelope analysis.
2. `notebooks/01_signal_analysis_walkthrough_v2.ipynb`: blinded single-record investigation that reports candidate regions before revealing synthetic validation timing.
3. `notebooks/03_feature_based_anomaly_detection_walkthrough.ipynb`: fair shared-feature comparison of interpretable anomaly scorers and region-level agreement.
4. `notebooks/04_synthetic_spectrogram_autoencoder_walkthrough.ipynb`: optional controlled neural reconstruction-error experiment, separate from the unlabeled single-record evidence.

`notebooks/01_signal_analysis_walkthrough.ipynb` remains the general synthetic pipeline first pass; the `v2` notebook is the focused investigation walkthrough.
