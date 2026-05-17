# AGENTS.md

## Project Purpose

This repository contains a reusable Python pipeline for analyzing vibration, acoustic, or other high-frequency time-series signals in the context of condition monitoring and fault detection.
The goal is not to build a large production framework, but to provide a clean, modular, and interview-ready analysis toolkit for unknown signal datasets.

The pipeline should support exploratory signal analysis, interpretable DSP features, simple baseline models, and clear reporting suitable for a take-home technical task or interview presentation.

For the detailed phased implementation roadmap, see `docs/roadmap.md`.

## Core Design Philosophy

1. Start simple and interpretable.
2. Prioritize signal understanding before machine learning.
3. Keep physical interpretation in mind.
4. Prefer clear plots and robust baselines over unnecessarily complex models.
5. Make every processing step explicit and easy to explain.
6. Keep the code readable, modular, and suitable for fast adaptation to unknown datasets.

## Preferred Analysis Workflow

The analysis should generally follow this order:

1. Load the dataset.
2. Inspect metadata and acquisition assumptions.
3. Check signal quality.
4. Visualize raw time-domain signals.
5. Compare examples across labels or conditions if available.
6. Analyze frequency content using FFT and PSD.
7. Analyze nonstationary behavior using STFT or spectrograms.
8. Apply filtering only when there is a clear reason.
9. Extract interpretable features.
10. Compare feature distributions.
11. Train simple baseline models if appropriate.
12. Evaluate results carefully.
13. Document assumptions, limitations, and next steps.

## Coding Standards

- Use Python 3.11 or newer.
- Use type hints for public functions.
- Use docstrings for all public functions.
- Prefer small functions with one clear responsibility.
- Avoid hidden global state.
- Use `pathlib` instead of raw string paths where possible.
- Use dataclasses for structured signal records or configuration objects.
- Keep plotting functions separate from computation functions.
- Do not hard-code dataset-specific assumptions unless clearly marked.
- Use configuration files for sampling rate, file patterns, frequency bands, and output paths.
- Make code robust to small datasets and missing labels.

## Technical Scope

The project should eventually support:

- Loading signal data from common formats such as CSV, TXT, NPY, and WAV.
- Synthetic example data so the full workflow can run without private datasets.
- Metadata such as sampling rate, labels, file names, and sensor information.
- Signal quality checks, time-domain plots, FFT, PSD, spectrograms, optional filtering, feature extraction, baseline models, and Markdown reporting.

Detailed feature lists, test expectations, plotting requirements, and phased acceptance criteria are maintained in `docs/roadmap.md`.

## Recommended Libraries

Use these libraries unless there is a clear reason not to:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `soundfile` or `scipy.io.wavfile`
- `pyyaml`
- `pytest`

Optional libraries should be added only when justified by the task, for example `pywavelets` for wavelet analysis or `librosa` for audio-specific features.

Avoid unnecessary dependencies.

## Important Constraints

- Do not assume the dataset is railway-specific unless metadata indicates it.
- Do not assume labels are available.
- Do not assume a fixed sampling rate.
- Do not assume all signals have the same length.
- Do not silently resample or truncate without documenting it.
- Do not overfit the analysis to one dataset.
- Do not report model accuracy without explaining dataset size, split strategy, and limitations.

## Reporting Mindset

The final output should help prepare a concise technical presentation.

The most important outcome is a clear engineering story:

1. What data was provided?
2. What was observed?
3. What signal-processing methods were applied?
4. What patterns were found?
5. What simple baseline method could detect them?
6. What are the limitations?
7. What would be the next steps in a real research project?
