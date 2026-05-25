"""Optional spectrogram autoencoder helpers for controlled synthetic demos."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from signal_processing_prep.artifacts import PredictionTable
from signal_processing_prep.modeling import ModelEvaluation
from signal_processing_prep.preprocessing import segment_signal
from signal_processing_prep.records import SignalDataset, SignalRecord
from signal_processing_prep.time_frequency import spectrogram_analysis


@dataclass(frozen=True)
class SpectrogramAutoencoderConfig:
    """Configuration for synthetic spectrogram autoencoder experiments."""

    patch_window_seconds: float = 0.5
    patch_step_seconds: float = 0.1
    spectrogram_window_seconds: float = 0.064
    spectrogram_step_seconds: float = 0.032
    max_frequency_hz: float | None = 800.0
    n_epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 1e-3
    latent_channels: int = 4
    threshold_quantile: float = 0.95
    random_state: int = 0
    device: str = "cpu"


@dataclass(frozen=True)
class SpectrogramAutoencoderDemoDataset:
    """Train/test records for the controlled spectrogram autoencoder demo."""

    train_records: tuple[SignalRecord, ...]
    test_records: tuple[SignalRecord, ...]


@dataclass(frozen=True)
class SpectrogramPatchSet:
    """Fixed-shape log-power spectrogram patches and their metadata."""

    patches: NDArray[np.float64]
    metadata: pd.DataFrame
    frequencies_hz: NDArray[np.float64]
    times_seconds: NDArray[np.float64]


@dataclass(frozen=True)
class SpectrogramAutoencoderResult:
    """Artifacts returned by a spectrogram autoencoder scoring run."""

    evaluation: ModelEvaluation
    training_losses: tuple[float, ...]
    threshold: float
    train_patch_mean: float
    train_patch_std: float
    train_patches: SpectrogramPatchSet
    test_patches: SpectrogramPatchSet
    reconstructed_test_patches: NDArray[np.float64]


@dataclass(frozen=True)
class SpectrogramAutoencoderExperiment:
    """Configured optional end-to-end spectrogram autoencoder experiment."""

    config: SpectrogramAutoencoderConfig = SpectrogramAutoencoderConfig()

    def run(self, dataset: SignalDataset) -> SpectrogramAutoencoderResult:
        """Select annotated train/test records, train, and score held-out patches."""
        split = split_autoencoder_demo_records(dataset.records)
        return run_spectrogram_autoencoder(split.train_records, split.test_records, self.config)


def split_autoencoder_demo_records(
    records: Sequence[SignalRecord],
) -> SpectrogramAutoencoderDemoDataset:
    """Split typed records into normal training records and mixed test records."""
    train_records: list[SignalRecord] = []
    test_records: list[SignalRecord] = []
    for record in records:
        split = str(record.annotations.split or "").lower()
        label = str(record.label or "").lower()
        if split == "train":
            if label != "normal":
                raise ValueError("Autoencoder training records must be labeled normal.")
            train_records.append(record)
        elif split == "test":
            test_records.append(record)
        else:
            raise ValueError("Autoencoder demo metadata must label each record split as train or test.")
    if not train_records:
        raise ValueError("At least one normal train record is required.")
    if not test_records:
        raise ValueError("At least one test record is required.")
    if not any(str(record.label).lower() == "anomalous" for record in test_records):
        raise ValueError("At least one anomalous test record is required.")
    return SpectrogramAutoencoderDemoDataset(
        train_records=tuple(train_records),
        test_records=tuple(test_records),
    )


def spectrogram_patches_from_records(
    records: tuple[SignalRecord, ...] | list[SignalRecord],
    config: SpectrogramAutoencoderConfig | None = None,
) -> SpectrogramPatchSet:
    """Convert signal records into fixed-size log-power spectrogram patches."""
    if config is None:
        config = SpectrogramAutoencoderConfig()
    _validate_patch_config(config)
    if len(records) == 0:
        raise ValueError("At least one SignalRecord is required.")

    patches: list[NDArray[np.float64]] = []
    metadata_rows: list[dict[str, object]] = []
    reference_shape: tuple[int, int] | None = None
    reference_frequencies: NDArray[np.float64] | None = None
    reference_times: NDArray[np.float64] | None = None

    for record in records:
        windows = segment_signal(
            record,
            window_seconds=config.patch_window_seconds,
            step_seconds=config.patch_step_seconds,
        )
        for window in windows:
            patch, frequencies, times = _spectrogram_patch(window, config)
            if reference_shape is None:
                reference_shape = patch.shape
                reference_frequencies = frequencies
                reference_times = times
            elif patch.shape != reference_shape:
                raise ValueError("All spectrogram patches must have the same shape.")
            elif (
                reference_frequencies is None
                or reference_times is None
                or not np.allclose(frequencies, reference_frequencies)
                or not np.allclose(times, reference_times)
            ):
                raise ValueError("All spectrogram patches must share frequency and time axes.")
            patches.append(patch)
            metadata_rows.append(_patch_metadata(record, window))

    if not patches or reference_frequencies is None or reference_times is None:
        raise ValueError("No spectrogram patches were produced.")
    return SpectrogramPatchSet(
        patches=np.stack(patches).astype(np.float64),
        metadata=pd.DataFrame(metadata_rows),
        frequencies_hz=reference_frequencies,
        times_seconds=reference_times,
    )


def run_spectrogram_autoencoder(
    train_records: tuple[SignalRecord, ...] | list[SignalRecord],
    test_records: tuple[SignalRecord, ...] | list[SignalRecord],
    config: SpectrogramAutoencoderConfig | None = None,
) -> SpectrogramAutoencoderResult:
    """Train a PyTorch spectrogram autoencoder on normal records and score test patches."""
    torch, nn, data = _require_torch()
    if config is None:
        config = SpectrogramAutoencoderConfig()
    _validate_training_config(config)

    torch.manual_seed(config.random_state)
    train_patches = spectrogram_patches_from_records(train_records, config)
    test_patches = spectrogram_patches_from_records(test_records, config)
    train_values, test_values, patch_mean, patch_std = _normalize_patch_sets(
        train_patches.patches,
        test_patches.patches,
    )

    device = torch.device(config.device)
    model = _build_autoencoder_model(nn, config.latent_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    dataset = data.TensorDataset(torch.as_tensor(train_values[:, None, :, :], dtype=torch.float32))
    generator = torch.Generator()
    generator.manual_seed(config.random_state)
    loader = data.DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    training_losses: list[float] = []
    model.train()
    for _ in range(config.n_epochs):
        total_loss = 0.0
        total_count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            total_count += len(batch)
        training_losses.append(total_loss / max(total_count, 1))

    train_errors, _ = _reconstruction_errors(torch, model, train_values, device, config.batch_size)
    test_errors, reconstructed_test = _reconstruction_errors(
        torch,
        model,
        test_values,
        device,
        config.batch_size,
    )
    threshold = float(np.quantile(train_errors, config.threshold_quantile))
    prediction_frame = _autoencoder_predictions(test_patches.metadata, test_errors, threshold)
    evaluation = ModelEvaluation(
        model_name="spectrogram_autoencoder",
        task_type="unsupervised_anomaly_score",
        estimator=model,
        feature_columns=("log_power_spectrogram_patch",),
        metrics={
            "n_train_patches": float(len(train_patches.patches)),
            "n_test_patches": float(len(test_patches.patches)),
            "n_epochs": float(config.n_epochs),
            "threshold": threshold,
            "threshold_quantile": float(config.threshold_quantile),
            "final_training_loss": float(training_losses[-1]) if training_losses else float("nan"),
        },
        predictions=PredictionTable.anomaly_scores(prediction_frame),
        split_strategy="trained on synthetic normal records; scored held-out synthetic test patches",
    )
    return SpectrogramAutoencoderResult(
        evaluation=evaluation,
        training_losses=tuple(training_losses),
        threshold=threshold,
        train_patch_mean=patch_mean,
        train_patch_std=patch_std,
        train_patches=train_patches,
        test_patches=test_patches,
        reconstructed_test_patches=reconstructed_test,
    )


def _spectrogram_patch(
    window: SignalRecord,
    config: SpectrogramAutoencoderConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    result = spectrogram_analysis(
        window,
        window_seconds=config.spectrogram_window_seconds,
        step_seconds=config.spectrogram_step_seconds,
    )
    frequencies = result.frequencies_hz
    power = result.power
    if config.max_frequency_hz is not None:
        frequency_mask = frequencies <= config.max_frequency_hz
        if not np.any(frequency_mask):
            raise ValueError("max_frequency_hz leaves no spectrogram frequency bins.")
        frequencies = frequencies[frequency_mask]
        power = power[frequency_mask, :]
    patch = 10.0 * np.log10(np.maximum(power, 1e-24))
    if not np.isfinite(patch).all():
        raise ValueError("Spectrogram patch contains non-finite values.")
    return patch.astype(np.float64), frequencies.astype(np.float64), result.times_seconds.astype(np.float64)


def _patch_metadata(record: SignalRecord, window: SignalRecord) -> dict[str, object]:
    if window.segment_span is None:
        raise ValueError("Spectrogram patches require records created by segment_signal().")
    start = window.segment_span.start_seconds
    end = window.segment_span.end_seconds
    intervals = _anomaly_intervals(record)
    overlaps = any(start < interval["end_seconds"] and end > interval["start_seconds"] for interval in intervals)
    return {
        "record_name": record.name,
        "label": record.label,
        "patch_start_seconds": start,
        "patch_end_seconds": end,
        "patch_center_seconds": 0.5 * (start + end),
        "known_anomaly_overlap": bool(overlaps),
        "is_synthetic_anomaly_record": bool(record.annotations.is_anomalous),
        "anomaly_kind": record.annotations.anomaly_kind,
    }


def _anomaly_intervals(record: SignalRecord) -> list[dict[str, float]]:
    return [
        {"start_seconds": interval.start_seconds, "end_seconds": interval.end_seconds}
        for interval in record.annotations.intervals
    ]


def _normalize_patch_sets(
    train_patches: NDArray[np.float64],
    test_patches: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    patch_mean = float(np.mean(train_patches))
    patch_std = float(np.std(train_patches))
    if patch_std <= 0.0 or not np.isfinite(patch_std):
        patch_std = 1.0
    return (
        ((train_patches - patch_mean) / patch_std).astype(np.float32),
        ((test_patches - patch_mean) / patch_std).astype(np.float32),
        patch_mean,
        patch_std,
    )


def _autoencoder_predictions(
    metadata: pd.DataFrame,
    errors: NDArray[np.float64],
    threshold: float,
) -> pd.DataFrame:
    predictions = metadata.copy().reset_index(drop=True)
    predictions.insert(0, "row_index", np.arange(len(predictions)))
    predictions.insert(1, "anomaly_score", errors.astype(np.float64))
    predictions["reconstruction_error"] = predictions["anomaly_score"]
    predictions["predicted_anomaly"] = predictions["anomaly_score"] > threshold
    predictions["anomaly_rank"] = (
        predictions["anomaly_score"].rank(method="first", ascending=False).astype(int)
    )
    return predictions


def _reconstruction_errors(
    torch: Any,
    model: Any,
    values: NDArray[np.float32],
    device: Any,
    batch_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    model.eval()
    tensor = torch.as_tensor(values[:, None, :, :], dtype=torch.float32)
    reconstructed_batches: list[NDArray[np.float64]] = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device)
            reconstructed = model(batch).cpu().numpy()[:, 0, :, :]
            reconstructed_batches.append(reconstructed.astype(np.float64))
    reconstructed_values = np.concatenate(reconstructed_batches, axis=0)
    errors = np.mean(np.square(values.astype(np.float64) - reconstructed_values), axis=(1, 2))
    return errors.astype(np.float64), reconstructed_values


def _build_autoencoder_model(nn: Any, latent_channels: int) -> Any:
    if latent_channels <= 0:
        raise ValueError("latent_channels must be positive.")

    class SpectrogramConvAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, latent_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(latent_channels, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 1, kernel_size=3, padding=1),
            )

        def forward(self, values: Any) -> Any:
            original_shape = values.shape[-2:]
            encoded = self.encoder(values)
            decoded = nn.functional.interpolate(
                encoded,
                size=original_shape,
                mode="bilinear",
                align_corners=False,
            )
            return self.decoder(decoded)

    return SpectrogramConvAutoencoder()


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils import data
    except ImportError as error:
        raise ImportError(
            "PyTorch is required for spectrogram autoencoder training. "
            "Install it with `python -m pip install -e .[deep-learning]`."
        ) from error
    return torch, nn, data


def _validate_patch_config(config: SpectrogramAutoencoderConfig) -> None:
    if config.patch_window_seconds <= 0:
        raise ValueError("patch_window_seconds must be positive.")
    if config.patch_step_seconds <= 0:
        raise ValueError("patch_step_seconds must be positive.")
    if config.patch_step_seconds > config.patch_window_seconds:
        raise ValueError("patch_step_seconds must not exceed patch_window_seconds.")
    if config.spectrogram_window_seconds <= 0:
        raise ValueError("spectrogram_window_seconds must be positive.")
    if config.spectrogram_step_seconds <= 0:
        raise ValueError("spectrogram_step_seconds must be positive.")
    if config.spectrogram_step_seconds > config.spectrogram_window_seconds:
        raise ValueError("spectrogram_step_seconds must not exceed spectrogram_window_seconds.")
    if config.spectrogram_window_seconds > config.patch_window_seconds:
        raise ValueError("spectrogram_window_seconds must not exceed patch_window_seconds.")
    if config.max_frequency_hz is not None and config.max_frequency_hz <= 0:
        raise ValueError("max_frequency_hz must be positive.")


def _validate_training_config(config: SpectrogramAutoencoderConfig) -> None:
    _validate_patch_config(config)
    if config.n_epochs <= 0:
        raise ValueError("n_epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if not 0.0 < config.threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be in the interval (0, 1).")
