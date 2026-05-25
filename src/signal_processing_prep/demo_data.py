"""File-based synthetic datasets used by optional walkthroughs and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SAMPLING_RATE_HZ = 2_000.0
DURATION_SECONDS = 4.0
RANDOM_SEED = 20260524
N_TRAIN_RECORDS = 8
N_TEST_NORMAL_RECORDS = 3
N_TEST_ANOMALOUS_RECORDS = 3


def generate_spectrogram_autoencoder_demo_records(
    *,
    output_dir: str | Path,
    n_train_records: int = N_TRAIN_RECORDS,
    n_test_normal_records: int = N_TEST_NORMAL_RECORDS,
    n_test_anomalous_records: int = N_TEST_ANOMALOUS_RECORDS,
    duration_seconds: float = DURATION_SECONDS,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Write spectrogram-autoencoder demo CSVs and return their metadata."""
    if n_train_records <= 0:
        raise ValueError("n_train_records must be positive.")
    if n_test_normal_records < 0 or n_test_anomalous_records <= 0:
        raise ValueError(
            "Test dataset must include non-negative normal and positive anomalous counts."
        )
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")

    root = Path(output_dir)
    records_dir = root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing_signal_csvs(records_dir)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []

    for index in range(n_train_records):
        rows.append(
            _write_record(
                records_dir,
                split="train",
                label="normal",
                name=f"normal_train_{index:02d}",
                seed=int(rng.integers(0, 1_000_000)),
                duration_seconds=duration_seconds,
                sampling_rate_hz=sampling_rate_hz,
            )
        )
    for index in range(n_test_normal_records):
        rows.append(
            _write_record(
                records_dir,
                split="test",
                label="normal",
                name=f"normal_test_{index:02d}",
                seed=int(rng.integers(0, 1_000_000)),
                duration_seconds=duration_seconds,
                sampling_rate_hz=sampling_rate_hz,
            )
        )

    anomaly_kinds = ("transient_tone", "broadband_impulse")
    for index in range(n_test_anomalous_records):
        rows.append(
            _write_record(
                records_dir,
                split="test",
                label="anomalous",
                name=f"anomalous_test_{index:02d}",
                seed=int(rng.integers(0, 1_000_000)),
                duration_seconds=duration_seconds,
                sampling_rate_hz=sampling_rate_hz,
                anomaly_kind=anomaly_kinds[index % len(anomaly_kinds)],
            )
        )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(root / "metadata.csv", index=False)
    return metadata


def _clear_existing_signal_csvs(records_dir: Path) -> None:
    for path in records_dir.glob("*.csv"):
        if path.is_file():
            path.unlink()


def _write_record(
    records_dir: Path,
    *,
    split: str,
    label: str,
    name: str,
    seed: int,
    duration_seconds: float,
    sampling_rate_hz: float,
    anomaly_kind: str | None = None,
) -> dict[str, object]:
    frame, anomaly_metadata = _vibration_like_frame(
        duration_seconds=duration_seconds,
        sampling_rate_hz=sampling_rate_hz,
        seed=seed,
        anomaly_kind=anomaly_kind,
    )
    file_name = f"{name}.csv"
    frame.to_csv(records_dir / file_name, index=False)
    return {
        "file_name": file_name,
        "source_path": (Path("records") / file_name).as_posix(),
        "split": split,
        "label": label,
        "is_anomalous": anomaly_kind is not None,
        "anomaly_kind": anomaly_kind,
        "anomaly_start_seconds": anomaly_metadata.get("anomaly_start_seconds"),
        "anomaly_end_seconds": anomaly_metadata.get("anomaly_end_seconds"),
        "sampling_rate_hz": sampling_rate_hz,
        "duration_seconds": duration_seconds,
        "seed": seed,
    }


def _vibration_like_frame(
    *,
    duration_seconds: float,
    sampling_rate_hz: float,
    seed: int,
    anomaly_kind: str | None,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    n_samples = int(round(duration_seconds * sampling_rate_hz))
    time = np.arange(n_samples, dtype=np.float64) / sampling_rate_hz
    amplitudes = np.array([0.35, 0.18, 0.08]) * rng.uniform(0.85, 1.15, size=3)
    frequencies = np.array([45.0, 130.0, 260.0]) * rng.uniform(0.96, 1.04, size=3)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
    drift = 1.0 + 0.08 * np.sin(
        2.0 * np.pi * 0.25 * time + rng.uniform(0.0, 2.0 * np.pi)
    )
    voltage = np.zeros_like(time)
    for amplitude, frequency, phase in zip(amplitudes, frequencies, phases):
        voltage += amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
    voltage = drift * voltage + rng.normal(0.0, 0.035, size=time.size)

    anomaly_metadata: dict[str, float | str] = {}
    if anomaly_kind is not None:
        anomaly_metadata = _inject_demo_anomaly(
            voltage,
            time,
            sampling_rate_hz=sampling_rate_hz,
            rng=rng,
            anomaly_kind=anomaly_kind,
        )
    return pd.DataFrame({"time": np.round(time, 8), "voltage": np.round(voltage, 8)}), anomaly_metadata


def _inject_demo_anomaly(
    voltage: np.ndarray,
    time: np.ndarray,
    *,
    sampling_rate_hz: float,
    rng: np.random.Generator,
    anomaly_kind: str,
) -> dict[str, float | str]:
    duration_seconds = voltage.size / sampling_rate_hz
    if anomaly_kind == "transient_tone":
        anomaly_duration = 0.24
        start_seconds = _random_anomaly_start(duration_seconds, anomaly_duration, rng=rng)
        end_seconds = min(start_seconds + anomaly_duration, duration_seconds)
        mask = (time >= start_seconds) & (time < end_seconds)
        envelope = np.hanning(max(np.count_nonzero(mask), 1))
        voltage[mask] += 0.9 * envelope * np.sin(2.0 * np.pi * 620.0 * time[mask])
    elif anomaly_kind == "broadband_impulse":
        anomaly_duration = 0.12
        start_seconds = _random_anomaly_start(duration_seconds, anomaly_duration, rng=rng)
        end_seconds = min(start_seconds + anomaly_duration, duration_seconds)
        mask = (time >= start_seconds) & (time < end_seconds)
        envelope = np.hanning(max(np.count_nonzero(mask), 1))
        voltage[mask] += 0.7 * envelope * rng.normal(0.0, 1.0, size=np.count_nonzero(mask))
    else:
        raise ValueError("anomaly_kind must be 'transient_tone' or 'broadband_impulse'.")
    return {
        "anomaly_start_seconds": float(start_seconds),
        "anomaly_end_seconds": float(end_seconds),
        "anomaly_kind": anomaly_kind,
    }


def _random_anomaly_start(
    duration_seconds: float,
    anomaly_duration_seconds: float,
    *,
    rng: np.random.Generator,
) -> float:
    latest_start = max(duration_seconds - anomaly_duration_seconds, 0.0)
    earliest_start = min(1.2, latest_start)
    if latest_start <= earliest_start:
        return float(earliest_start)
    return float(rng.uniform(earliest_start, latest_start))
