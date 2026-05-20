"""Generate the synthetic single-record vibration anomaly CSV.

The output is intentionally simple: one timestamp column and one voltage column,
matching the kind of export the project data loader can ingest directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SAMPLING_RATE_HZ = 20_000.0
DURATION_SECONDS = 12.0
RANDOM_SEED = 20260520
ANOMALY_START_SECONDS = 7.15
ANOMALY_DURATION_SECONDS = 0.42


def generate_vibration_anomaly_frame() -> pd.DataFrame:
    """Return a synthetic vibration/acoustic sensor record with one anomaly."""
    n_samples = int(SAMPLING_RATE_HZ * DURATION_SECONDS)
    time = np.arange(n_samples, dtype=np.float64) / SAMPLING_RATE_HZ
    rng = np.random.default_rng(RANDOM_SEED)

    voltage = 0.018 * np.sin(2.0 * np.pi * 120.0 * time + 0.2)
    voltage += 0.007 * np.sin(2.0 * np.pi * 240.0 * time + 1.1)
    voltage += 0.004 * np.sin(2.0 * np.pi * 360.0 * time + 2.0)
    voltage += 0.003 * np.sin(2.0 * np.pi * 1850.0 * time)
    voltage += 0.0025 * rng.standard_normal(n_samples)

    amplitude_modulation = 1.0 + 0.08 * np.sin(2.0 * np.pi * 0.18 * time)
    voltage *= amplitude_modulation

    anomaly_mask = (time >= ANOMALY_START_SECONDS) & (
        time < ANOMALY_START_SECONDS + ANOMALY_DURATION_SECONDS
    )
    anomaly_time = time[anomaly_mask] - ANOMALY_START_SECONDS
    impact_train = (np.sin(2.0 * np.pi * 85.0 * anomaly_time) > 0.96).astype(float)
    resonance = np.sin(2.0 * np.pi * 2600.0 * anomaly_time) * np.exp(
        -18.0 * anomaly_time
    )
    burst_envelope = np.hanning(anomaly_mask.sum())
    voltage[anomaly_mask] += 0.045 * burst_envelope * resonance
    voltage[anomaly_mask] += 0.025 * burst_envelope * impact_train
    voltage[anomaly_mask] += 0.010 * rng.standard_normal(anomaly_mask.sum())

    glitch_indices = rng.choice(
        np.arange(10_000, n_samples - 10_000),
        size=10,
        replace=False,
    )
    voltage[glitch_indices] += rng.choice(
        [-1.0, 1.0],
        size=glitch_indices.size,
    ) * rng.uniform(0.025, 0.04, size=glitch_indices.size)

    return pd.DataFrame(
        {
            "time": np.round(time, 8),
            "voltage": np.round(voltage, 8),
        }
    )


def main() -> None:
    """Write the synthetic CSV to data/raw."""
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "raw" / "vibration_anomaly_single_record.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = generate_vibration_anomaly_frame()
    frame.to_csv(output_path, index=False)
    print(f"Wrote {len(frame):,} samples to {output_path}")


if __name__ == "__main__":
    main()
