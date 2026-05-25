"""Generate the synthetic single-record vibration anomaly CSV.

The output is intentionally simple: one timestamp column and one voltage column,
matching the kind of export the project data loader can ingest directly.
"""

from __future__ import annotations

from pathlib import Path

from signal_processing_prep.demo_data import generate_vibration_anomaly_frame


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
