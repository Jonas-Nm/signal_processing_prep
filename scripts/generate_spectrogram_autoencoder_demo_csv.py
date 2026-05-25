"""Generate a file-based synthetic dataset for the spectrogram autoencoder demo.

The output mirrors the repository's other synthetic CSV workflow: signal files
are plain ``time,voltage`` CSVs, while labels and anomaly intervals live in a
separate metadata table that the existing project loaders can attach.
"""

from __future__ import annotations

from pathlib import Path

from signal_processing_prep.demo_data import (
    generate_spectrogram_autoencoder_demo_records as generate_demo_records,
)


def main() -> None:
    """Write the demo dataset to data/raw."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "raw" / "spectrogram_autoencoder_demo"
    metadata = generate_demo_records(output_dir=output_dir)
    print(
        f"Wrote {len(metadata):,} records and metadata to {output_dir}"
    )


if __name__ == "__main__":
    main()
