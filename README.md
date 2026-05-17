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

Run the test suite:

```powershell
pytest
```

The `.venv/` directory is local developer state and is ignored by Git.
