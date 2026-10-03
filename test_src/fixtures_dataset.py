"""CSV fixtures for dataset loader tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def vic_log_close(n: int = 100) -> np.ndarray:
    """close = [e, e^2, e^3, e, e, ...] so log(close) starts [1, 2, 3, 1, ...]."""
    return np.array([np.e, np.e**2, np.e**3] + [np.e] * (n - 3), dtype=float)


@pytest.fixture
def vic_series() -> np.ndarray:
    """Logged VIC series (what the loader uses after log=True)."""
    return np.log(vic_log_close())


@pytest.fixture
def vic_data_dir(tmp_path: Path) -> Path:
    pd.DataFrame({"close": vic_log_close()}).to_csv(tmp_path / "VIC.csv", index=False)
    return tmp_path


@pytest.fixture
def weather_data_dir(tmp_path: Path) -> Path:
    """OT column, 70/15/15 path (not VIC log)."""
    pd.DataFrame({"OT": np.arange(100, dtype=float)}).to_csv(
        tmp_path / "Weather.csv", index=False
    )
    return tmp_path
