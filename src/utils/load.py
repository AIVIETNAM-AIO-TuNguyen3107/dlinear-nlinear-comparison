from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def as_1d(y: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError("series must have length >= 2")
    if not np.isfinite(arr).all():
        raise ValueError("series contains NaN/Inf")
    return arr


def load_series(
    path: str | Path,
    *,
    column: str | None = None,
    log: bool = False,
) -> np.ndarray:
    """Load a univariate column from CSV. Default: last numeric column."""
    df = pd.read_csv(path)
    if column is None:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError(f"no numeric columns in {path}")
        series = numeric.iloc[:, -1]
    else:
        if column not in df.columns:
            raise KeyError(f"column {column!r} not in {path}; have {list(df.columns)}")
        series = df[column]
    y = series.to_numpy(dtype=float)
    if log:
        if np.any(y <= 0):
            raise ValueError("log=True requires strictly positive values")
        y = np.log(y)
    return as_1d(y)
