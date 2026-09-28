"""STL-based data characteristics (Hyndman / FPP3 strengths).

Descriptive profiling may use the full series.
Place CSVs under Working_Files/data/ (see DATASET_SPECS); re-run to dump
WEEK02_Data_Profiles.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

# Working_Files/data/ — fill later (W3 loaders may share this layout)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_PROFILES_CSV = Path(__file__).resolve().parents[1] / "WEEK02_Data_Profiles.csv"

# period: ETTh hourly → 24; daily series → 7 (document in notes; use 365 only if long enough)
DATASET_SPECS: list[dict[str, Any]] = [
    {
        "name": "ETTh1",
        "file": "ETTh1.csv",
        "column": "OT",
        "period": 24,
        "notes": "hourly; STL period=24; target OT",
    },
    {
        "name": "ETTh2",
        "file": "ETTh2.csv",
        "column": "OT",
        "period": 24,
        "notes": "hourly; STL period=24; target OT",
    },
    {
        "name": "Weather",
        "file": "Weather.csv",
        "column": "OT",
        "period": 24,
        "notes": "Weather LTSF; STL period=24; target OT",
    },
    {
        "name": "Exchange-Rate",
        "file": "Exchange.csv",
        "column": "OT",
        "period": 7,
        "notes": "daily FX; STL period=7; target OT",
    },
    {
        "name": "Electricity",
        "file": "Electricity.csv",
        "column": "OT",
        "period": 24,
        "notes": "hourly electricity; STL period=24; target OT",
    },
    {
        "name": "VIC",
        "file": "VIC.csv",
        "column": "close",
        "period": 7,
        "log": True,
        "notes": "VNM Vingroup daily close; log(close); STL period=7",
    },
]

PROFILE_FIELDS = (
    "dataset",
    "F_T",
    "F_S",
    "period",
    "n",
    "adf_pvalue",
    "notes",
)


def _as_1d(y: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError("series must have length >= 2")
    if not np.isfinite(arr).all():
        raise ValueError("series contains NaN/Inf")
    return arr


def _strength(var_comp_plus_r: float, var_r: float) -> float:
    if var_comp_plus_r <= 0 or not np.isfinite(var_comp_plus_r):
        return 0.0
    return float(max(0.0, 1.0 - var_r / var_comp_plus_r))


def stl_strengths(
    y: np.ndarray | Sequence[float],
    period: int,
    *,
    robust: bool = True,
) -> dict[str, float]:
    """Fit STL and return F_T, F_S in [0, 1] (FPP3)."""
    if period < 2:
        raise ValueError("period must be >= 2")
    arr = _as_1d(y)
    if arr.size < 2 * period:
        raise ValueError(f"need length >= 2*period ({2 * period}), got {arr.size}")

    res = STL(arr, period=period, robust=robust).fit()
    T, S, R = np.asarray(res.trend), np.asarray(res.seasonal), np.asarray(res.resid)
    var_r = float(np.var(R))
    return {
        "F_T": _strength(float(np.var(T + R)), var_r),
        "F_S": _strength(float(np.var(S + R)), var_r),
    }


def adf_pvalue(y: np.ndarray | Sequence[float]) -> float:
    """ADF unit-root test p-value (non-stationarity proxy)."""
    arr = _as_1d(y)
    # autolag AIC; regression with constant
    stat = adfuller(arr, autolag="AIC", result_object=True)
    return float(stat.pvalue)


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
    return _as_1d(y)


def profile_dataset(
    path_or_array: str | Path | np.ndarray | Sequence[float],
    *,
    period: int,
    column: str | None = None,
    name: str = "series",
    log: bool = False,
    robust: bool = True,
    with_adf: bool = True,
    notes: str = "",
) -> dict[str, Any]:
    """Profile one series: STL strengths (+ optional ADF)."""
    if isinstance(path_or_array, (str, Path)):
        y = load_series(path_or_array, column=column, log=log)
    else:
        y = _as_1d(path_or_array)
        if log:
            if np.any(y <= 0):
                raise ValueError("log=True requires strictly positive values")
            y = np.log(y)

    strengths = stl_strengths(y, period=period, robust=robust)
    row: dict[str, Any] = {
        "dataset": name,
        "F_T": strengths["F_T"],
        "F_S": strengths["F_S"],
        "period": int(period),
        "n": int(y.size),
        "adf_pvalue": adf_pvalue(y) if with_adf else "",
        "notes": notes,
    }
    return row


def write_profiles(
    rows: Sequence[Mapping[str, Any]],
    out_csv: str | Path = DEFAULT_PROFILES_CSV,
) -> Path:
    """Write profile rows to CSV (columns: dataset, F_T, F_S, …)."""
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PROFILE_FIELDS})
    return out


def profile_data_dir(
    data_dir: str | Path = DATA_DIR,
    specs: Sequence[Mapping[str, Any]] = DATASET_SPECS,
    *,
    robust: bool = True,
    with_adf: bool = True,
) -> list[dict[str, Any]]:
    """Profile every spec whose file exists under data_dir; skip missing."""
    root = Path(data_dir)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = root / spec["file"]
        if not path.is_file():
            continue
        rows.append(
            profile_dataset(
                path,
                period=int(spec["period"]),
                column=spec.get("column"),
                name=str(spec["name"]),
                log=bool(spec.get("log", False)),
                robust=robust,
                with_adf=with_adf,
                notes=str(spec.get("notes", "")),
            )
        )
    return rows


def _smoke_series(n: int = 336, period: int = 7, seed: int = 0) -> np.ndarray:
    """ponytail: synthetic trend+season for self-check when data/ is empty."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return (
        0.02 * t + 3.0 * np.sin(2 * np.pi * t / period) + 0.15 * rng.standard_normal(n)
    )


if __name__ == "__main__":
    rows = profile_data_dir()
    if rows:
        path = write_profiles(rows)
        print(f"wrote {len(rows)} rows → {path}")
        for r in rows:
            print(
                f"  {r['dataset']}: F_T={r['F_T']:.4f} F_S={r['F_S']:.4f} "
                f"period={r['period']} n={r['n']}"
            )
    else:
        print(
            f"no CSVs in {DATA_DIR}; expected e.g. ETTh1.csv, VIC.csv — skip CSV write"
        )
        smoke = profile_dataset(
            _smoke_series(), period=7, name="smoke", notes="synthetic"
        )
        assert 0.0 <= smoke["F_T"] <= 1.0 and 0.0 <= smoke["F_S"] <= 1.0
        assert smoke["F_S"] > 0.5, smoke
        print(
            f"smoke ok: F_T={smoke['F_T']:.4f} F_S={smoke['F_S']:.4f} "
            f"adf_p={smoke['adf_pvalue']:.4g}"
        )
