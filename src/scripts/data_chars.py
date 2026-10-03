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
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

from src.config import settings
from src.utils import as_1d, load_series, logger

PROFILE_FIELDS = (
    "dataset",
    "F_T",
    "F_S",
    "period",
    "n",
    "adf_pvalue",
    "notes",
)


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
    arr = as_1d(y)
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
    arr = as_1d(y)
    # autolag AIC; regression with constant
    stat = adfuller(arr, autolag="AIC", result_object=True)
    return float(stat.pvalue)


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
    logger.debug(f"Profiling dataset: {name}, period={period}, log={log}")
    if isinstance(path_or_array, (str, Path)):
        y = load_series(path_or_array, column=column, log=log)
    else:
        y = as_1d(path_or_array)
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
    logger.info(
        f"Profiled {name}: F_T={row['F_T']:.4f}, F_S={row['F_S']:.4f}, n={row['n']}, period={row['period']}"
    )
    return row


def write_profiles(
    rows: Sequence[Mapping[str, Any]],
    out_csv: str | Path = settings.profiles_csv,
) -> Path:
    """Write profile rows to CSV (columns: dataset, F_T, F_S, …)."""
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PROFILE_FIELDS})
    logger.info(f"Wrote {len(rows)} profile rows to {out}")
    return out


def profile_data_dir(
    data_dir: str | Path = settings.data_dir,
    specs: Sequence[Mapping[str, Any]] = settings.DATASET_SPECS,
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
            logger.warning(f"Missing file {path}, skipping...")
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
    logger.info(f"Profiled {len(rows)} datasets from {data_dir}")
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
        logger.info(f"Wrote {len(rows)} rows → {path}")
        for r in rows:
            logger.info(
                f"{r['dataset']}: F_T={r['F_T']:.4f} F_S={r['F_S']:.4f} period={r['period']} n={r['n']}"
            )
    else:
        logger.warning(
            f"No CSVs in {settings.data_dir}; expected e.g. ETTh1.csv, VIC.csv — skip CSV write"
        )
        smoke = profile_dataset(
            _smoke_series(), period=7, name="smoke", notes="synthetic"
        )
        assert 0.0 <= smoke["F_T"] <= 1.0 and 0.0 <= smoke["F_S"] <= 1.0
        assert smoke["F_S"] > 0.5, smoke
        logger.info(
            f"smoke ok: F_T={smoke['F_T']:.4f} F_S={smoke['F_S']:.4f} "
            f"adf_p={smoke['adf_pvalue']:.4g}"
        )
