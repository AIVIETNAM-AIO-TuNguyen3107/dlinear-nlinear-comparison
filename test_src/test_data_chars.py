"""Smoke tests for src.scripts.data_chars (EXECUTION_PLAN §2.2)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scripts.data_chars import (
    PROFILE_FIELDS,
    load_series,
    profile_data_dir,
    profile_dataset,
    stl_strengths,
    write_profiles,
)


def test_stl_strengths_in_unit_interval():
    t = np.arange(200, dtype=float)
    y = (
        0.01 * t
        + np.sin(2 * np.pi * t / 7)
        + 0.05 * np.random.default_rng(1).normal(size=200)
    )
    out = stl_strengths(y, period=7)
    assert 0.0 <= out["F_T"] <= 1.0
    assert 0.0 <= out["F_S"] <= 1.0


def test_stl_strengths_seasonal_series_high_fs():
    t = np.arange(336, dtype=float)
    y = 0.02 * t + 3.0 * np.sin(2 * np.pi * t / 7)
    out = stl_strengths(y, period=7, robust=False)
    assert out["F_S"] > 0.5


def test_profile_dataset_from_array():
    y = np.sin(2 * np.pi * np.arange(100) / 10) + 0.01 * np.arange(100)
    row = profile_dataset(y, period=10, name="toy", with_adf=False, notes="unit test")
    assert row["dataset"] == "toy"
    assert set(PROFILE_FIELDS) <= set(row)
    assert row["n"] == 100
    assert row["adf_pvalue"] == ""


def test_load_and_profile_csv(tmp_path: Path):
    path = tmp_path / "toy.csv"
    n = 120
    t = np.arange(n)
    pd.DataFrame({"date": t, "OT": np.sin(2 * np.pi * t / 12) + 0.01 * t}).to_csv(
        path, index=False
    )
    y = load_series(path, column="OT")
    assert y.shape == (n,)
    row = profile_dataset(path, period=12, column="OT", name="toy_csv", with_adf=True)
    assert row["dataset"] == "toy_csv"
    assert 0.0 <= row["F_T"] <= 1.0


def test_write_profiles_and_profile_data_dir(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    n = 96
    t = np.arange(n)
    pd.DataFrame({"OT": np.sin(2 * np.pi * t / 24) + 0.05 * t}).to_csv(
        data / "ETTh1.csv", index=False
    )
    specs = [
        {
            "name": "ETTh1",
            "file": "ETTh1.csv",
            "column": "OT",
            "period": 24,
            "notes": "test",
        },
        {
            "name": "VIC",
            "file": "VIC.csv",
            "column": "close",
            "period": 7,
            "log": True,
            "notes": "missing — skip",
        },
    ]
    rows = profile_data_dir(data, specs, with_adf=False)
    assert len(rows) == 1
    assert rows[0]["dataset"] == "ETTh1"
    out = write_profiles(rows, tmp_path / "WEEK02_Data_Profiles.csv")
    text = out.read_text(encoding="utf-8")
    assert "dataset,F_T,F_S" in text
    assert "ETTh1" in text


def test_stl_rejects_short_series():
    with pytest.raises(ValueError, match="2\\*period"):
        stl_strengths(np.arange(10, dtype=float), period=24)
