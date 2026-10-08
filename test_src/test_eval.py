"""Smoke tests for src.pipeline.eval (W4 data-char join)."""

import pandas as pd
import pytest

from src.pipeline.eval import (
    build_baseline_results,
    dataset_mean_gap,
    join_profiles,
    mse_gap,
)


def _grid_mini() -> pd.DataFrame:
    # two lrs per cell; min should pick the better one
    rows = [
        ("DLinear", "A", 96, 96, 0.01, 0.20),
        ("DLinear", "A", 96, 96, 0.001, 0.10),
        ("NLinear", "A", 96, 96, 0.01, 0.15),
        ("NLinear", "A", 96, 96, 0.001, 0.12),
        ("DLinear", "B", 96, 96, 0.001, 0.05),
        ("NLinear", "B", 96, 96, 0.001, 0.20),
    ]
    return pd.DataFrame(
        rows,
        columns=["model", "dataset", "seq_len", "pred_len", "lr", "test_loss"],
    )


def _profiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["A", "B"],
            "F_T": [0.9, 0.5],
            "F_S": [0.2, 0.8],
            "adf_pvalue": [0.1, 0.01],
        }
    )


def test_build_baseline_uses_best_lr_min():
    out = build_baseline_results(_grid_mini())
    a_d = out[(out.model == "DLinear") & (out.dataset == "A")].iloc[0]
    assert a_d["loss"] == 0.10
    assert set(out.columns) == {"model", "dataset", "lookback", "horizon", "loss"}
    assert len(out) == 4


def test_mse_gap_sign_and_join():
    results = build_baseline_results(_grid_mini())
    gap = mse_gap(results)
    joined = join_profiles(gap, _profiles())
    assert {"F_S", "F_T", "delta"} <= set(joined.columns)
    a = joined[joined.dataset == "A"].iloc[0]
    # A: 0.10 - 0.12 = -0.02 (DLinear better)
    assert a["delta"] == pytest.approx(-0.02)
    assert a["F_S"] == 0.2
    b = joined[joined.dataset == "B"].iloc[0]
    assert b["delta"] == pytest.approx(-0.15)
    mean = dataset_mean_gap(joined)
    assert list(mean["dataset"]) == ["A", "B"]
    assert mean.loc[mean.dataset == "B", "delta"].iloc[0] == pytest.approx(-0.15)
