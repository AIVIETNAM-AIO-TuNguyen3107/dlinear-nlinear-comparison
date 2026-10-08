"""Join grid MSE with data-char profiles and plot RQ1 gaps.

`loss` is test MSE in scaled space.
Δ = loss(DLinear) - loss(NLinear); negative means DLinear wins.
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

CharName = Literal["F_S", "F_T"]


def build_baseline_results(results: pd.DataFrame) -> pd.DataFrame:
    """Keep the best-lr test MSE per (model, dataset, L, H).

    Renames seq_len/pred_len/test_loss to lookback/horizon/loss.
    """
    return (
        results.groupby(["model", "dataset", "seq_len", "pred_len"], as_index=False)
        .agg({"test_loss": "min"})
        .rename(
            columns={"seq_len": "lookback", "pred_len": "horizon", "test_loss": "loss"}
        )
        .sort_values(by=["model", "dataset", "lookback", "horizon"])
        .reset_index(drop=True)
    )


def mse_gap(
    results: pd.DataFrame, _profiles: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Δ = loss(DLinear) - loss(NLinear) for each (dataset, lookback, horizon).

    `_profiles` is ignored; call `join_profiles` afterward.
    """
    loss_dlinear = results[results["model"] == "DLinear"][
        ["dataset", "lookback", "horizon", "loss"]
    ]
    loss_nlinear = results[results["model"] == "NLinear"][
        ["dataset", "lookback", "horizon", "loss"]
    ]
    merge_df = pd.merge(
        loss_dlinear,
        loss_nlinear,
        how="inner",
        on=["dataset", "lookback", "horizon"],
        suffixes=("_dlinear", "_nlinear"),
        validate="one_to_one",
    )
    merge_df["delta"] = merge_df["loss_dlinear"] - merge_df["loss_nlinear"]
    return merge_df


def join_profiles(gap: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Left-join STL/ADF columns onto the gap table by `dataset`."""
    return pd.merge(gap, profiles, how="left", on=["dataset"], validate="many_to_one")


def dataset_mean_gap(joined: pd.DataFrame) -> pd.DataFrame:
    """Mean Δ per dataset over lookback x horizon.

    Keeps the first F_T / F_S / adf_pvalue if those columns exist.
    """
    agg: dict[str, str] = {"delta": "mean"}
    for col in ("F_T", "F_S", "adf_pvalue"):
        if col in joined.columns:
            agg[col] = "first"
    return (
        joined.groupby("dataset", as_index=False)
        .agg(agg)
        .sort_values("dataset")
        .reset_index(drop=True)
    )


def scatter_char_vs_delta(
    joined: pd.DataFrame,
    char: CharName = "F_S",
    *,
    ax: Axes | None = None,
    aggregate: bool = True,
) -> Axes:
    """Scatter a profile strength (`F_S` or `F_T`) against Δ.

    With `aggregate=True` (default), one point per dataset using mean Δ.
    """
    df = dataset_mean_gap(joined) if aggregate else joined
    if char not in df.columns:
        raise KeyError(f"{char!r} missing; join profiles first")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0.0, color="0.5", lw=0.8, zorder=0)
    ax.scatter(df[char], df["delta"], s=48, zorder=2)
    if aggregate or df["dataset"].nunique() == len(df):
        for _, row in df.iterrows():
            ax.annotate(
                str(row["dataset"]),
                (row[char], row["delta"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
    ax.set_xlabel(char)
    ax.set_ylabel("Δ = MSE(DLinear) - MSE(NLinear)")
    ax.set_title(f"{char} vs Δ (negative => DLinear better)")
    return ax


def heatmap_delta(
    joined: pd.DataFrame,
    *,
    ax: Axes | None = None,
    how: Literal["mean", "median"] = "mean",
) -> Axes:
    """Heatmap of Δ by dataset and horizon (mean or median over lookback)."""
    pivot = joined.pivot_table(
        index="dataset",
        columns="horizon",
        values="delta",
        aggfunc=how,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=list(pivot.index))
    ax.set_xlabel("horizon")
    ax.set_ylabel("dataset")
    ax.set_title(f"Δ by dataset x horizon ({how} over lookback)")
    fig = ax.figure
    assert isinstance(fig, Figure)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax
