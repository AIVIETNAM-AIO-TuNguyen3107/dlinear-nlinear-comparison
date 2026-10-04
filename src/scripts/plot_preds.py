"""One full-test-timeline compare figure per dataset (best L per model)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.pipeline.dataset import border_pairs, fit_scaler
from src.pipeline.train import build_model, load_checkpoint
from src.utils import load_series

OUT = Path("figures/preds")
CKPT = Path("checkpoints")
MODELS = ("Linear", "NLinear", "DLinear")


def _safe_name(dataset: str) -> str:
    return dataset.replace(" ", "_")


def _horizon(dataset: str) -> int:
    return 5 if dataset == "VIC" else 96


def _raw_series(dataset: str, data_dir: Path) -> np.ndarray:
    spec = settings.get_spec(dataset)
    return load_series(
        data_dir / spec["file"],
        column=spec["column"],
        log=bool(spec.get("log", False)),
    )


def _train_scaler(dataset: str, y_raw: np.ndarray) -> StandardScaler:
    train, _, _ = border_pairs(len(y_raw), dataset)
    return fit_scaler(y_raw, train)


def _to_plot(y_log_or_raw: np.ndarray, dataset: str) -> np.ndarray:
    if settings.get_spec(dataset).get("log"):
        return np.exp(y_log_or_raw)
    return y_log_or_raw


def _ylabel(dataset: str) -> str:
    return "close (price)" if settings.get_spec(dataset).get("log") else "original scale"


def best_runs(
    ckpt_dir: Path = CKPT,
) -> dict[tuple[str, str, int], tuple[float, int, Path]]:
    """(dataset, model, pred_len) -> (test_loss, seq_len, run_dir)."""
    best: dict[tuple[str, str, int], tuple[float, int, Path]] = {}
    for cfg_path in ckpt_dir.glob("*/config.json"):
        run_dir = cfg_path.parent
        if not (run_dir / "best.pt").is_file():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if "test_loss" not in cfg:
            continue
        key = (cfg["dataset"], cfg["model"], int(cfg["pred_len"]))
        t = float(cfg["test_loss"])
        L = int(cfg["seq_len"])
        if key not in best or t < best[key][0]:
            best[key] = (t, L, run_dir)
    return best


def _load_model(run_dir: Path):
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ckpt = load_checkpoint(run_dir / "best.pt", map_location="cpu")
    model = build_model(
        cfg["model"],
        int(cfg["seq_len"]),
        int(cfg["pred_len"]),
        int(cfg["n_vars"]),
        individual=bool(cfg.get("individual", False)),
        kernel=int(cfg["kernel"] or 25),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return cfg, model


@torch.no_grad()
def test_timeline_pred(
    model,
    y_scaled: np.ndarray,
    test: tuple[int, int],
    L: int,
    H: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping H-step forecasts over the test border (scaled space)."""
    t0, t1 = test
    gt = y_scaled[t0:t1].copy()
    pred = np.full_like(gt, np.nan)
    t = t0
    while t + H <= t1:
        if t < L:
            t += H
            continue
        x = torch.from_numpy(y_scaled[t - L : t].astype(np.float32))[None, None, :]
        hat = model(x).squeeze().cpu().numpy()  # (H,)
        pred[t - t0 : t - t0 + H] = hat
        t += H
    return gt, pred


def plot_dataset_compare(
    dataset: str,
    runs: dict[str, tuple[float, int, Path]],
    *,
    out_dir: Path = OUT,
) -> Path | None:
    """3-row figure: Linear / NLinear / DLinear on full test timeline."""
    missing = [m for m in MODELS if m not in runs]
    if missing:
        print(f"skip {dataset}: missing models {missing}")
        return None

    data_dir = settings.data_dir
    y_raw = _raw_series(dataset, data_dir)
    scaler = _train_scaler(dataset, y_raw)
    y_scaled = scaler.transform(y_raw.reshape(-1, 1)).ravel()
    _, _, test = border_pairs(len(y_raw), dataset)
    H = _horizon(dataset)
    ylabel = _ylabel(dataset)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, model_name in zip(axes, MODELS):
        test_mse, L, run_dir = runs[model_name]
        _, model = _load_model(run_dir)
        gt_s, pred_s = test_timeline_pred(model, y_scaled, test, L, H)
        gt = _to_plot(scaler.inverse_transform(gt_s.reshape(-1, 1)).ravel(), dataset)
        # only inverse finite pred steps
        pred_plot = np.full_like(gt, np.nan)
        mask = np.isfinite(pred_s)
        if mask.any():
            pred_plot[mask] = _to_plot(
                scaler.inverse_transform(pred_s[mask].reshape(-1, 1)).ravel(),
                dataset,
            )
        t = np.arange(len(gt))
        ax.plot(t, gt, color="#54A24B", lw=1.2, label="GroundTruth")
        ax.plot(t, pred_plot, color="#E45756", lw=1.2, ls="--", label="Prediction")
        ax.set_title(
            f"{dataset}  |  {model_name}  L={L} H={H}  |  test MSE={test_mse:.6f}"
        )
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc="best")
    axes[-1].set_xlabel("test time step")
    fig.suptitle(f"{dataset} — full test timeline (best lookback per model)", y=1.01)
    fig.tight_layout()
    out = out_dir / f"{_safe_name(dataset)}_compare_full_timeline.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # wipe previous window plots
    for p in OUT.glob("*.png"):
        p.unlink()

    catalog = best_runs()
    datasets = [s["name"] for s in settings.DATASET_SPECS]
    written: list[Path] = []
    for ds in datasets:
        H = _horizon(ds)
        runs: dict[str, tuple[float, int, Path]] = {}
        for m in MODELS:
            key = (ds, m, H)
            if key in catalog:
                runs[m] = catalog[key]
        path = plot_dataset_compare(ds, runs)
        if path is not None:
            written.append(path)
            print(f"wrote {path}")
    print(f"done: {len(written)} figures in {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
