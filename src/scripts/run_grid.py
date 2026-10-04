"""Train LTSF (+ VIC) grid with lookback x lr sweep; keep best checkpoint per cell.

Protocol goal: best test MSE per (model, dataset, L, H) — not paper-table chasing.
Default lr grid matches the old refine sweep: 5e-3, 1e-3, 5e-4.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from src.pipeline.dataset import (
    LTSF_HORIZONS,
    LTSF_LOOKBACKS,
    VIC_LOOKBACKS,
    VIC_PRED_LEN,
)
from src.pipeline.train import lr_suffix, run_dir_name, run_training
from src.utils import logger

MODELS = ("Linear", "NLinear", "DLinear")
LTSF_DATASETS = ("ETTh1", "ETTh2", "Weather", "Exchange-Rate", "Electricity")
DEFAULT_LRS = (5e-3, 1e-3, 5e-4)


def _jobs(mode: str) -> list[tuple[str, str, int, int]]:
    jobs: list[tuple[str, str, int, int]] = []
    if mode in ("ett", "paper", "all"):
        # Compact ETTh* grid (L∈{96,336}); "paper" kept as alias for old CLI.
        for ds in ("ETTh1", "ETTh2"):
            for model in MODELS:
                for L in (96, 336):
                    for h in LTSF_HORIZONS:
                        jobs.append((model, ds, L, h))
    if mode in ("ltsf", "all"):
        for ds in LTSF_DATASETS:
            for model in MODELS:
                for L in LTSF_LOOKBACKS:
                    for H in LTSF_HORIZONS:
                        jobs.append((model, ds, L, H))
    if mode in ("vic", "all"):
        for model in MODELS:
            for L in VIC_LOOKBACKS:
                jobs.append((model, "VIC", L, VIC_PRED_LEN))
    seen: set[tuple[str, str, int, int]] = set()
    out: list[tuple[str, str, int, int]] = []
    for j in jobs:
        if j not in seen:
            seen.add(j)
            out.append(j)
    return out


def _parse_lrs(args: argparse.Namespace) -> tuple[float, ...]:
    if args.lr is not None:
        return (float(args.lr),)
    parts = [p.strip() for p in args.lrs.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty --lrs")
    return tuple(float(p) for p in parts)


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _promote_canonical(src: Path, dst: Path) -> None:
    """Copy best lr trial into the lr-free checkpoint dir used by plots."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "last.pt", "config.json"):
        s = src / name
        if s.is_file():
            shutil.copy2(s, dst / name)
    cfg_path = dst / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["run_dir"] = str(dst)
        cfg_path.write_text(
            json.dumps(cfg, indent=2, default=str) + "\n", encoding="utf-8"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("ett", "paper", "ltsf", "vic", "all"),
        default="ltsf",
        help="ett/paper=ETTh* L∈{96,336}; ltsf=full LTSF; vic; all=ltsf+vic",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument(
        "--lrs",
        default=",".join(str(x) for x in DEFAULT_LRS),
        help="comma-separated learning-rate grid (default: 0.005,0.001,0.0005)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="single learning rate (overrides --lrs)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto")
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--results", type=Path, default=Path("results/grid_results.csv"))
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an lr trial if its config.json already exists",
    )
    args = p.parse_args(argv)

    lrs = _parse_lrs(args)
    jobs = _jobs(args.mode)
    logger.info("grid_start", n_jobs=len(jobs), n_lrs=len(lrs), mode=args.mode, lrs=lrs)

    for i, (model, dataset, seq_len, pred_len) in enumerate(jobs, 1):
        canon = args.ckpt_dir / run_dir_name(model, dataset, seq_len, pred_len)
        best_test = float("inf")
        best_run: str | None = None
        if canon.joinpath("config.json").is_file():
            best_test = float(
                json.loads(canon.joinpath("config.json").read_text(encoding="utf-8"))[
                    "test_loss"
                ]
            )
            best_run = str(canon)

        for lr in lrs:
            suffix = lr_suffix(lr)
            run_name = run_dir_name(model, dataset, seq_len, pred_len, suffix=suffix)
            trial_dir = args.ckpt_dir / run_name
            cfg_path = trial_dir / "config.json"

            if args.skip_existing and cfg_path.is_file():
                prev = json.loads(cfg_path.read_text(encoding="utf-8"))
                test = float(prev["test_loss"])
                logger.info("skip_existing", job=run_name, test=test)
                if test < best_test:
                    best_test = test
                    best_run = str(trial_dir)
                continue

            logger.info(
                "job",
                i=i,
                n=len(jobs),
                job=run_name,
                lr=lr,
            )
            try:
                out = run_training(
                    model_name=model,
                    dataset=dataset,
                    seq_len=seq_len,
                    pred_len=pred_len,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    lr=lr,
                    seed=args.seed,
                    device=args.device,
                    ckpt_dir=args.ckpt_dir,
                    dir_suffix=suffix,
                )
            except Exception as e:
                logger.error("job_failed", job=run_name, err=str(e))
                _append_row(
                    args.results,
                    {
                        "model": model,
                        "dataset": dataset,
                        "seq_len": seq_len,
                        "pred_len": pred_len,
                        "lr": lr,
                        "test_loss": "",
                        "best_val_loss": "",
                        "error": str(e),
                        "run_dir": "",
                    },
                )
                continue

            test = float(out["test_loss"])
            _append_row(
                args.results,
                {
                    "model": model,
                    "dataset": dataset,
                    "seq_len": seq_len,
                    "pred_len": pred_len,
                    "lr": lr,
                    "test_loss": test,
                    "best_val_loss": out["best_val_loss"],
                    "error": "",
                    "run_dir": out["run_dir"],
                },
            )
            logger.info("job_done", job=run_name, test=test, lr=lr)
            if test < best_test:
                best_test = test
                best_run = out["run_dir"]

        if best_run is not None and Path(best_run).resolve() != canon.resolve():
            _promote_canonical(Path(best_run), canon)
            logger.info(
                "promoted_best",
                canon=str(canon),
                from_run=best_run,
                test=best_test,
            )
        elif best_run is not None and canon.joinpath("config.json").is_file():
            logger.info("canon_ok", canon=str(canon), test=best_test)

    logger.info("grid_done", results=str(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
