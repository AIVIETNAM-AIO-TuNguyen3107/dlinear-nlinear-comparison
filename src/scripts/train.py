"""Train one Linear/DLinear/NLinear run; save best/last under checkpoints/."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import settings
from src.pipeline.train import run_training
from src.utils import logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["Linear", "DLinear", "NLinear"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--pred-len", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--individual", action="store_true")
    p.add_argument("--kernel", type=int, default=25)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--data-dir", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # fail fast on unknown dataset
    settings.get_spec(args.dataset)
    out = run_training(
        model_name=args.model,
        dataset=args.dataset,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        individual=args.individual,
        kernel=args.kernel,
        device=args.device,
        ckpt_dir=args.ckpt_dir,
        data_dir=args.data_dir,
    )
    logger.info("finished", **out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
