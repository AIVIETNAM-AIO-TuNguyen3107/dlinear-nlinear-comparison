from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import settings
from src.pipeline.dataset import make_dataloaders
from src.pipeline.models import DLinear, Linear, NLinear
from src.utils import logger

MODEL_REGISTRY = {
    "Linear": Linear,
    "DLinear": DLinear,
    "NLinear": NLinear,
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("cuda requested but not available")
        return torch.device(device)
    raise ValueError(f"invalid device {device!r}; use auto|cpu|cuda")


def he_init_(model: nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


def build_model(
    name: str,
    seq_len: int,
    pred_len: int,
    n_vars: int,
    *,
    individual: bool = False,
    kernel: int = 25,
) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    kwargs: dict[str, Any] = {
        "seq_len": seq_len,
        "pred_len": pred_len,
        "n_vars": n_vars,
        "individual": individual,
    }
    if name == "DLinear":
        kwargs["kernel"] = kernel
    return cls(**kwargs)


def run_dir_name(
    model: str, dataset: str, seq_len: int, pred_len: int, *, suffix: str = ""
) -> str:
    return f"{model}_{dataset}_L{seq_len}_H{pred_len}{suffix}"


def lr_suffix(lr: float) -> str:
    return f"_lr{lr:g}"


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        bs = x.shape[0]
        total += loss.item() * bs
        n += bs
    if n == 0:
        raise ValueError("empty train loader")
    return total / n


@torch.no_grad()
def average_mse(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        bs = x.shape[0]
        total += loss.item() * bs
        n += bs
    if n == 0:
        raise ValueError("empty eval loader")
    return total / n


def run_training(
    *,
    model_name: str,
    dataset: str,
    seq_len: int,
    pred_len: int,
    loaders: dict[str, DataLoader] | None = None,
    epochs: int = 10,
    patience: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 2026,
    individual: bool = False,
    kernel: int = 25,
    device: str = "auto",
    ckpt_dir: Path = Path("checkpoints"),
    data_dir: Path | None = None,
    n_vars: int | None = None,
    dir_suffix: str = "",
) -> dict[str, Any]:
    set_seed(seed)
    dev = resolve_device(device)
    data_dir = Path(data_dir) if data_dir is not None else settings.data_dir
    if loaders is None:
        loaders = make_dataloaders(
            dataset, seq_len, pred_len, batch_size=batch_size, data_dir=data_dir
        )
    train_loader = loaders["train"]
    if len(train_loader) == 0:
        raise ValueError("empty train loader")

    if n_vars is None:
        xb, _ = next(iter(train_loader))
        n_vars = int(xb.shape[1])

    model = build_model(
        model_name,
        seq_len,
        pred_len,
        n_vars,
        individual=individual,
        kernel=kernel,
    )
    he_init_(model)
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    run_dir = Path(ckpt_dir) / run_dir_name(
        model_name, dataset, seq_len, pred_len, suffix=dir_suffix
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model_name,
        "dataset": dataset,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "individual": individual,
        "kernel": kernel if model_name == "DLinear" else None,
        "device": str(dev),
        "n_vars": n_vars,
        "run_dir": str(run_dir),
    }
    write_config(run_dir / "config.json", config)

    best_val = float("inf")
    epochs_without_improve = 0
    epochs_ran = 0
    best_path = run_dir / "best.pt"

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, dev)
        val_loss = average_mse(model, loaders["val"], dev)
        epochs_ran = epoch + 1

        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": min(best_val, val_loss),
            "val_loss": val_loss,
            "model": model_name,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "individual": individual,
            "kernel": kernel if model_name == "DLinear" else None,
            "n_vars": n_vars,
            "seed": seed,
        }
        save_checkpoint(run_dir / "last.pt", payload)

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improve = 0
            payload["best_val_loss"] = best_val
            save_checkpoint(best_path, payload)
        else:
            epochs_without_improve += 1

        logger.info(
            "epoch",
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val=best_val,
        )

        if epochs_without_improve >= patience:
            logger.info("early_stop", epoch=epoch, patience=patience)
            break

    if best_path.is_file():
        ckpt = load_checkpoint(best_path, map_location=dev)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val = float(ckpt["best_val_loss"])

    test_loss = average_mse(model, loaders["test"], dev)
    config["best_val_loss"] = best_val
    config["test_loss"] = test_loss
    config["epochs_ran"] = epochs_ran
    write_config(run_dir / "config.json", config)
    logger.info("done", best_val=best_val, test_loss=test_loss, epochs_ran=epochs_ran)

    return {
        "run_dir": str(run_dir),
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "epochs_ran": epochs_ran,
    }
