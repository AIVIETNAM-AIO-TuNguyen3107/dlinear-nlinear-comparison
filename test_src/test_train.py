from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.scripts.train import build_parser

from src.pipeline.train import (
    average_mse,
    build_model,
    he_init_,
    load_checkpoint,
    resolve_device,
    run_dir_name,
    run_training,
    save_checkpoint,
    set_seed,
    write_config,
)


def _toy_loaders(L=8, H=4, C=1, n=16, batch=4):
    x = torch.randn(n, C, L)
    y = torch.randn(n, C, H)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    return {"train": loader, "val": loader, "test": loader}


def test_resolve_device_auto_and_cpu():
    d = resolve_device("cpu")
    assert d.type == "cpu"
    d2 = resolve_device("auto")
    assert d2.type in ("cpu", "cuda")


def test_resolve_device_invalid():
    try:
        resolve_device("tpu")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "device" in str(e).lower() or "tpu" in str(e).lower()


def test_build_model_shapes():
    for name in ("Linear", "DLinear", "NLinear"):
        m = build_model(name, seq_len=8, pred_len=4, n_vars=1, individual=False, kernel=3)
        y = m(torch.randn(2, 1, 8))
        assert y.shape == (2, 1, 4)


def test_build_model_unknown():
    try:
        build_model("Foo", 8, 4, 1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_he_init_linear_weights_finite_nonzero():
    m = build_model("Linear", seq_len=8, pred_len=4, n_vars=1)
    # zero then re-init so we know he_init_ did something
    for p in m.parameters():
        nn.init.zeros_(p)
    he_init_(m)
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            assert torch.isfinite(mod.weight).all()
            assert not torch.all(mod.weight == 0)
            assert torch.all(mod.bias == 0)


def test_set_seed_reproducible():
    set_seed(2026)
    a = torch.randn(3)
    set_seed(2026)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_run_dir_name():
    assert run_dir_name("DLinear", "ETTh1", 96, 96) == "DLinear_ETTh1_L96_H96"
    assert (
        run_dir_name("DLinear", "ETTh1", 96, 96, suffix="_lr0.001")
        == "DLinear_ETTh1_L96_H96_lr0.001"
    )


def test_save_load_checkpoint_roundtrip(tmp_path: Path):
    m = build_model("NLinear", 8, 4, 1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    path = tmp_path / "last.pt"
    payload = {
        "model_state_dict": m.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "epoch": 0,
        "best_val_loss": 1.23,
        "val_loss": 1.23,
        "model": "NLinear",
        "seq_len": 8,
        "pred_len": 4,
        "individual": False,
        "kernel": None,
        "n_vars": 1,
        "seed": 2026,
    }
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path)
    m2 = build_model("NLinear", 8, 4, 1)
    m2.load_state_dict(loaded["model_state_dict"])
    assert loaded["epoch"] == 0
    assert loaded["best_val_loss"] == 1.23


def test_write_config(tmp_path: Path):
    p = tmp_path / "config.json"
    write_config(p, {"model": "Linear", "seq_len": 8})
    text = p.read_text(encoding="utf-8")
    assert "Linear" in text


def test_average_mse_finite():
    loaders = _toy_loaders()
    m = build_model("Linear", 8, 4, 1)
    he_init_(m)
    loss = average_mse(m, loaders["val"], torch.device("cpu"))
    assert loss >= 0 and np.isfinite(loss)


def test_run_training_writes_checkpoints(tmp_path: Path):
    loaders = _toy_loaders()
    out = run_training(
        model_name="Linear",
        dataset="Toy",
        seq_len=8,
        pred_len=4,
        loaders=loaders,
        epochs=2,
        patience=5,
        lr=1e-2,
        seed=2026,
        ckpt_dir=tmp_path,
        n_vars=1,
    )
    run_dir = Path(out["run_dir"])
    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "config.json").is_file()
    ckpt = load_checkpoint(run_dir / "best.pt")
    m = build_model("Linear", 8, 4, 1)
    m.load_state_dict(ckpt["model_state_dict"])


def test_run_training_early_stop(tmp_path: Path, monkeypatch):
    from src.pipeline import train as train_mod

    loaders = _toy_loaders()
    monkeypatch.setattr(train_mod, "average_mse", lambda *a, **k: 1.0)
    out = run_training(
        model_name="Linear",
        dataset="Toy",
        seq_len=8,
        pred_len=4,
        loaders=loaders,
        epochs=10,
        patience=1,
        seed=2026,
        ckpt_dir=tmp_path,
        n_vars=1,
    )
    # epoch 0: best=1.0; epoch 1: val=1.0 not < best → patience hit → stop
    assert out["epochs_ran"] == 2


def test_cli_parser_defaults():
    p = build_parser()
    args = p.parse_args(["--model", "DLinear", "--dataset", "ETTh1", "--seq-len", "96", "--pred-len", "96"])
    assert args.epochs == 10
    assert args.patience == 3
    assert args.seed == 2026
    assert args.lr == 1e-3


def test_run_training_empty_loader(tmp_path: Path):
    empty = DataLoader(TensorDataset(torch.empty(0, 1, 8), torch.empty(0, 1, 4)), batch_size=4)
    try:
        run_training(
            model_name="Linear",
            dataset="Toy",
            seq_len=8,
            pred_len=4,
            loaders={"train": empty, "val": empty, "test": empty},
            epochs=1,
            ckpt_dir=tmp_path,
            n_vars=1,
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower() or "train" in str(e).lower()
