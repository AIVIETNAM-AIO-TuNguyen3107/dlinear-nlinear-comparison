from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.config import VIC_PRED_LEN
from src.pipeline.dataset import (
    ForecastWindowDataset,
    Split,
    border_pairs,
    load_forecast_dataset,
    make_dataloaders,
    slice_for_split,
)


def test_window_shapes_and_length():
    # T=10, L=3, H=2 → n = 10 - 3 - 2 + 1 = 6
    data = np.arange(10, dtype=float).reshape(10, 1)
    ds = ForecastWindowDataset(data, seq_len=3, pred_len=2)
    assert len(ds) == 6
    x, y = ds[0]
    assert x.shape == (1, 3)
    assert y.shape == (1, 2)
    assert torch.is_tensor(x) and x.dtype == torch.float32
    assert torch.allclose(x.squeeze(), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(y.squeeze(), torch.tensor([3.0, 4.0]))


def test_ratio_borders_70_15_15():
    assert border_pairs(100, "VIC") == [(0, 70), (70, 85), (85, 100)]


def test_etth_borders_match_informer_months():
    n = 20000
    pairs = border_pairs(n, "ETTh1")
    assert pairs[0] == (0, 12 * 30 * 24)
    assert pairs[1] == (12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24)
    assert pairs[2] == (12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24)


def test_etth_borders_too_short_raises():
    with pytest.raises(ValueError, match="too short"):
        border_pairs(100, "ETTh1")


def test_slice_border_trick():
    series = np.arange(20.0)
    out = slice_for_split(series, (10, 15), seq_len=3)
    assert np.allclose(out, np.arange(7, 15))


def test_load_vic_applies_log(vic_data_dir: Path):
    ds = load_forecast_dataset(
        "VIC",
        seq_len=5,
        pred_len=5,
        split=Split.TRAIN,
        data_dir=vic_data_dir,
        scale=False,
    )
    x, y_win = ds[0]
    assert x.shape == (1, 5)
    assert y_win.shape == (1, 5)
    # unscaled: first lookback is log-close [1, 2, 3, 1, 1]
    assert torch.allclose(x.squeeze()[:3], torch.tensor([1.0, 2.0, 3.0]))


def test_load_ot_column_no_log(weather_data_dir: Path):
    ds = load_forecast_dataset(
        "Weather",
        seq_len=5,
        pred_len=2,
        split=Split.TRAIN,
        data_dir=weather_data_dir,
        scale=False,
    )
    x, y = ds[0]
    assert torch.allclose(x.squeeze(), torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(y.squeeze(), torch.tensor([5.0, 6.0]))


def test_scaler_fits_train_only(vic_data_dir: Path, vic_series: np.ndarray):
    seq_len, pred_len = 5, 2
    train, val, _ = border_pairs(len(vic_series), "VIC")

    ds_train = load_forecast_dataset(
        "VIC",
        seq_len=seq_len,
        pred_len=pred_len,
        split=Split.TRAIN,
        data_dir=vic_data_dir,
        scale=True,
    )
    ds_val = load_forecast_dataset(
        "VIC",
        seq_len=seq_len,
        pred_len=pred_len,
        split=Split.VAL,
        data_dir=vic_data_dir,
        scale=True,
    )

    # scaler fit on train segment of *logged* series only
    scaler = StandardScaler().fit(vic_series[train[0] : train[1]].reshape(-1, 1))

    x_train, _ = ds_train[0]
    expected_train = scaler.transform(vic_series[:seq_len].reshape(-1, 1)).ravel()
    assert np.allclose(x_train.squeeze().numpy(), expected_train, atol=1e-6)

    # val windows use border trick: first x starts at val_start - seq_len
    val_slice = slice_for_split(vic_series, val, seq_len)
    x_val, _ = ds_val[0]
    expected_val = scaler.transform(val_slice[:seq_len].reshape(-1, 1)).ravel()
    assert np.allclose(x_val.squeeze().numpy(), expected_val, atol=1e-6)


def test_too_short_series_raises():
    data = np.arange(5, dtype=float).reshape(5, 1)
    with pytest.raises(ValueError, match="less than"):
        ForecastWindowDataset(data, seq_len=3, pred_len=3)


def test_unknown_dataset_name(tmp_path: Path):
    with pytest.raises((KeyError, ValueError)):
        load_forecast_dataset("nope", 8, 4, Split.TRAIN, data_dir=tmp_path)


def test_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_forecast_dataset("VIC", 5, 5, Split.TRAIN, data_dir=tmp_path)


def test_make_dataloaders_keys(vic_data_dir: Path):
    loaders = make_dataloaders(
        "VIC",
        seq_len=5,
        pred_len=VIC_PRED_LEN,
        batch_size=8,
        data_dir=vic_data_dir,
    )
    assert set(loaders) == {"train", "val", "test"}
    assert isinstance(loaders["train"], DataLoader)
    # Shuffle window *indices* on train only (order inside each (x,y) stays chronological).
    assert isinstance(loaders["train"].sampler, RandomSampler)
    assert isinstance(loaders["val"].sampler, SequentialSampler)
    assert isinstance(loaders["test"].sampler, SequentialSampler)
    xb, yb = next(iter(loaders["train"]))
    assert xb.shape[0] <= 8 and xb.shape[1:] == (1, 5)
    assert yb.shape[1:] == (1, VIC_PRED_LEN)


def test_make_dataloaders_reads_csv_once(vic_data_dir: Path, monkeypatch):
    import src.pipeline.dataset as dataset_mod

    calls = {"n": 0}
    real = dataset_mod.load_series

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(dataset_mod, "load_series", counted)
    make_dataloaders(
        "VIC", seq_len=5, pred_len=VIC_PRED_LEN, batch_size=8, data_dir=vic_data_dir
    )
    assert calls["n"] == 1
