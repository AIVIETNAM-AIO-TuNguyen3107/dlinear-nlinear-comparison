from enum import Enum
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.config import settings
from src.utils import load_series


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value


class ForecastWindowDataset(Dataset):
    """
    A PyTorch Dataset for generating sliding windows of time series data for forecasting tasks.

    Each sample generated consists of:
        - An input sequence of length `seq_len` (historical data to base forecasts on),
        - A corresponding prediction target of length `pred_len` (the future values to forecast).

    Args:
        data (np.ndarray): The time series data, expected as a 2D array (timesteps x features).
        seq_len (int): The length of the input sequence window.
        pred_len (int): The length of the prediction (target) sequence window.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple (x, y), where
            - x: input tensor of shape (features, seq_len),
            - y: target tensor of shape (features, pred_len).

    Note:
        The dataset transposes the input and target windows so that the returned tensors
        have shape (features, window_length) instead of (window_length, features).
        This convention is useful for Linear Models in PyTorch, such as `torch.nn.Linear`,
        which expect the features to be in the last dimension (for example, input shape [batch, features]).
        By transposing to (features, seq_len), we make it straightforward to apply the same linear transformation along the feature axis, and this format can simplify batch processing and model implementation.

    Example:
        >>> data = np.array([
        ...     [ 0,  1,  2],
        ...     [ 3,  4,  5],
        ...     [ 6,  7,  8],
        ...     [ 9, 10, 11],
        ...     [12, 13, 14],
        ...     [15, 16, 17],
        ...     [18, 19, 20],
        ...     [21, 22, 23],
        ...     [24, 25, 26],
        ...     [27, 28, 29],
        ... ])
        >>> ds = ForecastWindowDataset(data, seq_len=4, pred_len=2)
        >>> x, y = ds[2]
        >>> x
        tensor([[ 6.,  9., 12., 15.],
                [ 7., 10., 13., 16.],
                [ 8., 11., 14., 17.]])
        >>> y
        tensor([[18., 21.],
                [19., 22.],
                [20., 23.]])
    """

    def __init__(self, arr: np.ndarray, seq_len: int, pred_len: int):
        if arr.ndim != 2:
            raise ValueError("Input array must be 2D (timesteps x features)")
        if seq_len <= 0:
            raise ValueError("Sequence length must be positive")
        if pred_len <= 0:
            raise ValueError("Prediction length must be positive")
        if seq_len + pred_len > arr.shape[0]:
            raise ValueError(
                "Sequence length and prediction length must be less than the length of the array"
            )

        self.data = arr
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # (L, C) -> (C, L); (H, C) -> (C, H)
        x = self.data[index : index + self.seq_len].T
        y = self.data[index + self.seq_len : index + self.seq_len + self.pred_len].T
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(
            y.astype(np.float32),
        )


ETH_NAMES = frozenset({"ETTh1", "ETTh2"})


def border_pairs(n: int, name: str) -> list[tuple[int, int]]:
    """
    Returns train, validation, and test index ranges for splitting a time series.

    When training time series models, it's important to split the dataset into
    training, validation, and test sets according to either standardized rules
    (for known datasets) or typical data science conventions (for custom data).
    This function returns the (start, end) index ranges for each set, so you
    can slice your timeseries array for each phase of model training.

    For certain benchmark datasets (like ETTh1 or ETTh2), the splits are
    standardized by the community (e.g., one year for training, 4 months for validation,
    and 8 months for testing). For other datasets, default ratios of 70% train,
    15% val, and 15% test are used.

    Args:
        n (int): Length of the time series array.
        name (str): Dataset name, used to trigger special-case splits.

    Returns:
        list[tuple[int, int]]: List of (start, end) index pairs for each split.
    """
    if name in ETH_NAMES:
        # These values produce "community standard" splits for ETTh1/ETTh2
        train_end = 30 * 24 * 12
        val_end = train_end + 4 * 30 * 24
        test_end = train_end + 8 * 30 * 24
        if n < test_end:
            raise ValueError(f"Array length {n} is too short for these standard splits")
        # Returns [(start, end) for train, val, test]
        return [(0, train_end), (train_end, val_end), (val_end, test_end)]

    # Generic split: 70% train, 15% val, 15% test
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    test_end = n
    return [(0, train_end), (train_end, val_end), (val_end, test_end)]


def slice_for_split(
    serires: np.ndarray, split: tuple[int, int], seq_len: int
) -> np.ndarray:
    start = max(0, split[0] - seq_len)
    end = min(len(serires), split[1])
    return np.asarray(serires[start:end])


def fit_scaler(series_1d: np.ndarray, train: tuple[int, int]) -> StandardScaler:
    """Fit StandardScaler on train segment only (after any log already applied)."""
    y = np.asarray(series_1d, dtype=np.float64).ravel()
    scaler = StandardScaler()
    scaler.fit(y[train[0] : train[1]].reshape(-1, 1))
    return scaler


def _load_raw_series(name: str, data_dir: Path) -> np.ndarray:
    spec = settings.get_spec(name)
    return load_series(
        data_dir / spec["file"],
        column=spec["column"],
        log=bool(spec.get("log", False)),
    )


def _forecast_split_dataset(
    y_full: np.ndarray,
    name: str,
    seq_len: int,
    pred_len: int,
    split: Split,
    *,
    scale: bool = True,
    scaler: StandardScaler | None = None,
) -> ForecastWindowDataset:
    train, val, test = border_pairs(len(y_full), name)
    if split == Split.TRAIN:
        border = train
    elif split == Split.VAL:
        border = val
    elif split == Split.TEST:
        border = test
    else:
        raise ValueError(f"Invalid split: {split}")

    y_split = slice_for_split(y_full, border, seq_len)
    if scale:
        if scaler is None:
            scaler = fit_scaler(y_full, train)
        y_split = scaler.transform(y_split.reshape(-1, 1))
    else:
        y_split = y_split.reshape(-1, 1)
    return ForecastWindowDataset(y_split, seq_len, pred_len)


def load_forecast_dataset(
    name: str,
    seq_len: int,
    pred_len: int,
    split: Split,
    *,
    data_dir: Path = Path("data"),
    scale: bool = True,
) -> ForecastWindowDataset:
    y_full = _load_raw_series(name, data_dir)
    return _forecast_split_dataset(y_full, name, seq_len, pred_len, split, scale=scale)


def make_dataloaders(
    name: str,
    seq_len: int,
    pred_len: int,
    *,
    batch_size: int = 32,
    data_dir: Path = Path("data"),
    scale: bool = True,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Train/val/test DataLoaders; shuffle only on train.

    Reads the CSV and fits the train scaler once, then builds all three splits.
    """
    y_full = _load_raw_series(name, data_dir)
    train, _, _ = border_pairs(len(y_full), name)
    scaler = fit_scaler(y_full, train) if scale else None

    out: dict[str, DataLoader] = {}
    for split in (Split.TRAIN, Split.VAL, Split.TEST):
        ds = _forecast_split_dataset(
            y_full,
            name,
            seq_len,
            pred_len,
            split,
            scale=scale,
            scaler=scaler,
        )
        out[str(split)] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == Split.TRAIN),
            num_workers=num_workers,
        )
    return out
