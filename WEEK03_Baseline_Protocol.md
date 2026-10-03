# WEEK03 Baseline Protocol

## API

- `src.pipeline.dataset.load_forecast_dataset(name, seq_len, pred_len, split, ...)`
- `src.pipeline.dataset.make_dataloaders(name, seq_len, pred_len, batch_size=32, ...)`
- Grid constants: `LTSF_LOOKBACKS`, `LTSF_HORIZONS`, `VIC_LOOKBACKS`, `VIC_PRED_LEN`

`split` is `Split.TRAIN` / `Split.VAL` / `Split.TEST`. Specs (file, column, log) come from `settings.get_spec` in `src.config`.

## Datasets (univariate)

| name | file | target | split rule |
|------|------|--------|------------|
| ETTh1 | ETTh1.csv | OT | Informer borders |
| ETTh2 | ETTh2.csv | OT | Informer borders |
| Weather | Weather.csv | OT | 70/15/15 |
| Exchange-Rate | Exchange.csv | OT | 70/15/15 |
| Electricity | Electricity.csv | OT | 70/15/15 |
| VIC | VIC.csv | log(close) | 70/15/15 |

## Splits

- **ETTh\***: train `[0, 12*30*24)`, val next `4*30*24`, test next `4*30*24`.
  Window buffer for a split starts at `max(0, border1 - seq_len)` (border trick).
- **Others / VIC**: `train_end=int(0.7n)`, `val_end=int(0.85n)`.

## Scaling

- `StandardScaler` fit on **train** values only (after VIC log).
- Default: report MSE/MAE on **scaled** space unless noted otherwise.

## Grids

- LTSF: lookback ∈ {96,192,336,720}, horizon ∈ {96,192,336,720}
- VIC: lookback ∈ {5,30,120,480}, pred_len = 5
- Optional time-box cut: LTSF {96,336}×{96,336} — record here if used

## DataLoader shuffle

- `make_dataloaders`: shuffle **train only** (randomize window *indices* for SGD).
- Val/test: no shuffle (sequential). Order inside each `(x, y)` window stays chronological.

## Seed

- Global experiment seed: **2026** (for training; loaders are deterministic given CSVs)
