# WEEK03 Baseline Protocol

## API

- `src.pipeline.dataset.load_forecast_dataset(name, seq_len, pred_len, split, ...)`
- `src.pipeline.dataset.make_dataloaders(name, seq_len, pred_len, batch_size=32, ...)`
- `src.pipeline.train.run_training(...)` — one run; He init, Adam, early stop, checkpoints
- Grid CLI: `src/scripts/run_grid.py` (lookback × lr); plots: `src/scripts/plot_preds.py`
- Protocol constants: `TRAIN_PROTOCOL`, `LTSF_LOOKBACKS`, `LTSF_HORIZONS`, `VIC_LOOKBACKS`, `VIC_PRED_LEN` in `src.config`

`split` is `Split.TRAIN` / `Split.VAL` / `Split.TEST`. Specs (file, column, log) come from `settings.get_spec` in `src.config`.

## Models (univariate, shared weights)

| Model | Idea |
|-------|------|
| Linear | Direct `L → H` linear map |
| NLinear | Subtract last value → Linear → add back |
| DLinear | MA decomp (kernel=25) → dual Linear (seasonal + trend) |

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

## Scaling & metrics

- `StandardScaler` fit on **train** values only (after VIC log).
- Train / select / report **test MSE in scaled space** (primary metric).
- Figures may inverse-transform (VIC → `exp` → close price) for display only.

## Lookback / horizon grids

- **LTSF** (`--mode ltsf` / `all`): lookback ∈ {96,192,336,720}, horizon ∈ {96,192,336,720}
- **VIC** (`--mode vic` / `all`): lookback ∈ {5,30,120,480}, `pred_len = 5`
- **Compact ETTh\*** (`--mode ett`, alias `paper`): lookback ∈ {96,336} × full LTSF horizons

## Learning-rate grid

- Default: `lr ∈ {0.005, 0.001, 0.0005}` per `(model, dataset, L, H)`
- Each trial → `checkpoints/{Model}_{Dataset}_L{L}_H{H}_lr{lr}/`
- **Best test MSE** promoted to canonical `checkpoints/{Model}_{Dataset}_L{L}_H{H}/`
- Report / plots use the **best** cell (over L and lr), not a single fixed lr
- Override: `--lr 0.001` (single) or `--lrs 0.005,0.001`

## Training loop

- Seed **2026**; Adam + MSE; **Kaiming (He)** init on `nn.Linear`
- Defaults (grid): `--epochs 50`, `--patience 10`, batch size 32
- Each epoch: train MSE → val MSE → `last.pt`; improve val → `best.pt`
- Early stop when val does not improve for `patience` epochs (strict `<`)
- Reload `best.pt`, evaluate test MSE → `config.json`

## DataLoader shuffle

- `make_dataloaders`: shuffle **train only** (randomize window *indices* for SGD).
- Val/test: no shuffle (sequential). Order inside each `(x, y)` window stays chronological.

## Figures

- One compare PNG per dataset: Linear / NLinear / DLinear at **best lookback** (for fixed H=96 LTSF, H=5 VIC)
- Full **test** timeline with non-overlapping H-step forecasts → `figures/preds/*_compare_full_timeline.png`
