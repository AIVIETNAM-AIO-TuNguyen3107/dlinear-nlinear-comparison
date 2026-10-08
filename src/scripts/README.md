# Scripts

Run all commands from `Working_Files/` (repo root for this project).

Shared training logic: `src/pipeline/train.py` (`run_training`).  
Splits / scaling / grid constants: [`WEEK03_Baseline_Protocol.md`](../../TA_reports/WEEK03_Baseline_Protocol.md).

| Script | Role |
| ------ | ---- |
| `download_data.py` | Fetch LTSF + VIC CSVs into `data/` |
| `data_chars.py` | STL / ADF profiles → `TA_reports/WEEK02_Data_Profiles.csv` |
| `train.py` | One model × dataset × `(L, H)` × one `lr` |
| `run_grid.py` | Lookback × horizon × lr sweep; promote best checkpoint |
| `plot_preds.py` | Full-test-timeline compare figures |

Suggested order: download → (optional) data_chars → smoke `train` → `run_grid` → `plot_preds`.

---

## `download_data.py`

Idempotent: skips non-empty files already in `data/`.

```bash
uv run python src/scripts/download_data.py
```

- LTSF CSVs ← Hugging Face [`thuml/Time-Series-Library`](https://huggingface.co/datasets/thuml/Time-Series-Library)
- `VIC.csv` ← `yfinance` ticker `VIC.VN` (`period="max"`)
- Cache under `data/.cache/`

Second run should print `skip:…` for every file.

---

## `data_chars.py`

Requires CSVs in `data/`. Writes / updates profile metrics (see module docstring).

```bash
uv run python src/scripts/data_chars.py
```

---

## Training overview

**One run** = one model × one dataset × one `(seq_len, pred_len)` × one `lr`:

1. Seed (default **2026**), build train/val/test loaders (`make_dataloaders`; scaler fit on train only; MSE in **scaled** space).
2. Build model → **Kaiming (He)** init on every `nn.Linear` → Adam + MSE.
3. Each epoch: train MSE → val MSE → always write `last.pt`; if val improves, write `best.pt`.
4. Early stop when val does not improve for `--patience` epochs (strict `<`).
5. Reload `best.pt`, report test MSE; write/update `config.json`.

### Checkpoint naming

| Kind | Directory under `checkpoints/` | Who writes it |
| ---- | ------------------------------ | ------------- |
| Canonical | `{Model}_{Dataset}_L{seq}_H{pred}/` | `train.py`, or `run_grid` after promoting the best lr |
| LR trial | `{Model}_{Dataset}_L{seq}_H{pred}_lr{lr}/` | `run_grid` only (e.g. `_lr0.001`) |

Each dir has `best.pt`, `last.pt`, `config.json`.  
`.pt` holds weights + optimizer state + rebuild fields (`model`, `seq_len`, `pred_len`, `individual`, `kernel`, `n_vars`, …).

`plot_preds.py` picks the lowest `test_loss` per `(dataset, model, pred_len)` among dirs with `config.json` + `best.pt`. Prefer `run_grid` promotion so the lr-free path is the winner.

---

## `train.py` — one run

Smoke-test, debug one cell, or retrain one setting without a sweep.

```bash
uv run python src/scripts/train.py \
  --model DLinear --dataset ETTh1 \
  --seq-len 96 --pred-len 96 \
  --epochs 10 --patience 3 --lr 1e-3 --seed 2026
```

| Flag | Default | Notes |
| ---- | ------- | ----- |
| `--model` | required | `Linear` \| `NLinear` \| `DLinear` |
| `--dataset` | required | e.g. `ETTh1`, `Weather`, `VIC` (must exist in `settings`) |
| `--seq-len` / `--pred-len` | required | lookback / horizon |
| `--epochs` / `--patience` | `10` / `3` | early stop on val MSE |
| `--lr` | `1e-3` | single learning rate |
| `--batch-size` | `32` | |
| `--seed` | `2026` | |
| `--individual` | off | per-channel Linear heads |
| `--kernel` | `25` | DLinear moving-average kernel |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `--ckpt-dir` / `--data-dir` | `checkpoints` / config default | |

Writes only the **canonical** dir (no `_lr…` suffix). Re-runs overwrite that cell.

---

## `run_grid.py` — lookback × lr grid

Orchestrator around the same `run_training`. Goal: **best test MSE per `(model, dataset, L, H)`**, not paper-table replication.

For each job `(model, dataset, seq_len, pred_len)`:

1. Optionally seed `best_test` from an existing **canonical** `config.json` (resume does not demote a known good run).
2. For each `lr` in the lr grid:
   - Train into `…_lr{lr}/` (unless `--skip-existing` and that trial’s `config.json` exists).
   - Append a row to `results/grid_results.csv`.
   - Track the trial with the lowest `test_loss`.
3. If the best trial is not already the canonical dir, **promote**: copy `best.pt`, `last.pt`, `config.json` into the lr-free path and rewrite `config.run_dir`.

Failed trials are logged and written with empty metrics + `error`; the loop continues.

### Modes (`--mode`)

Constants from `src/pipeline/dataset.py`:  
`LTSF_LOOKBACKS` / `LTSF_HORIZONS` / `VIC_*` and `TRAIN_PROTOCOL` live in `src.config`.  
Models always: `Linear`, `NLinear`, `DLinear`.

| Mode | Cells (before × lr) | What expands |
| ---- | ------------------- | ------------ |
| `ett` / `paper` | 48 | ETTh1+ETTh2 × 3 models × L∈{96,336} × H∈{96,192,336,720} (`paper` = old CLI alias) |
| `ltsf` (default) | 240 | 5 LTSF datasets × 3 models × 4L × 4H |
| `vic` | 12 | 3 models × VIC L∈{5,30,120,480}, H=5 |
| `all` | 252 | `ltsf` + `vic` (deduped) |

LTSF datasets: `ETTh1`, `ETTh2`, `Weather`, `Exchange-Rate`, `Electricity`.

Default lr grid: `5e-3,1e-3,5e-4` → e.g. `ltsf` ≈ **720** trainings. Use `--lr 0.001` for a single lr (overrides `--lrs`).

### CLI

```bash
# full LTSF lookback×horizon×lr (long)
uv run python src/scripts/run_grid.py --mode ltsf --epochs 50 --patience 10

# compact ETTh* first
uv run python src/scripts/run_grid.py --mode ett --epochs 50 --patience 10

# VIC only
uv run python src/scripts/run_grid.py --mode vic --epochs 50 --patience 10

# custom lr list, or one lr; resume without redoing finished trials
uv run python src/scripts/run_grid.py --mode ltsf --lrs 0.005,0.001,0.0005 --skip-existing
uv run python src/scripts/run_grid.py --mode ett --lr 0.001 --skip-existing
```

Outputs:

- `checkpoints/..._lr0.001/` — trial runs
- `checkpoints/..._L*_H*/` — promoted best per cell
- `results/grid_results.csv` — one row per trial (`model`, `dataset`, `seq_len`, `pred_len`, `lr`, `test_loss`, `best_val_loss`, `error`, `run_dir`)

| Flag | Default | Notes |
| ---- | ------- | ----- |
| `--mode` | `ltsf` | `ett` \| `paper` \| `ltsf` \| `vic` \| `all` |
| `--epochs` / `--patience` | `50` / `10` | stricter than `train.py` defaults |
| `--lrs` | `0.005,0.001,0.0005` | comma-separated |
| `--lr` | unset | if set, only that lr |
| `--batch-size` / `--seed` / `--device` | `32` / `2026` / `auto` | passed through to `run_training` |
| `--ckpt-dir` | `checkpoints` | |
| `--results` | `results/grid_results.csv` | append-only CSV |
| `--skip-existing` | off | skip trial if `…_lr*/config.json` exists; still use its `test_loss` for promotion |

Not on the grid CLI (always `run_training` defaults): `--individual` (False), `--kernel` (25), `--data-dir`. Use `train.py` for those.

### `train` + `run_grid` together

| Goal | Use |
| ---- | --- |
| Sanity-check one cell / new dataset | `train.py` (small `--epochs`) |
| Sweep lookbacks / horizons / lrs | `run_grid.py` |
| Fix one bad cell after a grid | `train.py` with the winning `--lr`, **or** `run_grid --mode … --lr … --skip-existing` after deleting that trial dir |
| Feed `plot_preds.py` | Prefer grid promotion (canonical dirs); or train once into the canonical path |

Workflow:

1. `download_data.py` → optional `data_chars.py`
2. Smoke: `train.py` on e.g. ETTh1 L96 H96, few epochs
3. Compact grid: `run_grid.py --mode ett` (or `--mode vic`)
4. Full baseline: `run_grid.py --mode ltsf` or `--mode all`; use `--skip-existing` on restarts
5. `plot_preds.py`

Do **not** overwrite a canonical dir with `train.py` during an unfinished lr sweep unless you intend that run to seed `best_test` for promotion (grid reads canonical `test_loss` at the start of each cell).

---

## `plot_preds.py`

Needs trained checkpoints under `checkpoints/`. Writes compare PNGs to `figures/preds/` (clears previous `*.png` there first). Horizon used for selection: **96** (LTSF) or **5** (VIC); best lookback per model by test MSE.

```bash
uv run python src/scripts/plot_preds.py
```
