# DLinear vs NLinear - data-characteristic comparison

**When does Decomposition beat Normalization?**

Research project comparing **DLinear** (decomposition) and **NLinear** (normalization) on long-term time series forecasting (LTSF). We relate which model wins to **data characteristics** (STL trend / seasonal strength, non-stationarity), not only MSE on a single dataset.

Part of AIO Module 4 (AIVIETNAM). Research outline: [`TA_reports/WEEK01_Outline.md`](TA_reports/WEEK01_Outline.md).

## Research questions

| ID | Question |
| -- | -------- |
| **RQ1** | Which data traits (trend / season / non-stationarity) predict DLinear vs NLinear winning? |
| **RQ2** | How does lookback sensitivity differ, and does that depend on data-char? |

**Hypothesis (falsifiable):** DLinear tends to win when seasonal/trend strength is high; NLinear when level/distribution shift is strong (e.g. Exchange, VIC).

## Datasets

| Name | File under `data/` | Notes |
| ---- | ------------------ | ----- |
| ETTh1 / ETTh2 | `ETTh1.csv`, `ETTh2.csv` | Hourly energy |
| Weather | `Weather.csv` | LTSF weather |
| Exchange-Rate | `Exchange.csv` | Daily FX |
| Electricity | `Electricity.csv` | Hourly load |
| VIC | `VIC.csv` | VinGroup (`VIC.VN`) daily close; use `log(close)` in profiling |

## Repo layout

```
Working_Files/
├── data/                 # CSVs (gitignored; download script fills this)
├── checkpoints/          # best.pt / last.pt / config.json (gitignored)
├── figures/preds/        # real vs pred plots
├── results/              # grid / refine CSVs
├── TA_reports/           # weekly TA writeups (outline, notes, LaTeX, protocol)
├── src/
│   ├── config.py             # paths, dataset specs, download targets
│   ├── pipeline/             # models, dataset, train, eval
│   ├── utils/
│   └── scripts/              # CLIs — see src/scripts/README.md
├── test_src/
├── pyproject.toml
└── uv.lock
```

## Setup

Requirements: Python **≥ 3.10**, [uv](https://docs.astral.sh/uv/) recommended. Network once for datasets (Hugging Face + Yahoo Finance).

```bash
uv sync --extra dev
# optional: source .venv/bin/activate
uv pip list   # uv venvs often have no pip
```

## How to run

**All script usage** (download, data-chars, train, grid, plots): [`src/scripts/README.md`](src/scripts/README.md).

Baseline protocol (splits, scaling, grids): [`TA_reports/WEEK03_Baseline_Protocol.md`](TA_reports/WEEK03_Baseline_Protocol.md).

```bash
uv run python src/scripts/download_data.py
uv run python src/scripts/run_grid.py --mode ett --epochs 50 --patience 10
uv run python src/scripts/plot_preds.py
```

## Models

In `src/pipeline/models.py` (channel-independent, shared weights):

- **Linear** — one linear map lookback → horizon
- **NLinear** — subtract last value, linear, add back
- **DLinear** — moving-average decomp + dual linear heads

Scratch notebook: [`TA_reports/WEEK02_Scratch_Models.ipynb`](TA_reports/WEEK02_Scratch_Models.ipynb).

## Tests

```bash
uv run --extra dev pytest
```

## Status (by week)

| Week | Focus | Status |
| ---- | ----- | ------ |
| W1 | Outline, reading notes, papers | Done |
| W2 | Gap note, models, data-char | Done |
| W3 | Multi-dataset baseline + train loop | In progress |
| W4 | Ablation + guideline | Planned |

## Citation / references

Core paper: Zeng et al., *Are Transformers Effective for Time Series Forecasting?* (AAAI 2023) — LTSF-Linear family. Related: Autoformer, RevIN, Toner & Darlow; STL strengths as in Hyndman / FPP3.

Local PDFs and tracker live under the parent `DLinear_NLinear/Paper_Tracker/` folder (see outline).
