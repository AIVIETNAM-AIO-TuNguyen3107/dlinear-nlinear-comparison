# DLinear vs NLinear - data-characteristic comparison

**When does Decomposition beat Normalization?**

Research project comparing **DLinear** (decomposition) and **NLinear** (normalization) on long-term time series forecasting (LTSF). We relate which model wins to **data characteristics** (STL trend / seasonal strength, non-stationarity), not only MSE on a single dataset.

Part of AIO Module 4 (AIVIETNAM). Research outline: [`WEEK01_Outline.md`](WEEK01_Outline.md).

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
├── src/
│   ├── models.py         # Linear, NLinear, DLinear
│   ├── data_chars.py     # STL strengths (F_T, F_S), ADF
│   ├── dataset.py        # (W3) loaders
│   ├── eval.py           # (W4) metrics / ablation helpers
│   └── scripts/
│       └── download_data.py
├── test_src/             # pytest
├── notebooks/ / WEEK0*   # weekly notes & experiments
├── pyproject.toml
└── uv.lock
```

## Requirements

- Python **≥ 3.10**
- [uv](https://docs.astral.sh/uv/) (recommended)
- Network once to download datasets (Hugging Face + Yahoo Finance)

## Setup

From `Working_Files/`:

```bash
# create/sync .venv from pyproject.toml + lockfile
uv sync --extra dev

# optional: activate (uv run … works without this)
source .venv/bin/activate
```

**List project packages** (uv venvs often have no `pip`):

```bash
uv pip list
```

## Download data

Idempotent: skips files that already exist and are non-empty.

```bash
uv run python src/scripts/download_data.py
```

- LTSF CSVs ← Hugging Face [`thuml/Time-Series-Library`](https://huggingface.co/datasets/thuml/Time-Series-Library)
- `VIC.csv` ← `yfinance` ticker `VIC.VN` (`period="max"`)
- Cache under `data/.cache/`

Second run should print `skip:…` for every file.

## Run data-character profiling

After data is present:

```bash
uv run python src/data_chars.py
```

Writes / updates profile metrics (see module docstring; expects CSVs in `data/`).

## Tests

```bash
uv run --extra dev pytest
# or, if pytest is already installed in .venv:
uv run pytest
```

## Models (scratch)

Implemented in `src/models.py` (channel-independent, shared weights):

- **Linear** — one linear map lookback → horizon
- **NLinear** — subtract last value, linear, add back
- **DLinear** — moving-average decomp + dual linear heads

Scratch notebook: `WEEK02_Scratch_Models.ipynb`.

## Status (by week)

| Week | Focus | Status |
| ---- | ----- | ------ |
| W1 | Outline, reading notes, papers | Done |
| W2 | Gap note, models, data-char | In progress |
| W3 | Multi-dataset baseline | Planned |
| W4 | Ablation + guideline | Planned |

## Citation / references

Core paper: Zeng et al., *Are Transformers Effective for Time Series Forecasting?* (AAAI 2023) — LTSF-Linear family. Related: Autoformer, RevIN, Toner & Darlow; STL strengths as in Hyndman / FPP3.

Local PDFs and tracker live under the parent `DLinear_NLinear/Paper_Tracker/` folder (see outline).
