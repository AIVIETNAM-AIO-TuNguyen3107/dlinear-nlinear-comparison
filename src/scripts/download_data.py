"""Ensure Working_Files/data/ has LTSF + VIC CSVs; skip if already present.

LTSF files come from Hugging Face thuml/Time-Series-Library (Autoformer-preprocessed).
VIC (VIC.VN / VinGroup close) comes from yfinance.
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / ".cache"

# local name -> remote path under HF dataset root
HF_BASE = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"
LTSF_TARGETS: dict[str, str] = {
    "ETTh1.csv": "ETT-small/ETTh1.csv",
    "ETTh2.csv": "ETT-small/ETTh2.csv",
    "Weather.csv": "weather/weather.csv",
    "Exchange.csv": "exchange_rate/exchange_rate.csv",
    "Electricity.csv": "electricity/electricity.csv",
}

VIC_TICKER = "VIC.VN"


def is_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_url(url: str, dest: Path) -> None:
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def ensure_ltsf(data_dir: Path, cache_dir: Path) -> list[str]:
    """Download missing LTSF CSVs into data_dir. Returns action strings."""
    ensure_dir(data_dir)
    ensure_dir(cache_dir)
    actions: list[str] = []
    for local_name, remote_path in LTSF_TARGETS.items():
        dest = data_dir / local_name
        if is_ready(dest):
            actions.append(f"skip:{local_name}")
            continue
        cache_file = cache_dir / remote_path.replace("/", "__")
        if not is_ready(cache_file):
            url = f"{HF_BASE}/{remote_path}"
            print(f"download {local_name} <- {url}")
            download_url(url, cache_file)
        shutil.copy2(cache_file, dest)
        actions.append(f"downloaded:{local_name}")
    return actions


def ensure_vic(data_dir: Path) -> str:
    """Fetch VIC.VN daily close into VIC.csv if missing."""
    ensure_dir(data_dir)
    dest = data_dir / "VIC.csv"
    if is_ready(dest):
        return "skip:VIC.csv"

    import pandas as pd
    import yfinance as yf  # lazy: only needed when VIC missing

    print(f"download VIC.csv <- yfinance {VIC_TICKER}")
    df = yf.download(VIC_TICKER, period="max", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned empty data for {VIC_TICKER}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    out = df.reset_index()
    date_col = "Date" if "Date" in out.columns else out.columns[0]
    close_col = "Close" if "Close" in out.columns else "close"
    slim = out[[date_col, close_col]].rename(
        columns={date_col: "date", close_col: "close"}
    )
    slim.to_csv(dest, index=False)
    return "downloaded:VIC.csv"


def main() -> int:
    ensure_dir(DATA_DIR)
    actions = ensure_ltsf(DATA_DIR, CACHE_DIR)
    try:
        actions.append(ensure_vic(DATA_DIR))
    except Exception as exc:  # noqa: BLE001 — report and fail exit
        print(f"failed:VIC.csv ({exc})", file=sys.stderr)
        actions.append("failed:VIC.csv")

    for a in actions:
        print(a)

    expected = list(LTSF_TARGETS) + ["VIC.csv"]
    missing = [n for n in expected if not is_ready(DATA_DIR / n)]
    if missing:
        print(f"missing after run: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
