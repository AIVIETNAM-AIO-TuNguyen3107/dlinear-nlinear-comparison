"""Ensure Working_Files/data/ has LTSF + VIC CSVs; skip if already present.

LTSF files come from Hugging Face thuml/Time-Series-Library (Autoformer-preprocessed).
VIC (VIC.VN / VinGroup close) comes from yfinance.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from src.config import settings
from src.utils import logger


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
    for local_name, remote_path in settings.LTSF_TARGETS.items():
        dest = data_dir / local_name
        if is_ready(dest):
            logger.info("skip", local_name=local_name)
            actions.append(f"skip:{local_name}")
            continue
        cache_file = cache_dir / remote_path.replace("/", "__")
        if not is_ready(cache_file):
            url = f"{settings.HF_BASE}/{remote_path}"
            logger.info(
                "download_start", dest=str(dest), local_name=local_name, url=url
            )
            download_url(url, cache_file)
        shutil.copy2(cache_file, dest)
        logger.info("download_success", dest=str(dest), local_name=local_name)
        actions.append(f"downloaded:{local_name}")
    return actions


def ensure_vic(data_dir: Path) -> str:
    """Fetch VIC.VN daily close into VIC.csv if missing."""
    ensure_dir(data_dir)
    dest = data_dir / "VIC.csv"
    if is_ready(dest):
        logger.info("skip", local_name="VIC.csv")
        return "skip:VIC.csv"
    import pandas as pd
    import yfinance as yf  # lazy: only needed when VIC missing

    logger.info("download_start", ticker=settings.VIC_TICKER)
    df = yf.download(
        settings.VIC_TICKER, period="max", progress=False, auto_adjust=True
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned empty data for {settings.VIC_TICKER}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    out = df.reset_index()
    date_col = "Date" if "Date" in out.columns else out.columns[0]
    close_col = "Close" if "Close" in out.columns else "close"
    slim = out[[date_col, close_col]].rename(
        columns={date_col: "date", close_col: "close"}
    )
    slim.to_csv(dest, index=False)
    logger.info("download_success", ticker=settings.VIC_TICKER)
    return "downloaded:VIC.csv"


def download_data(
    data_dir: Path | str = settings.data_dir, cache_dir: Path | str = settings.cache_dir
) -> list[str]:
    ensure_dir(data_dir)
    ensure_dir(cache_dir)
    actions = ensure_ltsf(Path(data_dir), Path(cache_dir))
    actions.append(ensure_vic(Path(data_dir)))
    return actions


if __name__ == "__main__":
    download_data()
