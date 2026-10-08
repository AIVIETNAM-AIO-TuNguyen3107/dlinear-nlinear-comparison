from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings

# Baseline protocol defaults (see TA_reports/WEEK03_Baseline_Protocol.md)
TRAIN_PROTOCOL = {
    "epochs": 50,
    "patience": 10,
    "batch_size": 32,
    "seed": 2026,
    "lrs": (5e-3, 1e-3, 5e-4),
}

LTSF_LOOKBACKS = (96, 192, 336, 720)
LTSF_HORIZONS = (96, 192, 336, 720)
VIC_LOOKBACKS = (5, 30, 120, 480)
VIC_PRED_LEN = 5


class Config(BaseSettings):
    # period: ETTh hourly → 24; daily series → 7 (document in notes; use 365 only if long enough)
    DATASET_SPECS: list[dict[str, Any]] = [
        {
            "name": "ETTh1",
            "file": "ETTh1.csv",
            "column": "OT",
            "period": 24,
            "notes": "hourly; STL period=24; target OT",
        },
        {
            "name": "ETTh2",
            "file": "ETTh2.csv",
            "column": "OT",
            "period": 24,
            "notes": "hourly; STL period=24; target OT",
        },
        {
            "name": "Weather",
            "file": "Weather.csv",
            "column": "OT",
            "period": 24,
            "notes": "Weather LTSF; STL period=24; target OT",
        },
        {
            "name": "Exchange-Rate",
            "file": "Exchange.csv",
            "column": "OT",
            "period": 7,
            "notes": "daily FX; STL period=7; target OT",
        },
        {
            "name": "Electricity",
            "file": "Electricity.csv",
            "column": "OT",
            "period": 24,
            "notes": "hourly electricity; STL period=24; target OT",
        },
        {
            "name": "VIC",
            "file": "VIC.csv",
            "column": "close",
            "period": 7,
            "log": True,
            "notes": "VNM Vingroup daily close; log(close); STL period=7",
        },
    ]

    root_dir: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = root_dir / "data"
    cache_dir: Path = data_dir / ".cache"
    profiles_csv: Path = root_dir / "TA_reports" / "WEEK02_Data_Profiles.csv"
    grid_result_csv: Path = root_dir / "results" / "grid_results.csv"
    baseline_results_csv: Path = root_dir / "TA_reports" / "WEEK03_Results.csv"
    HF_BASE: str = (
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"
    )
    LTSF_TARGETS: dict[str, str] = {
        "ETTh1.csv": "ETT-small/ETTh1.csv",
        "ETTh2.csv": "ETT-small/ETTh2.csv",
        "Weather.csv": "weather/weather.csv",
        "Exchange.csv": "exchange_rate/exchange_rate.csv",
        "Electricity.csv": "electricity/electricity.csv",
    }

    VIC_TICKER: str = "VIC.VN"

    def get_spec(self, name: str) -> dict[str, Any]:
        for spec in self.DATASET_SPECS:
            if spec["name"] == name:
                return spec
        known = [s["name"] for s in self.DATASET_SPECS]
        raise KeyError(f"unknown dataset {name!r}; known: {known}")


settings = Config()
