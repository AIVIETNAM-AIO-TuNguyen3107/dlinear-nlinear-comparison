from pathlib import Path

from src.scripts.download_data import LTSF_TARGETS, ensure_ltsf, is_ready


def test_is_ready_false_missing(tmp_path: Path):
    assert is_ready(tmp_path / "nope.csv") is False


def test_is_ready_false_empty(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    assert is_ready(p) is False


def test_is_ready_true_nonempty(tmp_path: Path):
    p = tmp_path / "ok.csv"
    p.write_text("a,b\n1,2\n")
    assert is_ready(p) is True


def test_ltsf_local_names():
    assert set(LTSF_TARGETS) == {
        "ETTh1.csv",
        "ETTh2.csv",
        "Weather.csv",
        "Exchange.csv",
        "Electricity.csv",
    }


def test_ensure_ltsf_skips_when_present(tmp_path: Path):
    for name in LTSF_TARGETS:
        (tmp_path / name).write_text("x")
    actions = ensure_ltsf(tmp_path, tmp_path / ".cache")
    assert all(a.startswith("skip:") for a in actions)
    assert len(actions) == len(LTSF_TARGETS)
