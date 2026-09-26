import json
import sys

from src.pipeline import run
from tests.helpers import write_raw


def test_run_all_end_to_end(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    write_raw(raw)
    monkeypatch.setattr(run.config, "RAW_DIR", raw)
    monkeypatch.setattr(run.config, "SILVER_DIR", tmp_path / "silver")
    monkeypatch.setattr(run.config, "SANDBOX_PATH", tmp_path / "s.db")
    monkeypatch.setattr(run, "QUALITY_REPORT", tmp_path / "quality.json")
    monkeypatch.setattr(sys, "argv", ["run", "all"])
    run.main()
    assert json.loads((tmp_path / "quality.json").read_text())["tables"]["customers"]["rows_out"] == 2
    assert (tmp_path / "s.db").exists()
