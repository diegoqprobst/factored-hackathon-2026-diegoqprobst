"""Pipeline entrypoint: uv run python -m src.pipeline.run {silver|sandbox|all}

The sandbox is a generated artifact: `sandbox` rebuilds it from scratch (disputes/blocks are reset)."""
import argparse
import json
from pathlib import Path

from src.bank import config
from src.pipeline.sandbox import build_sandbox
from src.pipeline.silver import build_silver

QUALITY_REPORT = Path("reports/data_quality_latest.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["silver", "sandbox", "all"])
    step = ap.parse_args().step
    if step in ("silver", "all"):
        report = build_silver(config.RAW_DIR, config.SILVER_DIR)
        QUALITY_REPORT.parent.mkdir(parents=True, exist_ok=True)
        QUALITY_REPORT.write_text(json.dumps(report, indent=2))
        summary = {t: {k: v for k, v in s.items() if not k.endswith("reasons")} for t, s in report["tables"].items()}
        print(json.dumps(summary, indent=2))
    if step in ("sandbox", "all"):
        Path(config.SANDBOX_PATH).unlink(missing_ok=True)
        print(build_sandbox(config.SILVER_DIR, config.SANDBOX_PATH))


if __name__ == "__main__":
    main()
