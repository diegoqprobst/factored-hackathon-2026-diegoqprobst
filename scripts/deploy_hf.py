"""Deploy to Hugging Face: private dataset (sandbox.db) + public Docker Space (app).
Usage: uv run --group embeddings --env-file .env python scripts/deploy_hf.py --user <hf_user>"""
import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--space", default="dispute-agent")
    ap.add_argument("--dataset", default="latam-bank-sandbox")
    a = ap.parse_args()
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    dataset, space = f"{a.user}/{a.dataset}", f"{a.user}/{a.space}"

    api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
    db_path = ROOT / "data" / "sandbox.db"
    local = sha256(db_path)
    remote = next((f.lfs.sha256 for f in api.list_repo_tree(dataset, repo_type="dataset")
                   if getattr(f, "path", "") == "sandbox.db" and getattr(f, "lfs", None)), None)
    if remote != local:
        api.upload_file(path_or_fileobj=str(db_path), path_in_repo="sandbox.db", repo_id=dataset, repo_type="dataset",
                        commit_message="sandbox.db (synthetic, organizer dataset derivative; private)")
    print("dataset:", dataset, "(private)", "uploaded" if remote != local else "unchanged")

    api.create_repo(space, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    for key, value in {"BANK_SESSION_SECRET": os.environ["BANK_SESSION_SECRET"],
                       "OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"], "HF_DATA_TOKEN": token}.items():
        api.add_space_secret(space, key, value)
    for key, value in {"AGENT_MODE": "hybrid", "DEMO_MODE": "1", "AGENT_LLM_DAILY_BUDGET_USD": "0.5",
                       "HF_DATASET": dataset}.items():
        api.add_space_variable(space, key, value)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for f in (ROOT / "deploy" / "hf_space").iterdir():
            shutil.copy(f, stage / f.name)
        for d in ("src", "policy", "models"):
            shutil.copytree(ROOT / d, stage / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for f in ("pyproject.toml", "uv.lock"):
            shutil.copy(ROOT / f, stage / f)
        api.upload_folder(folder_path=str(stage), repo_id=space, repo_type="space", commit_message="deploy app")
    print("space: https://huggingface.co/spaces/" + space)
    print("app:   https://" + space.replace("/", "-").replace("_", "-").lower() + ".hf.space")


if __name__ == "__main__":
    main()
