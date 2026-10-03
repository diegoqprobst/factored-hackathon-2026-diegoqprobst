"""Train and save the TF-IDF router (the documented, torch-free fallback) with the exact data, split and
threshold procedure of `src.router.evaluate`, for deployments that cannot carry the e5 model.
Usage: uv run python scripts/train_tfidf_router.py"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.router.classifier import tfidf_router  # noqa: E402
from src.router.dataset import dataset_hash, decontaminate, load_examples, split_train_val  # noqa: E402
from src.router.evaluate import evaluate_router, tune_threshold  # noqa: E402

OUT = Path("models/router_tfidf_v1")


def main():
    trainval, test = load_examples()
    trainval, dropped = decontaminate(trainval, test)
    train, val = split_train_val(trainval)
    router = tfidf_router().fit(train)
    preds = router.predict_many([e.text for e in val])
    router.threshold = tune_threshold([p.confidence for p in preds], [p.intent == e.intent for p, e in zip(preds, val)])
    val_m, test_m = evaluate_router(router, val), evaluate_router(router, test)
    meta = {"model": "tfidf", "version": router.version, "threshold": router.threshold,
            "purpose": "torch-free fallback for small deployments (not the pre-registered selection)",
            "dataset_hash": dataset_hash(trainval + test), "decontaminated": len(dropped),
            "n_train": len(train), "n_val": len(val), "n_test": len(test), "val": val_m, "test": test_m,
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "created_at": datetime.now(timezone.utc).isoformat()}
    router.save(OUT, meta)
    print(json.dumps({"threshold": round(router.threshold, 3), "val_macro_f1": round(val_m["macro_f1"], 4),
                      "test_macro_f1": round(test_m["macro_f1"], 4), "test_coverage": round(test_m["coverage"], 4)}))


if __name__ == "__main__":
    main()
