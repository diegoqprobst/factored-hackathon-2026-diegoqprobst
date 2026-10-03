"""Router evaluation: pre-registered selection on validation, threshold tuned on validation, every router scored
once on the sealed test set. Usage: uv run --group embeddings python -m src.router.evaluate"""
import csv
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.router.classifier import e5_router, tfidf_router
from src.router.dataset import dataset_hash, decontaminate, leakage_report, load_examples, split_train_val
from src.router.keyword import KeywordRouter
from src.router.labels import DATA_DIR, INTENTS, Example, RouterResult

SELECTION_RULE = ("Pre-registered: pick the router with the highest validation macro-F1; ties prefer the simpler "
                  "model (keyword > tfidf > e5); if e5 wins by less than 0.02 over tfidf, pick tfidf. The abstention "
                  "threshold is tuned on validation (selective precision >= 0.95). The test set is scored once, "
                  "after selection, for every router.")
SIMPLICITY = {"keyword": 2, "tfidf": 1, "e5": 0}
SPEC_DEVIATIONS = ("**Deviation from spec.** Spec §6 says models are selected by macro-F1 *and calibration*; the "
                   "pre-registered rule above (fixed before any test score existed) uses macro-F1 only, and it is kept "
                   "as registered rather than changed after seeing test results. Calibration (ECE) is reported, not "
                   "used for selection. Router confidences are therefore treated as scores thresholded on validation, "
                   "not as probabilities.")


def ece(conf, correct, bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(conf[idx == b].mean() - correct[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def tune_threshold(conf, correct, target: float = 0.95) -> float:
    pairs = list(zip(conf, correct))
    for t in sorted(set(conf)):
        accepted = [c for s, c in pairs if s >= t]
        if accepted and sum(accepted) / len(accepted) >= target:
            return float(t)
    return float(max(conf)) + 1e-9 if conf else 1.0


def select_model(val_f1: dict[str, float]) -> str:
    best = max(val_f1, key=lambda n: (val_f1[n], SIMPLICITY[n]))
    if best == "e5" and "tfidf" in val_f1 and val_f1["e5"] - val_f1["tfidf"] < 0.02:
        return "tfidf"
    return best


def _macro_f1(y, p) -> float:
    return float(f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0)) if y else 0.0


def metrics(examples: list[Example], preds: list[RouterResult]) -> dict:
    y, p = [e.intent for e in examples], [r.intent for r in preds]
    correct = [a == b for a, b in zip(y, p)]
    accepted = [not r.abstain for r in preds]
    labels = sorted(set(y), key=lambda l: (INTENTS.index(l) if l in INTENTS else len(INTENTS), l))
    prec, rec, f1, _ = precision_recall_fscore_support([e.injection for e in examples], [r.injection for r in preds],
                                                       average="binary", zero_division=0)
    return {
        "n": len(examples),
        "macro_f1": _macro_f1(y, p),
        "accuracy": float(np.mean(correct)),
        "per_class_f1": dict(zip(labels, f1_score(y, p, labels=labels, average=None, zero_division=0).tolist())),
        "per_language_macro_f1": {lang: _macro_f1([a for a, e in zip(y, examples) if e.language == lang],
                                                  [b for b, e in zip(p, examples) if e.language == lang])
                                  for lang in sorted({e.language for e in examples})},
        "n_by_language": dict(Counter(e.language for e in examples)),
        "coverage": float(np.mean(accepted)),
        "selective_accuracy": (sum(c for c, a in zip(correct, accepted) if a) / sum(accepted)) if any(accepted) else None,
        "ece": ece([r.confidence for r in preds], correct),
        "language_accuracy": float(np.mean([e.language == r.language for e, r in zip(examples, preds)])),
        "injection": {"precision": float(prec), "recall": float(rec), "f1": float(f1)},
    }


def evaluate_router(router, examples: list[Example]) -> dict:
    preds = router.predict_many([e.text for e in examples])
    latencies = []
    for e in examples:
        start = time.perf_counter()
        router.predict(e.text)
        latencies.append((time.perf_counter() - start) * 1000)
    out = metrics(examples, preds)
    out["threshold"] = float(getattr(router, "threshold", 0.0))
    out["latency_ms_p50"], out["latency_ms_p95"] = float(np.percentile(latencies, 50)), float(np.percentile(latencies, 95))
    return out


def _review_agreement(path: Path = Path("reports/router_label_review.csv")) -> dict | None:
    if not path.exists():
        return None
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["label_ok"] in ("0", "1")]
    return {"n": len(rows), "agreement": sum(r["label_ok"] == "1" for r in rows) / len(rows) if rows else None,
            "reviewer": rows[0]["reviewer"] if rows else None}


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def _report(res: dict) -> str:
    cols = ["macro_f1", "accuracy", "coverage", "selective_accuracy", "ece", "language_accuracy", "latency_ms_p50",
            "latency_ms_p95"]
    lines = ["# Router evaluation (generated by `make router` — do not edit)", "",
             f"**Chosen:** `{res['chosen']}` · threshold {res['threshold']:.3f} · dataset `{res['dataset_hash'][:12]}`",
             "", f"**Selection rule.** {SELECTION_RULE}", "", SPEC_DEVIATIONS, "",
             "Validation and test metrics below are at the tuned threshold of each router (keyword has none); "
             "macro-F1 and accuracy do not depend on the threshold.", "",
             f"Data: train {res['n_train']} · val {res['n_val']} · test {res['n_test']} (sealed, hand-written). "
             f"Decontamination: {res['decontaminated']} train/val rows dropped for near-duplicating a test utterance; "
             f"leakage check after (char-TF-IDF cosine ≥ 0.9 test↔train/val): {res['leakage_hits']} hits.", ""]
    for split in ("val", "test"):
        lines += [f"## {split}", "", "| router | " + " | ".join(cols) + " | injection F1 |",
                  "|---" * (len(cols) + 2) + "|"]
        for name, m in res[split].items():
            lines.append(f"| {name} | " + " | ".join(_fmt(m[c]) for c in cols) + f" | {_fmt(m['injection']['f1'])} |")
        lines.append("")
    chosen = res["test"][res["chosen"]]
    lines += ["## Chosen router on test — by language", "",
              "| language | n | macro-F1 |", "|---|---|---|"]
    lines += [f"| {l} | {chosen['n_by_language'][l]} | {_fmt(v)} |" for l, v in chosen["per_language_macro_f1"].items()]
    lines += ["", "## Chosen router on test — per intent F1", "", "| intent | F1 |", "|---|---|"]
    lines += [f"| {i} | {_fmt(v)} |" for i, v in chosen["per_class_f1"].items()]
    rev = res["label_review"]
    lines += ["", "## Label quality", "",
              f"Paraphrase label review: {rev['n']} rows, agreement {_fmt(rev['agreement'])}, reviewer: {rev['reviewer']}."
              if rev else "Paraphrase label review: not available.",
              "", "## Limitations", "",
              "- Test set and seeds share an author (Claude); the test set was sealed before seeds existed and a "
              "near-duplicate check guards against copying, but shared phrasing habits can still inflate scores.",
              "- Portuguese utterances were written by a non-native author; no real customer text exists in the "
              "dataset (transcripts are templated, see eda_findings Q3).",
              f"- Small test set (n={res['n_test']}); per-intent F1 rests on ~10 utterances each. Treat differences "
              "under ~0.05 as noise.", ""]
    return "\n".join(lines)


def run(data_dir: Path = DATA_DIR, out_dir: Path = Path("models/router_v1"),
        report_path: Path = Path("reports/router_eval.md"), include_e5: bool = True) -> dict:
    trainval, test = load_examples(data_dir)
    trainval, dropped = decontaminate(trainval, test)
    train, val = split_train_val(trainval)
    leaks = leakage_report([e.text for e in trainval], [e.text for e in test])
    if leaks:
        raise RuntimeError(f"{len(leaks)} test utterances near-duplicate training data, e.g. "
                           f"{[(test[i].id, trainval[j].id, round(s, 3)) for i, j, s in leaks[:5]]}")
    routers = {"keyword": KeywordRouter(), "tfidf": tfidf_router().fit(train)}
    if include_e5:
        routers["e5"] = e5_router().fit(train)
    val_res = {n: evaluate_router(r, val) for n, r in routers.items()}
    chosen = select_model({n: m["macro_f1"] for n, m in val_res.items()})
    for name, router in routers.items():
        if name != "keyword":
            preds = router.predict_many([e.text for e in val])
            router.threshold = tune_threshold([p.confidence for p in preds],
                                              [p.intent == e.intent for p, e in zip(preds, val)])
    val_res = {n: evaluate_router(r, val) for n, r in routers.items()}  # re-scored at the tuned threshold
    test_res = {n: evaluate_router(r, test) for n, r in routers.items()}
    res = {"chosen": chosen, "threshold": routers[chosen].threshold,
           "thresholds": {n: float(r.threshold) for n, r in routers.items()}, "dataset_hash": dataset_hash(trainval + test),
           "n_train": len(train), "n_val": len(val), "n_test": len(test), "leakage_hits": len(leaks),
           "decontaminated": len(dropped), "val": val_res, "test": test_res, "label_review": _review_agreement()}
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"model": chosen, "version": routers[chosen].version, "threshold": res["threshold"],
            "selection_rule": SELECTION_RULE, "dataset_hash": res["dataset_hash"], "n_train": len(train),
            "n_val": len(val), "n_test": len(test), "val": val_res[chosen], "test": test_res[chosen],
            "git_sha": git_sha, "created_at": datetime.now(timezone.utc).isoformat()}
    if chosen == "keyword":
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=2))
    else:
        routers[chosen].save(out_dir, meta)
    Path(report_path).write_text(_report(res))
    Path(report_path).with_suffix(".json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    out = run()
    print(json.dumps({"chosen": out["chosen"], "val_macro_f1": {n: m["macro_f1"] for n, m in out["val"].items()},
                      "test_macro_f1": {n: m["macro_f1"] for n, m in out["test"].items()}}, indent=2))
