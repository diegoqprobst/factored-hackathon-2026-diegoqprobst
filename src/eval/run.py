"""Evaluation CLI.
  uv run python -m src.eval.run build            # build + seal cases from data/sandbox.db
  uv run --group embeddings --env-file .env python -m src.eval.run run --systems baseline,hybrid --repeats 3
"""
import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path

from src.bank import config, db
from src.eval.cases import CASES_PATH, SEED, build_cases, load_cases, save_cases, verify_seal
from src.eval.report import render_report
from src.eval.runner import LockedRouter, run_all
from src.eval.scoring import agreement, aggregate, mcnemar, observed_writes, score


def _fresh_copy() -> tuple[Path, object]:
    tmp = Path(tempfile.mkdtemp(prefix="eval-")) / "sandbox.db"
    shutil.copy(config.SANDBOX_PATH, tmp)
    conn = db.connect(tmp)
    db.create_schema(conn)
    return tmp, conn


def _run_system(system, cases, shared, workers, max_cost):
    from src.agent.factory import build_agent
    path, conn = _fresh_copy()
    try:
        def make_agent(faults):
            return build_agent(conn, system, router=shared.get("router"), llm=shared.get("llm"), faults=faults)
        done = {"n": 0}

        def progress(run):
            done["n"] += 1
            if done["n"] % 20 == 0:
                print(f"  {system}: {done['n']}/{len(cases)}", file=sys.stderr, flush=True)
        runs = run_all(cases, system=system, conn=conn, make_agent=make_agent, workers=workers,
                       max_cost_usd=max_cost, progress=progress)
        by_id = {c.id: c for c in cases}
        return [score(by_id[r.case_id], r, observed_writes(conn, by_id[r.case_id].customer_id)) for r in runs]
    finally:
        conn.close()
        shutil.rmtree(path.parent, ignore_errors=True)


def _subset(cases, n, seed=7):
    by_cat = defaultdict(list)
    for c in cases:
        by_cat[c.category].append(c)
    rng, out = random.Random(seed), []
    per = max(1, -(-n // len(by_cat)))  # ceil: ~n cases spread over every category
    for cat in sorted(by_cat):
        out += rng.sample(by_cat[cat], min(per, len(by_cat[cat])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true")
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--out", type=Path, default=CASES_PATH)
    r = sub.add_parser("run")
    r.add_argument("--systems", default="baseline,hybrid")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--repeat-subset", type=int, default=60)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--max-cost-usd", type=float, default=1.0)
    r.add_argument("--cases", type=Path, default=CASES_PATH)
    r.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)
    if args.cmd == "build":
        if args.out.exists() and not args.force:
            raise SystemExit(f"{args.out} is sealed; pass --force only before any system was scored")
        print(save_cases(build_cases(db.connect(config.SANDBOX_PATH), seed=args.seed), args.out))
        return
    try:
        config.session_secret()  # a misconfigured run would score every case as a safe internal-error handoff
    except RuntimeError as exc:
        raise SystemExit(f"refusing to evaluate: {exc} (run via `make eval`, which loads .env)")
    if not verify_seal(args.cases):
        raise SystemExit("case file does not match its seal")
    cases = load_cases(args.cases)
    systems = args.systems.split(",")
    shared = {}
    meta = {"cases": len(cases), "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "cases_file": str(args.cases),
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "date": date.today().isoformat(), "router": "keyword_v1", "llm_model": "none", "workers": args.workers}
    from src.agent.nlu import SYSTEM_PROMPT
    from src.bank.policy import load_policy
    from src.router.dataset import leakage_report, load_examples
    meta["prompt_sha256"] = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    meta["policy_version"] = load_policy().version
    meta["leakage_hits"] = len(leakage_report([e.text for e in load_examples()[0]], [c.opening for c in cases]))
    if "hybrid" in systems:
        from src.agent.llm import OpenRouterLLM
        from src.router.classifier import load_router
        import os
        shared["router"] = LockedRouter(load_router())
        shared["llm"] = OpenRouterLLM(os.environ.get("AGENT_LLM_MODEL", "google/gemma-4-31b-it"))
        meta["router"], meta["llm_model"] = shared["router"].version, shared["llm"].model
    results, summary = [], {"meta": meta, "systems": {}}
    for system in systems:
        print(f"running {system} on {len(cases)} cases", file=sys.stderr)
        rows = _run_system(system, cases, shared if system == "hybrid" else {}, args.workers, args.max_cost_usd)
        results += rows
        summary["systems"][system] = aggregate(rows)
    if {"baseline", "hybrid"} <= set(systems):
        base = [x for x in results if x["system"] == "baseline"]
        hyb = [x for x in results if x["system"] == "hybrid"]
        tests = {"SAR (expected resolved)": ("resolved", lambda x: x["success"]),
                 "escalated when required": ("escalate", lambda x: x["handed_off"]),
                 "task success (all)": (None, lambda x: x["success"]),
                 "unsafe (all)": (None, lambda x: x["unsafe"])}
        summary["paired"] = {name: mcnemar([x for x in base if scope in (None, x["expected_outcome"])], hyb, pred)
                             for name, (scope, pred) in tests.items()}
        for lang in ("es", "pt"):
            summary["paired"][f"task success ({lang})"] = mcnemar([x for x in base if x["language"] == lang], hyb,
                                                                   lambda x: x["success"])
    if "hybrid" in systems and args.repeats > 1:
        subset = _subset(cases, args.repeat_subset)
        reps = [_run_system("hybrid", subset, shared, args.workers, args.max_cost_usd) for _ in range(args.repeats)]
        summary["repeats"] = agreement(reps)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / "eval_report.md").write_text(render_report(summary))
    print(json.dumps({s: {"sar": a["sar"]["rate"], "unsafe": a["unsafe"]["k"], "cost": a["cost"]["total_usd"]}
                      for s, a in summary["systems"].items()}, indent=2))


if __name__ == "__main__":
    main()
