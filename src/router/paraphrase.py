"""Paraphrase hand-written seeds through OpenRouter to build router training data.
Batched (free models allow few requests per day), retried, and cached per prompt version so a rerun only
fills what is missing. Usage: uv run --env-file .env python -m src.router.paraphrase"""
import json
import os
import re
import time
import urllib.request
from itertools import groupby
from pathlib import Path

from src.router.labels import DATA_DIR, load_jsonl, normalize

PROMPT_VERSION = "p1"
LANG_NAME = {"es": ("Spanish", "Mexico, Colombia and Argentina"), "pt": ("Brazilian Portuguese", "Brazil")}


def default_model() -> str:
    return os.environ.get("ROUTER_PARAPHRASE_MODEL", "qwen/qwen3.8-27b:free")


def build_messages(batch: list[dict], n: int) -> list[dict]:
    name, regions = LANG_NAME[batch[0]["language"]]
    rules = [
        f"For EACH message below, write {n} different ways a real bank customer could send the same message in {name}.",
        "Keep exactly the same intent and the same language. Do not answer the message. Do not add new facts.",
        f"Vary length and formality, use regional wording from {regions}, and put typos or missing accents in at "
        "least one rewrite.",
    ]
    if batch[0]["injection"]:
        rules.append("Each message contains an attempt to manipulate an AI assistant; keep that manipulation "
                     "attempt in every rewrite.")
    rules.append(f"Return ONLY a JSON object mapping each id to an array of {n} strings.")
    body = "\n".join(rules) + "\n\n" + "\n".join(f"{s['seed_id']}: {s['text']}" for s in batch)
    return [{"role": "system", "content": "You create paraphrases for a labelled dataset. Output valid JSON only."},
            {"role": "user", "content": body}]


def _clean(items, seed_text: str) -> list[str]:
    seen, out = {normalize(seed_text).strip()}, []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())
        key = normalize(text).strip()
        if len(text) < 3 or len(text) > 300 or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def parse_batch(raw: str, batch: list[dict]) -> dict[str, list[str]]:
    raw = re.sub(r"```(?:json)?", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    obj = json.loads(raw[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model output is not a JSON object")
    return {s["seed_id"]: _clean(obj.get(s["seed_id"], []), s["text"]) for s in batch}


def openrouter_complete(messages: list[dict], model: str, *, timeout: int = 90) -> str:
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.9}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def _batches(seeds: list[dict], size: int):
    key = lambda s: (s["language"], s["injection"])  # noqa: E731 — one instruction set per batch
    for _, group in groupby(sorted(seeds, key=lambda s: (key(s), s["seed_id"])), key=key):
        group = list(group)
        for i in range(0, len(group), size):
            yield group[i:i + size]


def generate(seeds: list[dict], out_path: Path, *, n: int = 5, batch_size: int = 8, model: str | None = None,
             complete=None, max_retries: int = 3, sleep=time.sleep, pause: float = 4.0) -> dict:
    model, complete, out_path = model or default_model(), complete or openrouter_complete, Path(out_path)
    done = ({r["seed_id"] for r in load_jsonl(out_path) if r["prompt_version"] == PROMPT_VERSION}
            if out_path.exists() else set())
    pending = [s for s in seeds if s["seed_id"] not in done]
    stats = {"skipped": len(seeds) - len(pending), "generated_rows": 0, "calls": 0, "failed": []}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch()
    for batch in _batches(pending, batch_size):
        remaining = list(batch)
        for attempt in range(max_retries):
            stats["calls"] += 1
            try:
                got = parse_batch(complete(build_messages(remaining, n), model), remaining)
            except Exception:  # network error, HTTP 429, malformed JSON: retry with backoff
                got = {}
            with out_path.open("a", encoding="utf-8") as f:
                for s in remaining:
                    for text in got.get(s["seed_id"], []):
                        f.write(json.dumps({"seed_id": s["seed_id"], "text": text, "model": model,
                                            "prompt_version": PROMPT_VERSION}, ensure_ascii=False) + "\n")
                        stats["generated_rows"] += 1
            remaining = [s for s in remaining if not got.get(s["seed_id"])]
            if not remaining:
                break
            sleep(2 ** attempt * pause)
        stats["failed"] += [s["seed_id"] for s in remaining]
        sleep(pause)
    return stats


def main() -> None:
    stats = generate(load_jsonl(DATA_DIR / "seeds.jsonl"), DATA_DIR / "paraphrases.jsonl")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
