"""Router dataset: join seeds + paraphrases, split by seed (the leakage unit), and check that the sealed test set
does not near-duplicate anything the models train or tune on."""
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from src.router.labels import DATA_DIR, Example, load_jsonl, normalize


def load_examples(data_dir: Path = DATA_DIR) -> tuple[list[Example], list[Example]]:
    data_dir = Path(data_dir)
    seeds = load_jsonl(data_dir / "seeds.jsonl")
    by_seed = {s["seed_id"]: s for s in seeds}
    trainval = [Example(s["seed_id"], s["text"], s["intent"], s["language"], s["injection"], s["seed_id"])
                for s in seeds]
    counters: dict[str, int] = defaultdict(int)
    para_path = data_dir / "paraphrases.jsonl"
    for p in load_jsonl(para_path) if para_path.exists() else []:
        s = by_seed.get(p["seed_id"])
        if s is None:
            continue
        k = counters[p["seed_id"]]
        counters[p["seed_id"]] += 1
        trainval.append(Example(f"{s['seed_id']}#p{k}", p["text"], s["intent"], s["language"], s["injection"],
                                s["seed_id"]))
    test = [Example(r["id"], r["text"], r["intent"], r["language"], r["injection"], r["id"])
            for r in load_jsonl(data_dir / "test_handwritten.jsonl")]
    return trainval, test


def split_train_val(examples: list[Example], val_fraction: float = 0.2,
                    seed: int = 42) -> tuple[list[Example], list[Example]]:
    cells: dict[tuple, set[str]] = defaultdict(set)
    for e in examples:
        cell = (e.language, "injection") if e.injection else (e.intent, e.language)
        cells[cell].add(e.group)
    val_groups: set[str] = set()
    for cell in sorted(cells):
        groups = sorted(cells[cell], key=lambda g: hashlib.sha1(f"{seed}:{g}".encode()).hexdigest())
        k = max(1, round(len(groups) * val_fraction)) if len(groups) >= 3 else 0
        val_groups.update(groups[:k])
    return ([e for e in examples if e.group not in val_groups], [e for e in examples if e.group in val_groups])


def _plain(text: str) -> str:
    """Punctuation-insensitive form, so "Bom dia!" and "bom dia" count as the same utterance."""
    return " ".join(re.sub(r"[^\w\s]", " ", normalize(text)).split())


def leakage_report(train_texts: list[str], test_texts: list[str],
                   threshold: float = 0.9) -> list[tuple[int, int, float]]:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=_plain).fit(train_texts + test_texts)
    sims = (vec.transform(test_texts) @ vec.transform(train_texts).T).toarray()
    hits = []
    for i, row in enumerate(sims):
        j = int(row.argmax())
        if row[j] >= threshold:
            hits.append((i, j, float(row[j])))
    return hits


def decontaminate(trainval: list[Example], test: list[Example],
                  threshold: float = 0.9) -> tuple[list[Example], list[Example]]:
    """Drop train/val rows whose text near-duplicates any test utterance (short canonical phrases such as
    "Bom dia" collide by chance). The sealed test set is never modified."""
    if not trainval or not test:
        return trainval, []
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=_plain)
    vec.fit([e.text for e in trainval] + [e.text for e in test])
    sims = (vec.transform([e.text for e in trainval]) @ vec.transform([e.text for e in test]).T).toarray()
    leaking = sims.max(axis=1) >= threshold
    return ([e for e, bad in zip(trainval, leaking) if not bad], [e for e, bad in zip(trainval, leaking) if bad])


def dataset_hash(examples: list[Example]) -> str:
    lines = sorted(json.dumps([e.id, e.text, e.intent, e.language, e.injection], ensure_ascii=False) for e in examples)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def write_review_sample(examples: list[Example], path: Path, n: int = 60, seed: int = 7) -> int:
    seed_text = {e.id: e.text for e in examples if e.id == e.group}
    paraphrases = [e for e in examples if "#p" in e.id]
    sample = random.Random(seed).sample(paraphrases, min(n, len(paraphrases)))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "seed_text", "text", "intent", "language", "injection", "label_ok", "reviewer"])
        for e in sample:
            w.writerow([e.id, seed_text.get(e.group, ""), e.text, e.intent, e.language, e.injection, "", ""])
    return len(sample)
