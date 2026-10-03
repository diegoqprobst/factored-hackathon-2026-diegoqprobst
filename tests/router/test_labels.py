import hashlib

from src.bank.policy import DISPUTE_TYPES
from src.router.labels import (DATA_DIR, DISPUTE_TYPE_BY_INTENT, INTENTS, LANGUAGES, TEST_MIN_CODE_SWITCH,
                               TEST_MIN_INJECTION_PER_LANG, TEST_MIN_PER_CELL, cell_counts, load_jsonl, normalize,
                               validate_records)


def test_normalize_strips_case_and_accents():
    assert normalize("Não RECONHEÇO la Compra") == "nao reconheco la compra"


def test_dispute_intents_map_to_policy_types():
    assert set(DISPUTE_TYPE_BY_INTENT.values()) == set(DISPUTE_TYPES)
    assert set(DISPUTE_TYPE_BY_INTENT) <= set(INTENTS)


def test_validate_records_reports_every_problem():
    recs = [{"id": "a", "text": "hola", "intent": "greeting_smalltalk", "language": "es", "injection": False},
            {"id": "a", "text": "", "intent": "nope", "language": "fr", "injection": "no"},
            {"text": "x", "intent": "oos_other", "language": "es", "injection": False}]
    errors = validate_records(recs, "id")
    assert any("duplicate id a" in e for e in errors)
    assert any("empty text" in e for e in errors)
    assert any("unknown intent nope" in e for e in errors)
    assert any("unknown language fr" in e for e in errors)
    assert any("injection must be bool" in e for e in errors)
    assert any("missing id" in e for e in errors)


def test_test_set_is_valid_complete_and_sealed():
    path = DATA_DIR / "test_handwritten.jsonl"
    recs = load_jsonl(path)
    assert validate_records(recs, "id") == []
    counts = cell_counts(recs)
    assert [(i, l) for i in INTENTS for l in LANGUAGES if counts[(i, l)] < TEST_MIN_PER_CELL] == []
    for lang in LANGUAGES:
        assert sum(r["injection"] and r["language"] == lang for r in recs) >= TEST_MIN_INJECTION_PER_LANG
    assert sum(r.get("note") == "code_switch" for r in recs) >= TEST_MIN_CODE_SWITCH
    sealed_hash = (DATA_DIR / "SEALED").read_text().split()[0]
    assert sealed_hash == hashlib.sha256(path.read_bytes()).hexdigest(), "test set changed after sealing"


from src.router.labels import SEED_MIN_INJECTION_PER_LANG, SEED_MIN_PER_CELL  # noqa: E402


def test_seeds_are_valid_complete_and_disjoint_from_test():
    seeds = load_jsonl(DATA_DIR / "seeds.jsonl")
    assert validate_records(seeds, "seed_id") == []
    counts = cell_counts(seeds)
    assert [(i, l) for i in INTENTS for l in LANGUAGES if counts[(i, l)] < SEED_MIN_PER_CELL] == []
    for lang in LANGUAGES:
        assert sum(s["injection"] and s["language"] == lang for s in seeds) >= SEED_MIN_INJECTION_PER_LANG
    test_texts = {normalize(r["text"]).strip() for r in load_jsonl(DATA_DIR / "test_handwritten.jsonl")}
    assert [s["seed_id"] for s in seeds if normalize(s["text"]).strip() in test_texts] == []
