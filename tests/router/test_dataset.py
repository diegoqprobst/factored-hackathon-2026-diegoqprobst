import csv
import json

from src.router.dataset import dataset_hash, leakage_report, load_examples, split_train_val, write_review_sample
from src.router.labels import Example


def fixture():
    out = []
    for intent in ("dispute_duplicate", "card_lost_stolen", "human_request"):
        for lang in ("es", "pt"):
            for g in range(5):
                gid = f"{intent}-{lang}-{g}"
                out += [Example(f"{gid}#p{i}", f"texto {gid} {i}", intent, lang, False, gid) for i in range(3)]
    return out


def test_split_is_grouped_stratified_and_deterministic():
    data = fixture()
    train, val = split_train_val(data, val_fraction=0.2)
    assert {e.group for e in train}.isdisjoint({e.group for e in val})
    assert {(e.intent, e.language) for e in val} == {(e.intent, e.language) for e in data}
    assert len({e.group for e in val}) == 6
    assert split_train_val(data, val_fraction=0.2) == (train, val)


def test_small_cells_stay_in_train():
    data = [Example("g#p0", "x", "oos_credit", "es", False, "g")]
    assert split_train_val(data) == (data, [])


def test_leakage_report_flags_near_duplicates():
    hits = leakage_report(["no reconozco un cargo de oxxo"], ["No reconozco un cargo de Oxxo!", "quero falar com uma pessoa"])
    assert [h[0] for h in hits] == [0] and hits[0][2] >= 0.9


def test_load_examples_joins_paraphrases_to_seed_labels(tmp_path):
    seed = {"seed_id": "s1", "text": "me cobraron doble", "intent": "dispute_duplicate", "language": "es", "injection": False}
    (tmp_path / "seeds.jsonl").write_text(json.dumps(seed) + "\n")
    (tmp_path / "paraphrases.jsonl").write_text("".join(
        json.dumps({"seed_id": "s1", "text": t, "model": "m", "prompt_version": "p1"}) + "\n"
        for t in ("me cobraron dos veces", "cobro duplicado")))
    (tmp_path / "test_handwritten.jsonl").write_text(json.dumps(
        {"id": "t1", "text": "doble cobro", "intent": "dispute_duplicate", "language": "es", "injection": False}) + "\n")
    trainval, test = load_examples(tmp_path)
    assert [e.id for e in trainval] == ["s1", "s1#p0", "s1#p1"]
    assert all(e.group == "s1" and e.intent == "dispute_duplicate" for e in trainval)
    assert (test[0].id, test[0].group) == ("t1", "t1")


def test_dataset_hash_is_order_independent():
    data = fixture()
    assert dataset_hash(data) == dataset_hash(list(reversed(data)))
    assert dataset_hash(data) != dataset_hash(data[1:])


def test_write_review_sample(tmp_path):
    data = fixture() + [Example("dispute_duplicate-es-0", "seed text", "dispute_duplicate", "es", False,
                                "dispute_duplicate-es-0")]
    n = write_review_sample(data, tmp_path / "r.csv", n=10)
    rows = list(csv.DictReader((tmp_path / "r.csv").open()))
    assert n == 10 and len(rows) == 10 and all("#p" in r["id"] for r in rows)
    assert set(rows[0]) == {"id", "seed_text", "text", "intent", "language", "injection", "label_ok", "reviewer"}


def test_decontaminate_drops_train_rows_that_near_duplicate_test():
    from src.router.dataset import decontaminate
    trainval = [Example("s1#p0", "Bom dia", "greeting_smalltalk", "pt", False, "s1"),
                Example("s1", "Olá, boa noite", "greeting_smalltalk", "pt", False, "s1")]
    test = [Example("t1", "bom dia", "greeting_smalltalk", "pt", False, "t1")]
    kept, dropped = decontaminate(trainval, test)
    assert [e.id for e in kept] == ["s1"] and [e.id for e in dropped] == ["s1#p0"]
    assert leakage_report([e.text for e in kept], [e.text for e in test]) == []
