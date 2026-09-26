import pytest

from src.router.evaluate import ece, evaluate_router, metrics, select_model, tune_threshold
from src.router.keyword import KeywordRouter
from src.router.labels import Example, RouterResult


def test_ece():
    assert ece([1.0, 1.0], [1, 0]) == pytest.approx(0.5)
    assert ece([0.5, 0.5], [1, 0]) == pytest.approx(0.0)


def test_tune_threshold_picks_smallest_meeting_target():
    assert tune_threshold([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 1], target=0.95) == 0.8


def test_tune_threshold_abstains_everything_when_target_unreachable():
    assert tune_threshold([0.9, 0.8], [0, 0]) > 0.9


@pytest.mark.parametrize("scores, chosen", [
    ({"keyword": 0.5, "tfidf": 0.8, "e5": 0.81}, "tfidf"),   # e5 wins by < 0.02 -> tfidf
    ({"keyword": 0.5, "tfidf": 0.8, "e5": 0.9}, "e5"),
    ({"keyword": 0.9, "tfidf": 0.8}, "keyword"),
    ({"keyword": 0.8, "tfidf": 0.8}, "keyword"),              # tie -> simpler
])
def test_select_model_rule(scores, chosen):
    assert select_model(scores) == chosen


def ex(i, intent, lang="es", inj=False):
    return Example(f"e{i}", "x", intent, lang, inj, f"e{i}")


def res(intent, conf=0.9, lang="es", inj=False, abstain=False):
    return RouterResult(intent, conf, lang, inj, float(inj), abstain, "v")


def test_metrics():
    examples = [ex(0, "a"), ex(1, "b"), ex(2, "b", "pt", True), ex(3, "a", "pt")]
    preds = [res("a"), res("a"), res("b", lang="pt", inj=True), res("a", lang="es", abstain=True)]
    m = metrics(examples, preds)
    assert m["n"] == 4 and m["accuracy"] == 0.75
    assert m["coverage"] == 0.75 and m["selective_accuracy"] == pytest.approx(2 / 3)
    assert m["language_accuracy"] == 0.75
    assert m["injection"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert set(m["per_language_macro_f1"]) == {"es", "pt"} and m["n_by_language"] == {"es": 2, "pt": 2}


def test_evaluate_router_adds_latency():
    out = evaluate_router(KeywordRouter(), [Example("t", "me robaron la tarjeta", "card_lost_stolen", "es", False, "t")])
    assert out["accuracy"] == 1.0 and out["latency_ms_p50"] >= 0 and out["latency_ms_p95"] >= out["latency_ms_p50"]
