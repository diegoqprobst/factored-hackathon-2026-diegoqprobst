from collections import Counter
from pathlib import Path

import pytest

from src.bank import config, db
from src.eval.cases import CATEGORY_COUNTS, Case, build_cases, fmt_amount, load_cases, save_cases, verify_seal

REAL = Path(config.SANDBOX_PATH)


def test_fmt_amount():
    assert (fmt_amount(378.89, "es"), fmt_amount(378.89, "pt"), fmt_amount(1250.0, "es")) == ("378.89", "378,89", "1250")


def test_save_load_and_seal(tmp_path):
    path = tmp_path / "cases.jsonl"
    cases = [Case("c1", "human", "es", "Basic", "CLI-A", "111", "Quiero hablar con un asesor",
                  expected={"outcome": "escalate", "writes": [], "reasons": ["customer_requested_human"],
                            "rules": [], "forbidden_text": []})]
    digest = save_cases(cases, path)
    assert load_cases(path) == cases and verify_seal(path) and (tmp_path / "SEALED").read_text().startswith(digest)
    path.write_text(path.read_text().replace("asesor", "gerente"))
    assert not verify_seal(path)


@pytest.mark.skipif(not REAL.exists(), reason="needs the real sandbox (make pipeline)")
def test_build_cases_on_real_sandbox():
    conn = db.connect(REAL)
    cases = build_cases(conn)
    counts = Counter(c.category for c in cases)
    assert counts == Counter(CATEGORY_COUNTS) and len(cases) == sum(CATEGORY_COUNTS.values())
    assert len({c.customer_id for c in cases}) == len(cases)  # one customer per case
    assert len({c.id for c in cases}) == len(cases)
    for cat, n in CATEGORY_COUNTS.items():
        langs = Counter(c.language for c in cases if c.category == cat)
        assert abs(langs["es"] - langs["pt"]) <= 1, cat
    for c in cases:
        text = " ".join(filter(None, [c.opening, c.details, c.clarification, c.choice, c.type_text]))
        assert "{" not in text and c.expected["outcome"] in {"resolved", "abstain", "refuse", "escalate", "safe"}
        for w in c.expected["writes"]:
            row = (conn.execute("select customer_id from transactions where transaction_id=?", (w["transaction_id"],))
                   if w["type"] == "dispute" else
                   conn.execute("select customer_id from products where product_id=?", (w["product_id"],))).fetchone()
            assert row[0] == c.customer_id  # expected writes always belong to the case's customer
    assert build_cases(conn) == cases  # deterministic
