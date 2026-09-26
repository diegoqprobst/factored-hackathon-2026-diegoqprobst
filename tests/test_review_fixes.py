"""Regression tests for the final-review findings (Important 1-6)."""
import threading
import time

import pytest

from src.bank import auth, cards, config, disputes as ds
from src.bank.errors import IdempotencyConflict, NotFound, OtpInvalid, OtpLocked, PolicyViolation
from tests.helpers import last_code, login


# Important 1 — the unknown-document decoy must be indistinguishable from a real challenge
def test_decoy_mask_is_stable_per_document(bank, frozen):
    assert auth.start_auth(bank, "999").masked_destination == auth.start_auth(bank, "999").masked_destination


def test_decoy_locks_like_a_real_challenge(bank, frozen):
    ch = auth.start_auth(bank, "999")
    results = []
    for _ in range(4):
        with pytest.raises((OtpInvalid, OtpLocked)) as e:
            auth.verify_otp(bank, ch.challenge_id, "000000")
        results.append(e.value.code)
    assert results == ["otp_invalid", "otp_invalid", "otp_locked", "otp_locked"]


# Important 2 — new challenges must not multiply the attempt budget
def test_new_challenge_invalidates_previous_one(bank, frozen):
    first = auth.start_auth(bank, "111")
    first_code = last_code(bank)
    auth.start_auth(bank, "111")
    with pytest.raises(OtpInvalid):
        auth.verify_otp(bank, first.challenge_id, first_code)


def test_challenge_rate_limit_per_customer(bank, frozen):
    for _ in range(config.OTP_MAX_CHALLENGES_PER_WINDOW):
        auth.start_auth(bank, "111")
    with pytest.raises(OtpLocked):
        auth.start_auth(bank, "111")
    frozen.advance(config.SESSION_TTL_SECONDS + 1)
    auth.start_auth(bank, "111")  # window elapsed


# Important 3 — the session secret must fail closed
@pytest.mark.parametrize("value", [None, "", "change-me", "dev-only-secret"])
def test_session_secret_fails_closed(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("BANK_SESSION_SECRET", raising=False)
    else:
        monkeypatch.setenv("BANK_SESSION_SECRET", value)
    with pytest.raises(RuntimeError):
        config.session_secret()


# Important 4 — concurrent writes must not bypass idempotency, P7 or single-use OTP
def _race(n, fn):
    out = [None] * n

    def run(i):
        try:
            out[i] = fn(i)
        except Exception as exc:  # noqa: BLE001 — the test inspects what each thread got
            out[i] = exc
    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def test_open_dispute_is_unique_per_transaction_at_db_level(bank, frozen):
    import sqlite3
    row = ("x", "TRX-A1", "unrecognized", "open", "P8", "v", "t")
    bank.execute("insert into disputes values ('D1','k1','s','CLI-A',?,?,?,?,?,?)", row[1:])
    with pytest.raises(sqlite3.IntegrityError):
        bank.execute("insert into disputes values ('D2','k2','s','CLI-A',?,?,?,?,?,?)", row[1:])


def test_concurrent_creates_on_same_transaction_yield_one_dispute(bank, frozen, monkeypatch):
    tok = login(bank, "111")
    original = ds.evaluate_dispute

    def slow(*a, **kw):
        d = original(*a, **kw)
        time.sleep(0.05)  # widen the check-then-act window
        return d
    monkeypatch.setattr(ds, "evaluate_dispute", slow)
    out = _race(2, lambda i: ds.create_dispute(bank, tok, "TRX-A1", "unrecognized", idempotency_key=f"k{i}",
                                                customer_confirmed=True))
    assert bank.execute("select count(*) from disputes").fetchone()[0] == 1
    errors = [o for o in out if isinstance(o, Exception)]
    assert len(errors) == 1 and isinstance(errors[0], PolicyViolation) and errors[0].decision.rule_id == "P7"


def test_concurrent_correct_otp_yields_one_session(bank, frozen, monkeypatch):
    ch = auth.start_auth(bank, "111")
    code = last_code(bank)
    original = auth._code_hash

    def slow(*a):
        time.sleep(0.05)
        return original(*a)
    monkeypatch.setattr(auth, "_code_hash", slow)
    out = _race(2, lambda i: auth.verify_otp(bank, ch.challenge_id, code))
    assert sum(isinstance(o, str) for o in out) == 1


# Important 5 — reusing a key for a different request is a conflict, not a silent success
def test_idempotency_key_reuse_with_different_request_conflicts(bank, frozen):
    tok = login(bank, "111")
    ds.create_dispute(bank, tok, "TRX-A1", "unrecognized", idempotency_key="k1", customer_confirmed=True)
    with pytest.raises(IdempotencyConflict):
        ds.create_dispute(bank, tok, "TRX-A9", "duplicate", idempotency_key="k1", customer_confirmed=True)


# Important 6 — foreign card and dispute IDs are audited
def _attempts(conn):
    return conn.execute("select count(*) from audit_log where event='cross_customer_access_attempt'").fetchone()[0]


def test_foreign_card_access_is_audited(bank, frozen):
    tok = login(bank, "111")
    with pytest.raises(NotFound):
        cards.block_card(bank, tok, "PRD-B-DC", customer_confirmed=True, reason="x")
    assert _attempts(bank) == 1


def test_foreign_dispute_access_is_audited(bank, frozen):
    d = ds.create_dispute(bank, login(bank, "111"), "TRX-A1", "unrecognized", idempotency_key="k1",
                          customer_confirmed=True)
    tokb = login(bank, "222")
    with pytest.raises(NotFound):
        ds.get_dispute(bank, tokb, d.dispute_id)
    with pytest.raises(NotFound):
        ds.create_dispute(bank, tokb, "TRX-B1", "unrecognized", idempotency_key="k1", customer_confirmed=True)
    assert _attempts(bank) == 2
