import base64
import json

import pytest

from src.bank import auth, config
from src.bank.errors import (AccountNotServiceable, NoVerifiedChannel, OtpExpired, OtpInvalid, OtpLocked,
                             SessionExpired, SessionInvalid)
from tests.helpers import last_code, login


def test_happy_path_creates_session(bank, frozen):
    s = auth.verify_session(login(bank, "111"))
    assert s.customer_id == "CLI-A"
    events = [r["event"] for r in bank.execute("select event from audit_log order by id")]
    assert events == ["otp_sent", "session_started"]


def test_challenge_masks_destination(bank, frozen):
    ch = auth.start_auth(bank, "111")
    assert ch.masked_destination == "***1234"


def test_wrong_codes_lock_the_challenge(bank, frozen):
    ch = auth.start_auth(bank, "111")
    good = last_code(bank)
    bad = "000000" if good != "000000" else "111111"
    with pytest.raises(OtpInvalid):
        auth.verify_otp(bank, ch.challenge_id, bad)
    with pytest.raises(OtpInvalid):
        auth.verify_otp(bank, ch.challenge_id, bad)
    with pytest.raises(OtpLocked):
        auth.verify_otp(bank, ch.challenge_id, bad)
    with pytest.raises(OtpLocked):
        auth.verify_otp(bank, ch.challenge_id, good)


def test_otp_expires(bank, frozen):
    ch = auth.start_auth(bank, "111")
    frozen.advance(config.OTP_TTL_SECONDS + 1)
    with pytest.raises(OtpExpired):
        auth.verify_otp(bank, ch.challenge_id, last_code(bank))


def test_otp_is_single_use(bank, frozen):
    ch = auth.start_auth(bank, "111")
    code = last_code(bank)
    auth.verify_otp(bank, ch.challenge_id, code)
    with pytest.raises(OtpInvalid):
        auth.verify_otp(bank, ch.challenge_id, code)


def test_unknown_document_gets_unverifiable_decoy(bank, frozen):
    ch = auth.start_auth(bank, "999999")
    assert ch.masked_destination.startswith("***") and len(ch.masked_destination) == 7
    with pytest.raises(OtpInvalid):
        auth.verify_otp(bank, ch.challenge_id, "123456")
    assert bank.execute("select count(*) from sandbox_outbox").fetchone()[0] == 0


def test_customer_without_phone_cannot_authenticate(bank, frozen):  # Review Focus 1
    with pytest.raises(NoVerifiedChannel):
        auth.start_auth(bank, "333")
    assert bank.execute("select count(*) from otp_challenges").fetchone()[0] == 0


def test_closed_customer_is_not_serviceable(bank, frozen):
    with pytest.raises(AccountNotServiceable):
        auth.start_auth(bank, "444")


def test_session_expires(bank, frozen):
    token = login(bank, "111")
    frozen.advance(config.SESSION_TTL_SECONDS)
    with pytest.raises(SessionExpired):
        auth.verify_session(token)


def test_tampered_or_garbage_token_rejected(bank, frozen):
    token = login(bank, "111")
    payload, sig = token.rsplit(".", 1)
    data = json.loads(base64.urlsafe_b64decode(payload))
    data["cid"] = "CLI-B"
    forged = base64.urlsafe_b64encode(json.dumps(data, sort_keys=True).encode()).decode() + "." + sig
    for bad in (forged, "garbage", ""):
        with pytest.raises(SessionInvalid):
            auth.verify_session(bad)
