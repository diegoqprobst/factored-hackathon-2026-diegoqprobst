"""Simulated OTP authentication and HMAC-signed session tokens.
A document number alone never authenticates: only a verified, unexpired, single-use OTP creates a session."""
import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.bank import audit, clock, config
from src.bank.errors import (AccountNotServiceable, NoVerifiedChannel, OtpExpired, OtpInvalid, OtpLocked,
                             SessionExpired, SessionInvalid)

SERVICEABLE_STATUSES = {"Active", "Inactive"}


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    masked_destination: str
    expires_at: datetime


@dataclass(frozen=True)
class Session:
    session_id: str
    customer_id: str
    expires_at: datetime


def _code_hash(challenge_id: str, code: str) -> str:
    return hashlib.sha256(f"{challenge_id}:{code}".encode()).hexdigest()


def _mask(phone: str) -> str:
    return "***" + "".join(ch for ch in phone if ch.isdigit())[-4:]


def start_auth(conn: sqlite3.Connection, document_number: str) -> Challenge:
    challenge_id = secrets.token_urlsafe(12)
    expires_at = clock.now() + timedelta(seconds=config.OTP_TTL_SECONDS)
    row = conn.execute("select customer_id, mobile_phone, customer_status from customers where document_number = ?",
                       (document_number.strip(),)).fetchone()
    if row is None:
        # Decoy: same response shape, can never verify -> no account enumeration.
        conn.execute("insert into otp_challenges(challenge_id, customer_id, code_hash, expires_at) values (?,?,?,?)",
                     (challenge_id, None, "!", expires_at.isoformat()))
        conn.commit()
        audit.log(conn, "auth_unknown_document")
        return Challenge(challenge_id, "***" + f"{secrets.randbelow(10**4):04d}", expires_at)
    customer_id = row["customer_id"]
    if row["customer_status"] not in SERVICEABLE_STATUSES:
        audit.log(conn, "auth_refused_status", customer_id=customer_id, status=row["customer_status"])
        raise AccountNotServiceable(row["customer_status"])
    if not row["mobile_phone"]:
        audit.log(conn, "auth_no_verified_channel", customer_id=customer_id)
        raise NoVerifiedChannel()
    code = f"{secrets.randbelow(10**6):06d}"
    conn.execute("insert into otp_challenges(challenge_id, customer_id, code_hash, expires_at) values (?,?,?,?)",
                 (challenge_id, customer_id, _code_hash(challenge_id, code), expires_at.isoformat()))
    conn.execute("insert into sandbox_outbox(channel, destination, body, created_at) values (?,?,?,?)",
                 ("sms", row["mobile_phone"], f"Tu código de verificación es {code}", clock.now().isoformat()))
    conn.commit()
    audit.log(conn, "otp_sent", customer_id=customer_id, challenge_id=challenge_id)
    return Challenge(challenge_id, _mask(row["mobile_phone"]), expires_at)


def verify_otp(conn: sqlite3.Connection, challenge_id: str, code: str) -> str:
    row = conn.execute("select * from otp_challenges where challenge_id = ?", (challenge_id,)).fetchone()
    if row is None or row["customer_id"] is None or row["consumed"]:
        audit.log(conn, "otp_rejected", reason="unknown_or_used")
        raise OtpInvalid()
    customer_id = row["customer_id"]
    if clock.now() > datetime.fromisoformat(row["expires_at"]):
        raise OtpExpired()
    if row["attempts"] >= config.OTP_MAX_ATTEMPTS:
        raise OtpLocked()
    if not hmac.compare_digest(_code_hash(challenge_id, code.strip()), row["code_hash"]):
        attempts = row["attempts"] + 1
        conn.execute("update otp_challenges set attempts = ? where challenge_id = ?", (attempts, challenge_id))
        conn.commit()
        audit.log(conn, "otp_rejected", customer_id=customer_id, reason="wrong_code", attempts=attempts)
        if attempts >= config.OTP_MAX_ATTEMPTS:
            raise OtpLocked()
        raise OtpInvalid()
    conn.execute("update otp_challenges set consumed = 1 where challenge_id = ?", (challenge_id,))
    conn.commit()
    session_id = secrets.token_urlsafe(12)
    expires_at = clock.now() + timedelta(seconds=config.SESSION_TTL_SECONDS)
    audit.log(conn, "session_started", customer_id=customer_id, session_id=session_id)
    return issue_token(session_id, customer_id, expires_at)


def _sign(payload: str) -> str:
    return hmac.new(config.session_secret(), payload.encode(), hashlib.sha256).hexdigest()


def issue_token(session_id: str, customer_id: str, expires_at: datetime) -> str:
    body = json.dumps({"sid": session_id, "cid": customer_id, "exp": expires_at.isoformat()}, sort_keys=True)
    payload = base64.urlsafe_b64encode(body.encode()).decode()
    return f"{payload}.{_sign(payload)}"


def verify_session(token: str) -> Session:
    try:
        payload, sig = token.rsplit(".", 1)
    except (ValueError, AttributeError):
        raise SessionInvalid()
    if not hmac.compare_digest(sig, _sign(payload)):
        raise SessionInvalid()
    data = json.loads(base64.urlsafe_b64decode(payload))
    expires_at = datetime.fromisoformat(data["exp"])
    if clock.now() >= expires_at:
        raise SessionExpired()
    return Session(data["sid"], data["cid"], expires_at)
