"""Project-wide constants for the mock bank and pipeline (values fixed by the design spec)."""
import os
from datetime import date
from pathlib import Path

SIM_TODAY = date(2026, 6, 17)  # dataset end; the sandbox "today"
SANDBOX_DAYS = 120
REPROCESS_DAYS = 3
SESSION_TTL_SECONDS = 15 * 60
OTP_TTL_SECONDS = 5 * 60
OTP_MAX_ATTEMPTS = 3
OTP_MAX_CHALLENGES_PER_WINDOW = 3  # per customer, per SESSION_TTL_SECONDS window

RAW_DIR = Path(os.environ.get("RAW_DIR", "data/raw"))
SILVER_DIR = Path(os.environ.get("SILVER_DIR", "data/silver"))
SANDBOX_PATH = Path(os.environ.get("SANDBOX_PATH", "data/sandbox.db"))
POLICY_PATH = Path(os.environ.get("POLICY_PATH", "policy/dispute_policy_v1.yaml"))


_PLACEHOLDER_SECRETS = {"", "change-me", "dev-only-secret"}


def session_secret() -> bytes:
    """Fails closed: a missing or placeholder secret would let anyone mint tokens for any customer."""
    value = os.environ.get("BANK_SESSION_SECRET", "")
    if value in _PLACEHOLDER_SECRETS:
        raise RuntimeError("BANK_SESSION_SECRET is unset or a placeholder; set a real secret in .env")
    return value.encode()
