"""Label space and shared types for the router."""
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

INTENTS = (
    "dispute_unrecognized", "dispute_duplicate", "dispute_amount_mismatch", "dispute_undue_fee",
    "dispute_refund_not_received", "card_lost_stolen", "human_request", "oos_balance_movements", "oos_credit",
    "oos_other", "greeting_smalltalk",
)
LANGUAGES = ("es", "pt")
MAX_CHARS = 1000
DATA_DIR = Path("data_labels/router")
DISPUTE_TYPE_BY_INTENT = {
    "dispute_unrecognized": "unrecognized", "dispute_duplicate": "duplicate",
    "dispute_amount_mismatch": "amount_mismatch", "dispute_undue_fee": "undue_fee",
    "dispute_refund_not_received": "refund_not_received",
}
TEST_MIN_PER_CELL, TEST_MIN_INJECTION_PER_LANG, TEST_MIN_CODE_SWITCH = 5, 8, 6
SEED_MIN_PER_CELL, SEED_MIN_INJECTION_PER_LANG = 8, 12


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    intent: str
    language: str
    injection: bool
    group: str  # leakage unit: the seed id (test rows use their own id)


@dataclass(frozen=True)
class RouterResult:
    intent: str
    confidence: float
    language: str
    injection: bool
    injection_score: float
    abstain: bool
    model_version: str


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_records(records: list[dict], id_field: str) -> list[str]:
    errors, seen = [], set()
    for n, r in enumerate(records):
        rid = r.get(id_field)
        where = f"row {n} ({rid})"
        if not rid:
            errors.append(f"{where}: missing id")
        elif rid in seen:
            errors.append(f"{where}: duplicate id {rid}")
        seen.add(rid)
        if not str(r.get("text", "")).strip():
            errors.append(f"{where}: empty text")
        if r.get("intent") not in INTENTS:
            errors.append(f"{where}: unknown intent {r.get('intent')}")
        if r.get("language") not in LANGUAGES:
            errors.append(f"{where}: unknown language {r.get('language')}")
        if not isinstance(r.get("injection"), bool):
            errors.append(f"{where}: injection must be bool")
    return errors


def cell_counts(records: list[dict]) -> Counter:
    return Counter((r["intent"], r["language"]) for r in records)
