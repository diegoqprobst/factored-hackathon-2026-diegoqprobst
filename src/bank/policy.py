"""Deterministic dispute policy. The LLM never evaluates these rules; it only communicates the Decision.
Rules are checked in order and the first non-eligible match wins."""
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import yaml

from src.bank import config
from src.bank.transactions import TransactionRecord

DISPUTE_TYPES = frozenset({"unrecognized", "duplicate", "amount_mismatch", "undue_fee", "refund_not_received"})
Outcome = Literal["eligible", "ineligible", "requires_human"]


@dataclass(frozen=True)
class Policy:
    version: str
    window_days: int
    auto_limit_usd: float
    human_review_dispute_count: int


@dataclass(frozen=True)
class DisputeContext:
    transaction: TransactionRecord
    amount_usd: float | None
    disputed_in_session: int
    is_repeat_complainer: bool
    existing_dispute_id: str | None
    today: date


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    rule_id: str
    reason: str
    policy_version: str
    existing_dispute_id: str | None = None


def load_policy(path: Path = config.POLICY_PATH) -> Policy:
    data = yaml.safe_load(Path(path).read_text())
    return Policy(str(data["version"]), int(data["window_days"]), float(data["auto_limit_usd"]),
                  int(data["human_review_dispute_count"]))


def evaluate(ctx: DisputeContext, policy: Policy) -> Decision:
    def d(outcome: Outcome, rule: str, reason: str, existing: str | None = None) -> Decision:
        return Decision(outcome, rule, reason, policy.version, existing)

    t = ctx.transaction
    if (ctx.today - t.local_date).days > policy.window_days:
        return d("ineligible", "P1", "outside dispute window")
    if t.status == "Declined":
        return d("ineligible", "P2", "no charge was made")
    if t.status == "Reversed":
        return d("ineligible", "P3", "already reversed")
    if ctx.amount_usd is None:
        return d("requires_human", "P4", "amount could not be converted to USD")
    if ctx.amount_usd > policy.auto_limit_usd:
        return d("requires_human", "P4", "amount above automatic limit")
    if ctx.disputed_in_session >= policy.human_review_dispute_count:
        return d("requires_human", "P5", "multiple disputed charges")
    if ctx.is_repeat_complainer:
        return d("requires_human", "P6", "repeat complainer review")
    if ctx.existing_dispute_id:
        return d("ineligible", "P7", "already disputed", ctx.existing_dispute_id)
    return d("eligible", "P8", "eligible for automatic filing")
