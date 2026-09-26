"""Per-conversation state. The orchestrator owns it; neither the router nor the LLM can change it directly."""
from dataclasses import dataclass, field
from enum import Enum


class Stage(str, Enum):
    INTAKE = "intake"
    AUTH_DOC = "auth_doc"
    AUTH_OTP = "auth_otp"
    IDENTIFY = "identify"
    CHOOSE = "choose"
    CLASSIFY = "classify"
    CONFIRM = "confirm"
    BLOCK_OFFER = "block_offer"
    DONE = "done"
    HANDOFF = "handoff"


SLOT_FIELDS = ("amount", "date_from", "date_to", "merchant")


@dataclass
class Conversation:
    id: str
    stage: Stage = Stage.INTAKE
    language: str = "es"
    token: str | None = None
    challenge_id: str | None = None
    customer_id: str | None = None
    intent: str | None = None
    dispute_type: str | None = None
    slots: dict = field(default_factory=dict)
    charges_count: int | None = None
    candidates: list[dict] = field(default_factory=list)
    choose_kind: str = "transaction"
    transaction_id: str | None = None
    selected_view: dict | None = None
    product_id: str | None = None
    decision: dict | None = None
    disputes: list[str] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    injection_count: int = 0
    low_conf_count: int = 0
    not_found_count: int = 0
    choose_retries: int = 0
    handoff_id: str | None = None
    turns: int = 0

    def remember(self, ext) -> None:
        for name in SLOT_FIELDS:
            value = getattr(ext, name, None)
            if value is not None:
                self.slots[name] = value
        if getattr(ext, "charges_count", None):
            self.charges_count = max(self.charges_count or 0, ext.charges_count)

    def reset_case(self) -> None:
        self.intent = self.dispute_type = self.transaction_id = self.selected_view = None
        self.product_id = self.decision = self.charges_count = None
        self.slots, self.candidates, self.choose_kind = {}, [], "transaction"
        self.not_found_count = self.choose_retries = 0
