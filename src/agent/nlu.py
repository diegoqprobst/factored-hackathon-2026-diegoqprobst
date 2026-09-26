"""Message understanding. Extractors return data only — the orchestrator decides what, if anything, to do.
RuleExtractor is the baseline and the fallback; LLMExtractor (below) adds an LLM for free-form messages."""
import re
from dataclasses import dataclass, fields
from datetime import date, timedelta

from src.agent.state import Stage
from src.bank import config
from src.router.keyword import KeywordRouter
from src.router.labels import DISPUTE_TYPE_BY_INTENT, normalize

TYPE_BY_NUMBER = {1: "unrecognized", 2: "duplicate", 3: "amount_mismatch", 4: "undue_fee", 5: "refund_not_received"}
YES = {"si", "s", "dale", "ok", "okay", "confirmo", "confirmar", "correcto", "claro", "afirmativo", "sim", "isso",
       "pode", "exato", "yes", "listo", "perfecto", "bora", "confirma", "hazlo", "adelante", "certo"}
NO = {"no", "nao", "cancela", "cancelar", "negativo", "nop", "nope", "jamas", "nunca"}
ORDINALS = {"primero": 1, "primera": 1, "primeiro": 1, "segundo": 2, "segunda": 2, "tercero": 3, "tercera": 3,
            "terceiro": 3, "cuarto": 4, "quarto": 4, "quinto": 5, "ultimo": -1, "ultima": -1}
WORD_NUMBERS = {"dos": 2, "dois": 2, "duas": 2, "tres": 3, "cuatro": 4, "quatro": 4, "cinco": 5, "varios": 3,
                "varias": 3}
REFUND_OR_CREDIT = re.compile(r"(provisional|provisori|compensa[cç]|abon[ae]n?me|devu[eé]lv[ae]n?me|"
                              r"reembols[ea]n?me|me reembolse|devolva[m]? (o |meu )?dinheiro|credito na conta)")
_AMOUNT = re.compile(r"(?<![\w/.,-])(\d{1,3}(?:[.,]\d{3})+|\d+)([.,]\d{1,2})?(?!\d)\s*(mil\b|k\b)?"
                     r"(?!\s*(?:[/-]\d|veces\b|vezes\b|d[ií]as\b|dias\b|semanas\b|meses\b|horas\b|x\b|cargos\b|compras\b|cobros\b|"
                     r"cobran[cç]as\b|transacciones\b|transa[cç][oõ]es\b|movimientos\b|consumos\b))")
_DATE = re.compile(r"(?<!\d)(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?(?!\d)")
_MERCHANT = re.compile(r"\b(?:en|de|del|na|no|em|da|do)\s+(?:el\s+|la\s+|o\s+|a\s+)?"
                       r"([A-ZÁÉÍÓÚÑ][\w&'.-]*(?:\s+[A-ZÁÉÍÓÚÑ][\w&'.-]*)?)")
_KEYWORDS = KeywordRouter()


@dataclass(frozen=True)
class Extraction:
    document_number: str | None = None
    otp_code: str | None = None
    amount: float | None = None
    date_from: date | None = None
    date_to: date | None = None
    merchant: str | None = None
    dispute_type: str | None = None
    choice: int | None = None
    confirm: bool | None = None
    charges_count: int | None = None
    wants_human: bool = False
    wants_refund_or_credit: bool = False
    source: str = "rules"

    def present_fields(self) -> list[str]:
        return [f.name for f in fields(self) if f.name != "source" and getattr(self, f.name) not in (None, False)]


def parse_amount(text: str) -> float | None:
    for m in _AMOUNT.finditer((text or "").lower()):
        value = float(re.sub(r"[.,]", "", m.group(1)))
        if m.group(2):
            value += float("0." + m.group(2)[1:])
        if m.group(3):
            value *= 1000
        if value > 0:
            return value
    return None


def parse_dates(text: str, today: date) -> tuple[date | None, date | None]:
    t = normalize(text or "")
    for pattern, delta in ((r"\b(anteayer|antier|anteontem)\b", 2), (r"\b(ayer|ontem)\b", 1), (r"\b(hoy|hoje)\b", 0)):
        if re.search(pattern, t):
            d = today - timedelta(days=delta)
            return d, d
    if re.search(r"semana pasada|semana passada", t):
        return today - timedelta(days=14), today - timedelta(days=7)
    m = _DATE.search(t)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else today.year
        year = year + 2000 if year < 100 else year
        try:
            d = date(year, month, day)
        except ValueError:
            return None, None
        if not m.group(3) and d > today:
            d = date(year - 1, month, day)
        return d, d
    return None, None


def parse_confirm(text: str) -> bool | None:
    words = re.findall(r"[a-z]+", normalize(text or ""))
    yes, no = any(w in YES for w in words), any(w in NO for w in words)
    if yes and not no:
        return True
    if no and not yes:
        return False
    return None


def parse_document(text: str) -> str | None:
    m = re.search(r"(?<![\w])([A-Za-z]?\d[\d. -]{1,14}\d)(?![\w])", text or "")
    return re.sub(r"[. -]", "", m.group(1)).upper() if m else None


def parse_otp(text: str) -> str | None:
    m = re.search(r"(?<!\d)(\d{3})\s?(\d{3})(?!\d)", text or "")
    return m.group(1) + m.group(2) if m else None


def parse_charges_count(text: str) -> int | None:
    t = normalize(text or "")
    m = re.search(r"\b(\d{1,2}|dos|dois|duas|tres|cuatro|quatro|cinco|varios|varias)\s+"
                  r"(cargos|compras|cobros|cobrancas|transacciones|transacoes|movimientos|consumos)", t)
    if not m:
        return None
    word = m.group(1)
    n = int(word) if word.isdigit() else WORD_NUMBERS[word]
    return n if n >= 2 else None


def _merchant(text: str) -> str | None:
    for m in _MERCHANT.finditer(text or ""):
        candidate = m.group(1).strip(" .,;:!?")
        if normalize(candidate) not in {"no", "mi", "la", "el", "sim", "nao"} and not candidate.isdigit():
            return candidate
    return None


def _choice(text: str) -> int | None:
    t = normalize(text or "")
    m = re.search(r"(?<![\d.,])([1-9])(?![\d.,])", t)
    if m:
        return int(m.group(1))
    for word in re.findall(r"[a-z]+", t):
        if word in ORDINALS:
            return ORDINALS[word]
    return None


class RuleExtractor:
    def __init__(self, today: date = config.SIM_TODAY):
        self.today = today

    def extract(self, text: str, stage: Stage, context: dict | None = None, tracer=None) -> Extraction:
        text = text or ""
        if stage is Stage.AUTH_DOC:
            return Extraction(document_number=parse_document(text))
        if stage is Stage.AUTH_OTP:
            return Extraction(otp_code=parse_otp(text))
        route = _KEYWORDS.predict(text)
        dispute_type = DISPUTE_TYPE_BY_INTENT.get(route.intent) if not route.abstain else None
        choice = _choice(text) if stage is Stage.CHOOSE else None
        if stage is Stage.CLASSIFY:
            n = _choice(text)
            dispute_type = TYPE_BY_NUMBER.get(n, dispute_type) if n and n > 0 else dispute_type
        date_from, date_to = parse_dates(text, self.today)
        return Extraction(
            amount=None if stage in (Stage.CONFIRM, Stage.BLOCK_OFFER, Stage.CLASSIFY) else parse_amount(text),
            date_from=date_from, date_to=date_to, merchant=_merchant(text), dispute_type=dispute_type, choice=choice,
            confirm=parse_confirm(text) if stage in (Stage.CONFIRM, Stage.BLOCK_OFFER) else None,
            charges_count=parse_charges_count(text),
            wants_human=route.intent == "human_request" and not route.abstain,
            wants_refund_or_credit=bool(REFUND_OR_CREDIT.search(normalize(text))))
