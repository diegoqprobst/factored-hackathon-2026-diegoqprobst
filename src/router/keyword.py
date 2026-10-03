"""Deterministic keyword router: the baseline the learned routers must beat, and the router of Plan 3's
rules-only baseline system. Matching is on normalised text (lowercase, no accents)."""
import re

from src.router.labels import INTENTS, MAX_CHARS, RouterResult, normalize

KEYWORDS = {
    "dispute_unrecognized": ["no reconozco", "no lo reconozco", "no la reconozco", "yo no hice", "no hice esa",
                             "no autorice", "cargo desconocido", "compra desconocida", "nao reconheco", "nao fiz",
                             "nao autorizei", "compra desconhecida", "cobranca desconhecida", "fraude"],
    "dispute_duplicate": ["dos veces", "doble", "duplicad", "repetid", "duas vezes", "em dobro"],
    "dispute_amount_mismatch": ["monto distinto", "monto diferente", "monto incorrecto", "mas de lo que",
                                "cobraron mas", "no coincide", "valor diferente", "valor errado", "valor incorreto",
                                "cobraram mais"],
    "dispute_undue_fee": ["comision", "cuota de manejo", "cobro indebido", "anualidad", "tarifa", "taxa",
                          "cobranca indevida", "juros", "cargo por"],
    "dispute_refund_not_received": ["reembolso", "devolucion", "no me han devuelto", "me devuelvan", "estorno",
                                    "devolucao", "cancele la compra", "cancelei"],
    "card_lost_stolen": ["robaron", "robo", "perdi la tarjeta", "perdi mi tarjeta", "extravie", "hurto",
                         "bloquea", "roubaram", "roubo", "perdi meu cartao", "perdi o cartao", "bloquei"],
    "human_request": ["asesor", "humano", "persona real", "hablar con alguien", "agente", "operador", "atendente",
                      "falar com uma pessoa", "falar com alguem", "pessoa real"],
    "oos_balance_movements": ["saldo", "movimientos", "extracto", "cuanto tengo", "extrato", "quanto tenho"],
    "oos_credit": ["prestamo", "emprestimo", "financiamiento", "financiamento", "solicitar un credito",
                   "pedir un credito", "quero um credito", "aumentar mi cupo", "limite de credito"],
    "oos_other": ["abrir una cuenta", "abrir cuenta", "horario", "sucursal", "transferencia", "abrir uma conta",
                  "agencia", "pix", "inversion", "investimento"],
    "greeting_smalltalk": ["hola", "buenos dias", "buenas tardes", "gracias", "ola", "oi", "bom dia", "boa tarde",
                           "obrigad", "chau", "tchau"],
}
ES_WORDS = {"el", "la", "los", "las", "mi", "mis", "tarjeta", "cobro", "cobraron", "hola", "gracias", "pero",
            "cuenta", "dinero", "hice", "quiero", "por", "favor", "usted", "no", "me", "hablar", "con", "una"}
PT_WORDS = {"o", "os", "meu", "minha", "cartao", "voce", "obrigado", "obrigada", "conta", "dinheiro", "ola", "oi",
            "fiz", "quero", "pra", "nao", "um", "com", "estou", "falar", "pessoa", "uma", "reconheco"}
# Keywords match at a word start ("ola" must not fire inside "hola"); stems like "duplicad" still match suffixes.
KEYWORD_PATTERNS = {i: [re.compile(r"\b" + re.escape(k)) for k in ks] for i, ks in KEYWORDS.items()}
INJECTION_PATTERNS = [re.compile(p) for p in (
    r"ignor(a|e|ar|em|ad)\b.*\b(instruc|instru|regla|regra|rule|polit)",
    r"(olvida|esquec[ae])\w*\b.*\b(regla|regra|instruc|instru)",
    r"system prompt|prompt del sistema|prompt do sistema",
    r"(modo|mode) (desarrollador|developer|desenvolvedor|dios|god)",
    r"\b(eres|actua como|voce e|aja como)\b.*\b(admin|administrador|root|sistema)",
    r"\b(otro|outro) cliente|\bcliente \d+",
    r"(todos|todas) (los|las|os|as) (clientes|transacciones|transacoes)",
    r"(aprueba|aprove|reembolsa|reembolse|devuelve).*\b(sin|sem) (verificar|confirmar|validar)",
    r"\bjailbreak\b",
)]


def intent_hits(text: str) -> dict[str, int]:
    t = normalize((text or "")[:MAX_CHARS])
    return {i: sum(len(p.findall(t)) for p in KEYWORD_PATTERNS[i]) for i in INTENTS}


class KeywordRouter:
    version = "keyword_v1"
    threshold = 0.0

    def predict(self, text: str) -> RouterResult:
        t = normalize((text or "")[:MAX_CHARS])
        injection = any(p.search(t) for p in INJECTION_PATTERNS)
        words = re.findall(r"[a-z]+", t)
        es, pt = sum(w in ES_WORDS for w in words), sum(w in PT_WORDS for w in words)
        language = "pt" if pt > es else "es"
        hits = intent_hits(text)
        total = sum(hits.values())
        if total == 0:
            return RouterResult("oos_other", 0.0, language, injection, float(injection), True, self.version)
        best = max(INTENTS, key=lambda i: (hits[i], -INTENTS.index(i)))
        return RouterResult(best, hits[best] / total, language, injection, float(injection), False, self.version)

    def predict_many(self, texts: list[str]) -> list[RouterResult]:
        return [self.predict(t) for t in texts]
