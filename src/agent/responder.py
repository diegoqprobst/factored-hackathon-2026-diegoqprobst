"""Customer-facing text. Every reply is a template filled with facts the bank services verified; the LLM never
writes prose the customer sees, so amounts, case ids and policy outcomes cannot be hallucinated."""

TEMPLATES = {
    "es": {
        "ask_document": "Hola, con gusto te ayudo. Antes de revisar tus movimientos necesito verificar tu identidad: ¿cuál es tu número de documento?",
        "otp_sent": "Te envié un código de 6 dígitos por SMS al número {masked}. Escríbelo aquí, por favor.",
        "ask_otp": "Necesito el código de 6 dígitos que te llegó por SMS.",
        "otp_invalid": "Ese código no coincide. Revísalo e inténtalo de nuevo.",
        "otp_expired": "El código venció. Escríbeme de nuevo tu número de documento y te envío uno nuevo.",
        "ask_charge": "Listo, ya verifiqué tu identidad. ¿Qué cargo quieres revisar? El monto, la fecha o el comercio me ayudan a encontrarlo.",
        "not_found": "No encontré ese movimiento en los últimos 120 días. ¿Me das otro dato, como el monto exacto, la fecha o el comercio?",
        "choose": "Encontré varios movimientos que coinciden:\n{options}\n¿Cuál es? Respóndeme con el número.",
        "choose_card": "Tienes varias tarjetas activas:\n{options}\n¿Cuál quieres bloquear? Respóndeme con el número.",
        "ask_type": "Encontré {txn}. ¿Qué pasó con este cargo?\n1. No lo reconozco\n2. Me cobraron dos veces\n3. El monto es distinto\n4. Es una comisión o cargo que no corresponde\n5. No me llegó un reembolso",
        "confirm_dispute": "{lead}Voy a abrir una disputa por {txn}, motivo: {dispute_label}. ¿Confirmas? (sí/no)",
        "dispute_created": "Listo: abrí la disputa {dispute_id} por {txn}. El banco la revisará y te avisará del resultado. ¿Te ayudo con algo más?",
        "dispute_created_offer_block": "Listo: abrí la disputa {dispute_id} por {txn}. Como no reconoces el cargo, te recomiendo bloquear la tarjeta terminada en {last4} para evitar más cargos. ¿La bloqueo? (sí/no)",
        "offer_block": "{lead}¿Quieres que bloquee la tarjeta terminada en {last4}? Quedará inutilizable y podrás pedir una nueva. (sí/no)",
        "card_blocked": "Listo: bloqueé la tarjeta terminada en {last4}. ¿Te ayudo con algo más?",
        "lead_card_blocked": "Bloqueé la tarjeta terminada en {last4}. ",
        "lead_one_at_a_time": "Revisemos un cargo a la vez; cuando terminemos con este te ayudo con el otro. ",
        "block_declined": "Entendido, no bloqueé la tarjeta. ¿Te ayudo con algo más?",
        "cancelled": "Entendido, no hice ningún cambio. ¿Te ayudo con algo más?",
        "ineligible": "No puedo abrir una disputa por este cargo porque {ineligible_reason}. ¿Te ayudo con algo más?",
        "handoff": "{lead}Voy a pasar tu caso a un asesor porque {handoff_reason}. Ya le envié lo que verificamos, así que no tendrás que repetirlo. Tu referencia es {handoff_id}.",
        "already_handed_off": "Tu caso ya está con un asesor (referencia {handoff_id}). Te contactarán pronto.",
        "out_of_scope": "Por este canal solo puedo ayudarte a disputar cargos y a bloquear tarjetas. Para {topic_label}, usa la app o la línea de atención.",
        "clarify": "No estoy seguro de haberte entendido. ¿Quieres disputar un cargo, bloquear una tarjeta o hablar con un asesor?",
        "greeting": "¡Hola! Puedo ayudarte a disputar un cargo que no reconoces o a bloquear una tarjeta. ¿Qué necesitas?",
        "goodbye": "Con gusto. ¡Que tengas un buen día!",
        "anything_else": "¿Te ayudo con algo más?",
        "injection_refused": "No puedo seguir esas instrucciones. Puedo ayudarte con los cargos de tu propia cuenta.",
        "session_expired": "Tu sesión se cerró por seguridad. Para continuar, escríbeme de nuevo tu número de documento.",
    },
    "pt": {
        "ask_document": "Olá, posso ajudar. Antes de ver suas movimentações preciso confirmar sua identidade: qual é o número do seu documento?",
        "otp_sent": "Enviei um código de 6 dígitos por SMS para o número {masked}. Digite aqui, por favor.",
        "ask_otp": "Preciso do código de 6 dígitos que chegou por SMS.",
        "otp_invalid": "Esse código não confere. Confira e tente de novo.",
        "otp_expired": "O código expirou. Me envie de novo o número do seu documento e mando um novo.",
        "ask_charge": "Pronto, identidade confirmada. Qual cobrança você quer revisar? O valor, a data ou a loja me ajudam a encontrá-la.",
        "not_found": "Não encontrei essa movimentação nos últimos 120 dias. Pode me dar outro dado, como o valor exato, a data ou a loja?",
        "choose": "Encontrei várias movimentações parecidas:\n{options}\nQual é? Responda com o número.",
        "choose_card": "Você tem vários cartões ativos:\n{options}\nQual quer bloquear? Responda com o número.",
        "ask_type": "Encontrei {txn}. O que aconteceu com essa cobrança?\n1. Não reconheço\n2. Fui cobrado duas vezes\n3. O valor está diferente\n4. É uma tarifa ou cobrança indevida\n5. Não recebi um reembolso",
        "confirm_dispute": "{lead}Vou abrir uma contestação de {txn}, motivo: {dispute_label}. Confirma? (sim/não)",
        "dispute_created": "Pronto: abri a contestação {dispute_id} de {txn}. O banco vai analisar e avisar o resultado. Posso ajudar em algo mais?",
        "dispute_created_offer_block": "Pronto: abri a contestação {dispute_id} de {txn}. Como você não reconhece a cobrança, recomendo bloquear o cartão final {last4} para evitar novas cobranças. Bloqueio? (sim/não)",
        "offer_block": "{lead}Quer que eu bloqueie o cartão final {last4}? Ele deixará de funcionar e você poderá pedir outro. (sim/não)",
        "card_blocked": "Pronto: bloqueei o cartão final {last4}. Posso ajudar em algo mais?",
        "lead_card_blocked": "Bloqueei o cartão final {last4}. ",
        "lead_one_at_a_time": "Vamos ver uma cobrança por vez; quando terminarmos esta, ajudo com a outra. ",
        "block_declined": "Entendido, não bloqueei o cartão. Posso ajudar em algo mais?",
        "cancelled": "Entendido, não fiz nenhuma alteração. Posso ajudar em algo mais?",
        "ineligible": "Não posso abrir uma contestação dessa cobrança porque {ineligible_reason}. Posso ajudar em algo mais?",
        "handoff": "{lead}Vou passar seu caso para um atendente porque {handoff_reason}. Já enviei o que verificamos, então você não precisará repetir. Seu protocolo é {handoff_id}.",
        "already_handed_off": "Seu caso já está com um atendente (protocolo {handoff_id}). Entrarão em contato em breve.",
        "out_of_scope": "Por este canal só posso ajudar a contestar cobranças e bloquear cartões. Para {topic_label}, use o app ou a central de atendimento.",
        "clarify": "Não tenho certeza se entendi. Você quer contestar uma cobrança, bloquear um cartão ou falar com um atendente?",
        "greeting": "Olá! Posso ajudar a contestar uma cobrança que você não reconhece ou a bloquear um cartão. Do que você precisa?",
        "goodbye": "Por nada. Tenha um ótimo dia!",
        "anything_else": "Posso ajudar em algo mais?",
        "injection_refused": "Não posso seguir essas instruções. Posso ajudar com as cobranças da sua própria conta.",
        "session_expired": "Sua sessão foi encerrada por segurança. Para continuar, me envie de novo o número do seu documento.",
    },
}

DISPUTE_LABEL = {
    "es": {"unrecognized": "cargo no reconocido", "duplicate": "cobro duplicado",
           "amount_mismatch": "monto distinto al de la compra", "undue_fee": "comisión o cargo indebido",
           "refund_not_received": "reembolso no recibido"},
    "pt": {"unrecognized": "cobrança não reconhecida", "duplicate": "cobrança duplicada",
           "amount_mismatch": "valor diferente da compra", "undue_fee": "tarifa ou cobrança indevida",
           "refund_not_received": "reembolso não recebido"},
}
INELIGIBLE = {
    "es": {"P1": "pasaron más de 90 días desde el cargo, que es el plazo para disputarlo",
           "P2": "esa transacción fue rechazada, así que no se te cobró",
           "P3": "ese cargo ya fue reversado", "P7": "ya existe una disputa abierta por este cargo ({existing})"},
    "pt": {"P1": "passaram mais de 90 dias desde a cobrança, que é o prazo para contestar",
           "P2": "essa transação foi recusada, então você não foi cobrado",
           "P3": "essa cobrança já foi estornada", "P7": "já existe uma contestação aberta dessa cobrança ({existing})"},
}
HANDOFF_REASON = {
    "es": {"customer_requested_human": "pediste hablar con un asesor",
           "refund_or_credit_requested": "los reembolsos y abonos los aprueba un asesor",
           "repeated_manipulation": "no puedo continuar esta conversación de forma automática",
           "not_understood": "no logré entender bien tu solicitud",
           "auth_no_verified_channel": "no tenemos un celular registrado para verificar tu identidad",
           "auth_account_not_serviceable": "tu cuenta requiere atención personalizada",
           "auth_rate_limited": "hubo demasiados intentos de verificación",
           "identity_not_verified": "no pudimos verificar tu identidad",
           "charge_not_found": "no logré ubicar el cargo", "could_not_identify_charge": "no logré identificar cuál es el cargo",
           "policy_P4": "el monto del cargo requiere revisión de un asesor",
           "policy_P5": "reportas varios cargos y eso requiere revisión de un asesor",
           "policy_P6": "tu caso requiere revisión de un asesor",
           "stolen_card_with_charges": "reportaste cargos con una tarjeta robada y eso lo revisa el equipo de fraude",
           "no_active_card": "no encontré una tarjeta activa para bloquear",
           "action_failed": "no pude completar la operación en este momento",
           "action_not_verified": "no pude confirmar que la operación quedara registrada"},
    "pt": {"customer_requested_human": "você pediu para falar com um atendente",
           "refund_or_credit_requested": "reembolsos e créditos são aprovados por um atendente",
           "repeated_manipulation": "não posso continuar esta conversa de forma automática",
           "not_understood": "não consegui entender bem sua solicitação",
           "auth_no_verified_channel": "não temos um celular cadastrado para confirmar sua identidade",
           "auth_account_not_serviceable": "sua conta precisa de atendimento personalizado",
           "auth_rate_limited": "houve tentativas demais de verificação",
           "identity_not_verified": "não conseguimos confirmar sua identidade",
           "charge_not_found": "não consegui localizar a cobrança",
           "could_not_identify_charge": "não consegui identificar qual é a cobrança",
           "policy_P4": "o valor da cobrança precisa da análise de um atendente",
           "policy_P5": "você relata várias cobranças e isso precisa da análise de um atendente",
           "policy_P6": "seu caso precisa da análise de um atendente",
           "stolen_card_with_charges": "você relatou cobranças com um cartão roubado e isso é analisado pela equipe de fraude",
           "no_active_card": "não encontrei um cartão ativo para bloquear",
           "action_failed": "não consegui concluir a operação agora",
           "action_not_verified": "não consegui confirmar que a operação foi registrada"},
}
GENERIC_REASON = {"es": "tu caso requiere atención de un asesor", "pt": "seu caso precisa de um atendente"}
TOPIC = {
    "es": {"oos_balance_movements": "consultar saldo o movimientos", "oos_credit": "créditos o préstamos"},
    "pt": {"oos_balance_movements": "consultar saldo ou movimentações", "oos_credit": "crédito ou empréstimos"},
}
GENERIC_TOPIC = {"es": "otros trámites", "pt": "outros serviços"}
CARD_WORD = {"es": "tarjeta", "pt": "cartão"}


def _txn(v: dict, lang: str) -> str:
    card = f" ({CARD_WORD[lang]} •{v['card_last4']})" if v.get("card_last4") else ""
    money = f"{v['amount']:,.2f} {v['currency']}"
    if lang == "pt":
        return f"{v['description']} de {money} em {v['local_date']}{card}"
    return f"{v['description']} por {money} el {v['local_date']}{card}"


def _options(options: list[dict], lang: str, kind: str) -> str:
    if kind == "card":
        return "\n".join(f"{i}. {o['product_type']} •{o['last4']}" for i, o in enumerate(options, 1))
    return "\n".join(f"{i}. {_txn(o, lang)}" for i, o in enumerate(options, 1))


def render(key: str, language: str, **facts) -> str:
    lang = language if language in TEMPLATES else "es"
    f = dict(facts)
    if isinstance(f.get("txn"), dict):
        f["txn"] = _txn(f["txn"], lang)
    if "options" in f:
        f["options"] = _options(f["options"], lang, f.get("kind", "transaction"))
    f["dispute_label"] = DISPUTE_LABEL[lang].get(f.get("dispute_type"), "")
    f["ineligible_reason"] = INELIGIBLE[lang].get(f.get("rule"), GENERIC_REASON[lang]).format(existing=f.get("existing", ""))
    f["handoff_reason"] = HANDOFF_REASON[lang].get(f.get("reason"), GENERIC_REASON[lang])
    f["topic_label"] = TOPIC[lang].get(f.get("topic"), GENERIC_TOPIC[lang])
    f["lead"] = TEMPLATES[lang][f["lead_key"]].format(**f) if f.get("lead_key") else ""
    return TEMPLATES[lang][key].format(**f)
