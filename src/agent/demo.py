"""One real sandbox customer per demo scenario, so judges can try every path without knowing the data."""
from src.bank import db

ACTIVE = ("u.customer_status = 'Active' and u.mobile_phone is not null and u.customer_id not in "
          "(select customer_id from complaint_flags where is_repeat_complainer = 1)")
CARD = "p.product_type like 'Tarjeta%' and p.product_status = 'Active'"
USD = "coalesce(t.amount_usd, case when t.currency = 'USD' then t.amount end)"


def _amount(a: float, lang: str) -> str:
    s = f"{a:.2f}".removesuffix(".00")
    return s.replace(".", ",") if lang == "pt" else s


def demo_scenarios(conn) -> list[dict]:
    q = lambda sql: conn.execute(sql).fetchone()  # noqa: E731
    out = []
    with db.LOCK:
        r = q(f"""select u.document_number d, t.merchant_name m, t.amount a from transactions t
                  join products p on p.product_id = t.product_id join customers u on u.customer_id = t.customer_id
                  where {ACTIVE} and {CARD} and t.transaction_status = 'Approved' and t.merchant_name is not null
                    and t.local_date >= '2026-04-01' and {USD} < 300 and (select count(*) from transactions t2 where t2.customer_id = t.customer_id
                                         and t2.merchant_name = t.merchant_name) = 1
                  order by u.customer_id, t.transaction_id limit 1""")
        if r:
            out.append({"scenario": "normal", "document": r["d"],
                        "message_es": f"No reconozco un cargo de {_amount(r['a'], 'es')} en {r['m']}",
                        "message_pt": f"Não reconheço uma cobrança de {_amount(r['a'], 'pt')} na {r['m']}"})
        r = q(f"""select u.document_number d, t.merchant_name m from transactions t
                  join products p on p.product_id = t.product_id join customers u on u.customer_id = t.customer_id
                  where {ACTIVE} and {CARD} and t.transaction_status = 'Approved' and t.merchant_name is not null
                    and t.local_date >= '2026-04-01' and {USD} < 300
                  group by u.customer_id, t.merchant_name having count(*) between 2 and 3
                  order by u.customer_id limit 1""")
        if r:
            out.append({"scenario": "ambiguous", "document": r["d"],
                        "message_es": f"No reconozco un cargo en {r['m']}",
                        "message_pt": f"Não reconheço uma cobrança na {r['m']}"})
        r = q(f"""select u.document_number d, t.amount a from transactions t
                  join products p on p.product_id = t.product_id join customers u on u.customer_id = t.customer_id
                  where {ACTIVE} and {CARD} and t.transaction_status = 'Approved' and t.merchant_name is null
                    and t.local_date >= '2026-04-01' and {USD} > 600 and t.transaction_type = 'Withdrawal'
                  order by u.customer_id limit 1""")
        if r:
            out.append({"scenario": "large_amount", "document": r["d"],
                        "message_es": f"No reconozco un retiro de {_amount(r['a'], 'es')} en mi tarjeta",
                        "message_pt": f"Não reconheço um saque de {_amount(r['a'], 'pt')} no meu cartão"})
        r = q(f"""select u.document_number d from customers u join products p on p.customer_id = u.customer_id
                  where {ACTIVE} and {CARD} group by u.customer_id having count(*) = 1
                  order by u.customer_id limit 1""")
        if r:
            out.append({"scenario": "stolen_card", "document": r["d"],
                        "message_es": "Me robaron la tarjeta y hay 3 compras que no hice",
                        "message_pt": "Roubaram meu cartão e tem 3 compras que não fiz"})
        r = q("""select document_number d from customers where customer_status = 'Active' and mobile_phone is null
                 order by customer_id limit 1""")
        if r:
            out.append({"scenario": "no_phone", "document": r["d"],
                        "message_es": "No reconozco un cargo de mi tarjeta",
                        "message_pt": "Não reconheço uma cobrança do meu cartão"})
    return out
