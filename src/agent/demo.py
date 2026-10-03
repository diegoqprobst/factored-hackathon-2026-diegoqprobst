"""Real sandbox customers per demo scenario, so judges can try every path without knowing the data.

Each scenario keeps a small pool of candidates. A candidate is skipped once it is used up — its transaction
already has an open dispute, the customer has a blocked card, or someone requested an OTP for the customer
in the current window (a second judge would otherwise invalidate the first one's code). Building the pool is
the slow part (full scans); the freshness checks are cheap point lookups, so they run on every request."""
from datetime import timedelta

from src.bank import clock, config, db

ACTIVE = ("u.customer_status = 'Active' and u.mobile_phone is not null and u.customer_id not in "
          "(select customer_id from complaint_flags where is_repeat_complainer = 1)")
CARD = "p.product_type like 'Tarjeta%' and p.product_status = 'Active'"
KIND = {"Withdrawal": ("retiro", "saque"), "Transfer": ("transferencia", "transferência"), "Payment": ("pago", "pagamento")}
USD = "coalesce(t.amount_usd, case when t.currency = 'USD' then t.amount end)"
POOL_SIZE = 12


def _amount(a: float, lang: str) -> str:
    s = f"{a:.2f}".removesuffix(".00")
    return s.replace(".", ",") if lang == "pt" else s


def _candidate(scenario, row, message_es, message_pt, txns=()):
    return {"scenario": scenario, "document": row["d"], "message_es": message_es, "message_pt": message_pt,
            "_customer_id": row["c"], "_txns": tuple(txns)}


def demo_pool(conn, size: int = POOL_SIZE) -> dict[str, list[dict]]:
    q = lambda sql: conn.execute(sql, (size,)).fetchall()  # noqa: E731
    pool: dict[str, list[dict]] = {}
    with db.LOCK:
        pool["normal"] = [
            _candidate("normal", r, f"No reconozco un cargo de {_amount(r['a'], 'es')} en {r['m']}",
                       f"Não reconheço uma cobrança de {_amount(r['a'], 'pt')} na {r['m']}", [r["t"]])
            for r in q(f"""select u.customer_id c, u.document_number d, t.transaction_id t, t.merchant_name m,
                                  t.amount a from transactions t
                  join products p on p.product_id = t.product_id join customers u on u.customer_id = t.customer_id
                  where {ACTIVE} and {CARD} and t.transaction_status = 'Approved' and t.merchant_name is not null
                    and t.local_date >= '2026-04-01' and {USD} < 300 and (select count(*) from transactions t2 where t2.customer_id = t.customer_id
                                         and t2.merchant_name = t.merchant_name) = 1
                  order by u.customer_id, t.transaction_id limit ?""")]
        pool["ambiguous"] = [
            _candidate("ambiguous", r, f"No reconozco un cargo en {r['m']}", f"Não reconheço uma cobrança na {r['m']}",
                       r["ts"].split(","))
            for r in q(f"""select u.customer_id c, u.document_number d, t.merchant_name m,
                                  group_concat(t.transaction_id) ts from transactions t
                  join products p on p.product_id = t.product_id join customers u on u.customer_id = t.customer_id
                  where {ACTIVE} and {CARD} and t.transaction_status = 'Approved' and t.merchant_name is not null
                    and t.local_date >= '2026-04-01' and {USD} < 300
                  group by u.customer_id, t.merchant_name having count(*) between 2 and 3
                  order by u.customer_id limit ?""")]
        pool["large_amount"] = [
            _candidate("large_amount", r, f"No reconozco un {KIND[r['k']][0]} de {_amount(r['a'], 'es')} en mi tarjeta",
                       f"Não reconheço um {KIND[r['k']][1]} de {_amount(r['a'], 'pt')} no meu cartão", [r["t"]])
            for r in q(f"""select u.customer_id c, u.document_number d, t.transaction_id t, t.amount a,
                                  t.transaction_type k from transactions t
                  join products p on p.product_id = t.product_id join customers u on u.customer_id = t.customer_id
                  where {ACTIVE} and {CARD} and t.transaction_status = 'Approved' and t.merchant_name is null
                    and t.local_date >= '2026-04-01' and {USD} > 600
                    and t.transaction_type in ('Withdrawal', 'Transfer', 'Payment')
                  order by u.customer_id limit ?""")]
        pool["stolen_card"] = [
            _candidate("stolen_card", r, "Me robaron la tarjeta y hay 3 compras que no hice",
                       "Roubaram meu cartão e tem 3 compras que não fiz")
            for r in q(f"""select u.customer_id c, u.document_number d from customers u
                  join products p on p.customer_id = u.customer_id
                  where {ACTIVE} and {CARD} group by u.customer_id having count(*) = 1
                  order by u.customer_id limit ?""")]
        pool["no_phone"] = [
            _candidate("no_phone", r, "No reconozco un cargo de mi tarjeta", "Não reconheço uma cobrança do meu cartão")
            for r in q("""select customer_id c, document_number d from customers
                          where customer_status = 'Active' and mobile_phone is null order by customer_id limit ?""")]
    return {k: v for k, v in pool.items() if v}


def _fresh(conn, cand: dict, window_start: str) -> bool:
    cid, txns = cand["_customer_id"], cand["_txns"]
    if cand["scenario"] == "no_phone":  # never reaches OTP or any write
        return True
    if conn.execute("select 1 from card_blocks where customer_id = ? limit 1", (cid,)).fetchone():
        return False
    if conn.execute("select 1 from otp_challenges where customer_id = ? and created_at >= ? limit 1",
                    (cid, window_start)).fetchone():
        return False
    return not (txns and conn.execute(
        f"select 1 from disputes where customer_id = ? and status = 'open' and transaction_id in "
        f"({','.join('?' * len(txns))}) limit 1", (cid, *txns)).fetchone())


def demo_scenarios(conn, pool: dict[str, list[dict]] | None = None) -> list[dict]:
    """First still-fresh candidate per scenario, public fields only; a used-up scenario is omitted."""
    pool = demo_pool(conn) if pool is None else pool
    window_start = (clock.now() - timedelta(seconds=config.SESSION_TTL_SECONDS)).isoformat()
    out = []
    with db.LOCK:
        for cands in pool.values():
            cand = next((c for c in cands if _fresh(conn, c, window_start)), None)
            if cand:
                out.append({k: v for k, v in cand.items() if not k.startswith("_")})
    return out
