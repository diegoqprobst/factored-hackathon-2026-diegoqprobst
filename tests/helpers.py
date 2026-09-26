import csv
import re
from pathlib import Path


def last_code(conn) -> str:
    body = conn.execute("select body from sandbox_outbox order by id desc limit 1").fetchone()["body"]
    return re.search(r"(\d{6})", body).group(1)


def login(conn, document_number: str) -> str:
    from src.bank import auth

    ch = auth.start_auth(conn, document_number)
    return auth.verify_otp(conn, ch.challenge_id, last_code(conn))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig mimics the BOM present in the real S3 files
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


TXN_HEADER = ["transaction_id", "transaction_date", "process_date", "product_id", "customer_id",
              "transaction_type", "transaction_category", "amount", "currency", "amount_usd", "channel",
              "merchant_name", "merchant_category", "transaction_country", "transaction_city",
              "transaction_status", "is_fraud"]


def txn_row(tid, ts, day, pid="PRD-A1", cid="CLI-A", amount="350.00", status="Approved", usd="0.09",
            merchant="Oxxo"):
    return [tid, ts, day, pid, cid, "Purchase", "Food", amount, "COP", usd, "POS", merchant, "Grocery",
            "Colombia", "Bogotá", status, "False"]


def write_txn_day(raw: Path, day: str, rows: list[list]) -> None:
    y, m, d = day.split("-")
    write_csv(raw / "transactions" / f"year={y}" / f"month={m}" / f"day={d}" / f"transactions_{y}{m}{d}.csv",
              TXN_HEADER, rows)


def write_raw(raw: Path) -> None:
    """Small bronze fixture with one planted defect of each kind the contracts must catch."""
    write_csv(raw / "customers.csv",
              ["customer_id", "document_number", "document_type", "first_name", "last_name", "mobile_phone",
               "country", "segment", "customer_status", "last_updated"],
              [["CLI-A", "111", "CC", "Ana", "Gómez", "+57 300 000 1234", "Colombia", "Basic", "Active", "2026-06-01 00:00:00"],
               ["CLI-A", "111", "CC", "Ana", "Gómez", "+57 300 000 1234", "Colombia", "Basic", "Inactive", "2026-01-01 00:00:00"],  # older duplicate
               ["CLI-B", "222", "DNI", "Beto", "Paz", "+54 9 11 0000 2222", "Argentina", "Plus", "Active", "2026-06-01 00:00:00"],
               ["CLI-BAD", "333", "XX", "Carla", "Ruiz", "", "Colombia", "Basic", "Active", "2026-06-01 00:00:00"],  # bad enum
               ["CLI-MISS", "", "CC", "Dana", "Soto", "", "Colombia", "Basic", "Active", "2026-06-01 00:00:00"]])  # missing doc
    write_csv(raw / "products.csv",
              ["product_id", "customer_id", "product_type", "product_number", "currency", "product_status", "last_updated"],
              [["PRD-A1", "CLI-A", "Tarjeta Crédito", "4000000000001111", "COP", "Active", "2026-06-01 00:00:00"],
               ["PRD-B1", "CLI-B", "Tarjeta Débito", "4000000000002222", "ARS", "Active", "2026-06-01 00:00:00"]])
    write_csv(raw / "daily_exchange_rates.csv",
              ["date", "source_currency", "target_currency", "exchange_rate"],
              [["2026-06-15", "COP", "USD", "0.00025"], ["2026-06-15", "USD", "COP", "4000"]])
    write_csv(raw / "complaints" / "year=2026" / "month=05" / "day=01" / "complaints_20260501.csv",
              ["complaint_id", "creation_date", "customer_id", "is_repeat_complainer"],
              [["CMP-1", "2026-05-01 10:00:00", "CLI-B", "True"], ["CMP-2", "2026-05-01 11:00:00", "CLI-B", "False"]])
    write_txn_day(raw, "2026-06-14", [txn_row("TRX-1", "2026-06-14 18:00:00", "2026-06-14")])
    write_txn_day(raw, "2026-06-15", [
        txn_row("TRX-2", "2026-06-15 18:00:00", "2026-06-15", usd=""),
        txn_row("TRX-3", "2026-06-15 18:00:00", "2026-06-15", cid="CLI-B"),       # product owned by CLI-A
        txn_row("TRX-4", "2026-06-15 18:00:00", "2026-06-15", amount="abc"),      # bad type
    ])
    write_txn_day(raw, "2026-06-16", [
        txn_row("TRX-5", "2026-06-16 18:00:00", "2026-06-16", merchant=""),
        txn_row("TRX-5", "2026-06-16 18:00:00", "2026-06-16", merchant=""),       # exact duplicate
        txn_row("TRX-6", "2026-06-16 18:00:00", "2026-06-16", pid="PRD-ZZ"),      # orphan product
        txn_row("TRX-7", "2026-06-16 18:00:00", "2026-06-16", status="Approvedd"),  # bad enum
        txn_row("TRX-8", "2026-06-16 03:00:00", "2026-06-16"),                    # breaks the UTC-6 rule
    ])
