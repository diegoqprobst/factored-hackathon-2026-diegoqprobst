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
