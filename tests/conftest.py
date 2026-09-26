from datetime import datetime, timedelta, timezone

import pytest

from src.bank import clock, db


class FrozenClock:
    def __init__(self, t: datetime):
        self.t = t

    def now(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


@pytest.fixture
def frozen(monkeypatch):
    fc = FrozenClock(datetime(2026, 6, 17, 15, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(clock, "now", fc.now)
    return fc


CUSTOMERS = [
    ("CLI-A", "111", "CC", "Ana", "Gómez", "+57 300 000 1234", "Colombia", "Basic", "Active"),
    ("CLI-B", "222", "DNI", "Beto", "Paz", "+54 9 11 0000 2222", "Argentina", "Plus", "Active"),
    ("CLI-NOPHONE", "333", "CC", "Carla", "Ruiz", None, "Colombia", "Basic", "Active"),
    ("CLI-CLOSED", "444", "CC", "Dana", "Soto", "+57 300 000 4444", "Colombia", "Basic", "Closed"),
]
PRODUCTS = [
    ("PRD-A-CC", "CLI-A", "Tarjeta Crédito", "4000000000001111", "COP", "Active"),
    ("PRD-A-SAV", "CLI-A", "Cuenta Ahorro", "0011223344", "COP", "Active"),
    ("PRD-A-OLD", "CLI-A", "Tarjeta Débito", "4000000000003333", "COP", "Closed"),
    ("PRD-B-DC", "CLI-B", "Tarjeta Débito", "4000000000002222", "USD", "Active"),
]


def _t(tid, cid, pid, day, amount, cur, usd, merchant, status, fraud=0, ttype="Purchase"):
    return (tid, cid, pid, f"{day}T18:00:00", day, ttype, amount, cur, usd, "POS", merchant, None,
            "Bogotá", "Colombia", status, fraud)


TRANSACTIONS = [
    _t("TRX-A1", "CLI-A", "PRD-A-CC", "2026-06-16", 350.0, "COP", 0.09, "Oxxo", "Approved"),
    _t("TRX-A2", "CLI-A", "PRD-A-CC", "2026-06-10", 120000.0, "COP", None, None, "Approved"),
    _t("TRX-A3", "CLI-A", "PRD-A-CC", "2026-01-05", 50.0, "COP", 0.01, "Tienda", "Approved"),
    _t("TRX-A4", "CLI-A", "PRD-A-CC", "2026-06-15", 80.0, "COP", 0.02, "Rappi", "Declined"),
    _t("TRX-A5", "CLI-A", "PRD-A-CC", "2026-06-14", 90.0, "COP", 0.02, "Rappi", "Reversed"),
    _t("TRX-A6", "CLI-A", "PRD-A-CC", "2026-06-12", 2500000.0, "COP", 625.0, "Falabella", "Approved"),
    _t("TRX-A7", "CLI-A", "PRD-A-CC", "2026-06-11", 990.0, "ARS", None, "Kiosco", "Approved"),
    _t("TRX-A8", "CLI-A", "PRD-A-CC", "2026-06-16", 350.0, "COP", 0.09, "Oxxo", "Approved"),
    _t("TRX-A9", "CLI-A", "PRD-A-CC", "2026-06-13", 45.0, "COP", 0.01, "Uber", "Approved"),
    _t("TRX-B1", "CLI-B", "PRD-B-DC", "2026-06-16", 40.0, "USD", None, "Amazon", "Approved"),
]
FX = [("2026-06-10", "COP", 0.00025)]
FLAGS = [("CLI-B", 2, 1)]


@pytest.fixture
def bank(tmp_path):
    conn = db.connect(tmp_path / "bank.db")
    db.create_schema(conn)
    conn.executemany("insert into customers values (?,?,?,?,?,?,?,?,?)", CUSTOMERS)
    conn.executemany("insert into products values (?,?,?,?,?,?)", PRODUCTS)
    conn.executemany("insert into transactions values (" + ",".join("?" * 16) + ")", TRANSACTIONS)
    conn.executemany("insert into fx_rates values (?,?,?)", FX)
    conn.executemany("insert into complaint_flags values (?,?,?)", FLAGS)
    conn.commit()
    yield conn
    conn.close()
