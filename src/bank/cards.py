"""Card lookups and blocking, scoped to the authenticated customer. Blocking needs explicit confirmation,
is idempotent, and returns the state read back from storage."""
import secrets
import sqlite3
from dataclasses import dataclass

from src.bank import audit, clock, db
from src.bank.auth import verify_session
from src.bank.errors import ConfirmationRequired, InvalidProduct, NotFound

CARD_TYPES = frozenset({"Tarjeta Crédito", "Tarjeta Débito"})


@dataclass(frozen=True)
class CardStatus:
    product_id: str
    product_type: str
    last4: str
    status: str


def _card(row) -> CardStatus:
    return CardStatus(row["product_id"], row["product_type"], row["product_number"][-4:], row["product_status"])


def list_cards(conn: sqlite3.Connection, token: str) -> list[CardStatus]:
    s = verify_session(token)
    rows = conn.execute("select * from products where customer_id = ? and product_type in (?, ?) order by product_id",
                        (s.customer_id, *sorted(CARD_TYPES))).fetchall()
    return [_card(r) for r in rows]


def get_card(conn: sqlite3.Connection, token: str, product_id: str) -> CardStatus:
    s = verify_session(token)
    row = conn.execute("select * from products where product_id = ? and customer_id = ?",
                       (product_id, s.customer_id)).fetchone()
    if row is None:
        if conn.execute("select 1 from products where product_id = ?", (product_id,)).fetchone():
            audit.log(conn, "cross_customer_access_attempt", customer_id=s.customer_id, session_id=s.session_id,
                      product_id=product_id)
        raise NotFound()
    if row["product_type"] not in CARD_TYPES:
        raise InvalidProduct("not a card")
    return _card(row)


def block_card(conn: sqlite3.Connection, token: str, product_id: str, *, customer_confirmed: bool,
               reason: str) -> CardStatus:
    s = verify_session(token)
    with db.LOCK:
        card = get_card(conn, token, product_id)
        if not customer_confirmed:
            raise ConfirmationRequired()
        if card.status == "Blocked":
            audit.log(conn, "card_block_noop", customer_id=s.customer_id, session_id=s.session_id,
                      product_id=product_id)
            return card
        if card.status == "Closed":
            raise InvalidProduct("card is closed")
        with conn:
            conn.execute("update products set product_status = 'Blocked' where product_id = ? and customer_id = ?",
                         (product_id, s.customer_id))
            conn.execute("insert into card_blocks values (?,?,?,?,?,?)",
                         ("BLK-" + secrets.token_hex(5).upper(), product_id, s.customer_id, s.session_id, reason,
                          clock.now().isoformat()))
        audit.log(conn, "card_blocked", customer_id=s.customer_id, session_id=s.session_id, product_id=product_id,
                  reason=reason)
        return get_card(conn, token, product_id)
