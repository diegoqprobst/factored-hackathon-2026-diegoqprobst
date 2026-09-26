"""Data contracts for the tables the dispute workflow uses. Raw is read as VARCHAR; the contract
decides what is valid, casts it, and explains every rejection."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Col:
    name: str
    type: str = "VARCHAR"
    required: bool = False
    enum: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Contract:
    table: str
    pk: tuple[str, ...]
    columns: tuple[Col, ...]
    partitioned: bool
    order_by: str  # dedup tie-break: first row per PK in this order is kept

    def reject_reason_sql(self) -> str:
        parts = []
        for c in self.columns:
            q = f'"{c.name}"'
            present = f"nullif(trim({q}), '') is not null"
            if c.required:
                parts.append(f"case when not ({present}) then '{c.name}:missing' end")
            if c.type != "VARCHAR":
                parts.append(f"case when {present} and try_cast(trim({q}) as {c.type}) is null then '{c.name}:bad_type' end")
            if c.enum:
                vals = ", ".join("'" + v.replace("'", "''") + "'" for v in c.enum)
                parts.append(f"case when {present} and trim({q}) not in ({vals}) then '{c.name}:bad_enum' end")
        return "nullif(concat_ws('; ', " + ", ".join(parts) + "), '')"

    def select_sql(self) -> str:
        return ", ".join(f"try_cast(nullif(trim(\"{c.name}\"), '') as {c.type}) as \"{c.name}\"" for c in self.columns)


CURRENCIES = ("MXN", "COP", "ARS", "USD")
TS = "TIMESTAMP"

CONTRACTS = (
    Contract("customers", ("customer_id",), (
        Col("customer_id", required=True), Col("document_number", required=True),
        Col("document_type", required=True, enum=("DNI", "CURP", "CC", "CE", "Pasaporte", "Passport")),
        Col("first_name", required=True), Col("last_name", required=True), Col("mobile_phone"),
        Col("country", required=True, enum=("México", "Colombia", "Argentina")),
        Col("segment", required=True, enum=("Premium", "Plus", "Basic", "Student")),
        Col("customer_status", required=True, enum=("Active", "Inactive", "Suspended", "Closed")),
        Col("last_updated", TS, required=True),
    ), partitioned=False, order_by="last_updated desc"),
    Contract("products", ("product_id",), (
        Col("product_id", required=True), Col("customer_id", required=True),
        Col("product_type", required=True, enum=("Cuenta Ahorro", "Cuenta Corriente", "Tarjeta Crédito",
                                                  "Tarjeta Débito", "Préstamo Personal", "Préstamo Hipotecario",
                                                  "Inversión", "Seguro")),
        Col("product_number", required=True), Col("currency", required=True, enum=CURRENCIES),
        Col("product_status", required=True, enum=("Active", "Blocked", "Closed", "Suspended")),
        Col("last_updated", TS, required=True),
    ), partitioned=False, order_by="last_updated desc"),
    Contract("transactions", ("transaction_id",), (
        Col("transaction_id", required=True), Col("transaction_date", TS, required=True),
        Col("process_date", "DATE", required=True), Col("product_id", required=True), Col("customer_id", required=True),
        Col("transaction_type", required=True, enum=("Deposit", "Withdrawal", "Transfer", "Payment", "Purchase", "Adjustment")),
        Col("amount", "DOUBLE", required=True), Col("currency", required=True, enum=CURRENCIES),
        Col("amount_usd", "DOUBLE"),
        Col("channel", required=True, enum=("ATM", "Branch", "Web", "App", "POS", "Transfer")),
        Col("merchant_name"), Col("merchant_category"), Col("transaction_city"),
        Col("transaction_country", required=True),
        Col("transaction_status", required=True, enum=("Approved", "Declined", "Pending", "Reversed")),
        Col("is_fraud", "BOOLEAN", required=True),
    ), partitioned=True, order_by="process_date desc"),
    Contract("complaints", ("complaint_id",), (
        Col("complaint_id", required=True), Col("creation_date", TS, required=True),
        Col("customer_id", required=True), Col("is_repeat_complainer", "BOOLEAN", required=True),
    ), partitioned=True, order_by="creation_date desc"),
    Contract("daily_exchange_rates", ("date", "source_currency", "target_currency"), (
        Col("date", "DATE", required=True), Col("source_currency", required=True, enum=CURRENCIES),
        Col("target_currency", required=True, enum=CURRENCIES), Col("exchange_rate", "DOUBLE", required=True),
    ), partitioned=False, order_by="date"),
)
