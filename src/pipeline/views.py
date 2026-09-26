"""DuckDB views over the raw CSV mirror (data/raw). Single place that knows file layout."""
import duckdb

RAW = "data/raw"
FACTS = ["call_center_interactions", "call_transcripts", "complaints", "satisfaction_surveys", "transactions", "campaign_sends"]
DIMS = ["customers", "products", "service_agents", "branches", "marketing_campaigns", "daily_exchange_rates"]


def connect(raw: str = RAW) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in FACTS:
        con.sql(f"create view {t} as select * from read_csv('{raw}/{t}/**/*.csv', hive_partitioning=true, union_by_name=true, sample_size=-1)")
    for t in DIMS:
        con.sql(f"create view {t} as select * from read_csv('{raw}/{t}.csv', sample_size=-1)")
    return con
