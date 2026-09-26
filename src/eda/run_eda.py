"""Reproducible EDA: runs every check behind reports/eda_findings.md and dumps results to reports/eda_output.md.

Usage: uv run python -m src.eda.run_eda
"""
from pathlib import Path

from src.pipeline.views import connect

CHECKS = {
    "Row counts vs dictionary (and PK uniqueness)": """
        select 'call_center_interactions' t, 800000 documented, count(*) n, count(distinct interaction_id) uniq from call_center_interactions
        union all select 'call_transcripts', 200000, count(*), count(distinct transcript_id) from call_transcripts
        union all select 'complaints', 80000, count(*), count(distinct complaint_id) from complaints
        union all select 'transactions', 5000000, count(*), count(distinct transaction_id) from transactions
        union all select 'satisfaction_surveys', 250000, count(*), count(distinct survey_id) from satisfaction_surveys
        union all select 'customers', 150000, count(*), count(distinct customer_id) from customers
        union all select 'products', 400000, count(*), count(distinct product_id) from products""",
    "Contact demand by reason (contact_reason == reason_category)": """
        select reason_category, count(*) n, round(100*count(*)/sum(count(*)) over (),1) pct,
               round(avg(was_resolved::int),3) fcr, round(avg(was_escalated::int),3) escalated,
               round(avg(requires_followup::int),3) followup, median(duration_seconds) dur_median_s,
               count(distinct contact_reason) distinct_reasons
        from call_center_interactions group by all order by n desc""",
    "Contact demand by channel": """
        select channel, count(*) n, round(avg(was_resolved::int),3) fcr, median(wait_time_seconds) wait_median_s
        from call_center_interactions group by all order by n desc""",
    "Contact volume by year": "select year(interaction_date) y, count(*) n from call_center_interactions group by 1 order by 1",
    "Complaints by category/subcategory (top)": """
        select category, subcategory, count(*) n, round(avg(sla_breached::int),3) sla_breach,
               median(resolution_days) res_days_median, round(avg(claimed_amount)) claimed_avg,
               round(avg(compensation_granted)) comp_avg
        from complaints group by all order by n desc limit 12""",
    "Complaints by case_type (compensation on Requests/Suggestions is suspicious)": """
        select case_type, count(*) n, count(claimed_amount) with_claim, count(compensation_granted) with_comp
        from complaints group by all order by n desc""",
    "Complaint status": "select status, count(*) n from complaints group by all order by n desc",
    "Complaint linkage & cross-customer integrity": """
        select count(*) complaints, count(origin_interaction_id) with_origin_interaction,
               count(affected_product_id) with_product, count(p.product_id) product_exists,
               count(case when p.customer_id = c.customer_id then 1 end) product_owned_by_complainant
        from complaints c left join products p on c.affected_product_id = p.product_id""",
    "Transcript diversity (templated text)": """
        select count(*) n, count(distinct full_text) distinct_texts,
               count(distinct split_part(full_text, chr(10), 1)) distinct_opening_lines,
               count(distinct detected_intents) distinct_intents, count(distinct detected_language) languages,
               count(case when full_text like '%{monto}%' then 1 end) unfilled_placeholders
        from call_transcripts""",
    "Transcript opening line vs labelled topic": """
        select split_part(full_text, chr(10), 1) opening_line, main_topics, count(*) n
        from call_transcripts group by all order by 1, n desc""",
    "Transactions: status x fraud label": """
        select transaction_status, is_fraud, count(*) n, round(avg(fraud_score),1) fraud_score_avg
        from transactions group by all order by n desc""",
    "Transactions: fraud_score vs label correlation": "select round(corr(fraud_score, is_fraud::int),3) corr from transactions",
    "Transactions: product ownership consistency": """
        select count(*) n, count(*) - count(p.product_id) orphan_product,
               count(case when p.customer_id <> t.customer_id then 1 end) owner_mismatch
        from transactions t left join products p using (product_id)""",
    "Transactions: event date vs partition date": """
        select min(transaction_date) min_ts, max(transaction_date) max_ts,
               count(case when transaction_date::date <> process_date then 1 end) date_ne_partition, count(*) n
        from transactions""",
    "Customers: identity attributes uniqueness": """
        select count(*) n, count(distinct document_number) documents, count(distinct lower(email)) emails,
               count(distinct mobile_phone) phones from customers""",
    "Customers by country/segment": "select country, segment, count(*) n from customers group by all order by 1, 3 desc",
    "Products by type/status (top)": "select product_type, product_status, count(*) n from products group by all order by n desc limit 15",
    "Nulls in call_center_interactions": """
        select count(*) n, count(*)-count(duration_seconds) duration_null, count(*)-count(customer_detected_accent) accent_null,
               count(*)-count(sentiment_score) sentiment_null, count(*)-count(wait_time_seconds) wait_null
        from call_center_interactions""",
}


def main():
    con = connect()
    out = ["# EDA output (auto-generated by src/eda/run_eda.py — do not edit)\n"]
    for title, q in CHECKS.items():
        print("running:", title, flush=True)
        out += [f"## {title}\n", "```sql", q.strip(), "```\n", con.sql(q).df().to_markdown(index=False), "\n"]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/eda_output.md").write_text("\n".join(out))
    print("-> reports/eda_output.md")


if __name__ == "__main__":
    main()
