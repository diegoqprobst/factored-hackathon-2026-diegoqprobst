"""Idempotent S3 -> data/raw mirror with a lineage manifest.

Usage: uv run --env-file .env python -m src.pipeline.download [table ...] [--exclude table ...]
Skips files already present with the same size. Writes data/raw/_manifest.csv
(key, size, etag, last_modified, local_path) so every downstream artifact can be traced
back to the exact S3 object version it came from.
"""
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config

PREFIX = "data/"
RAW = Path("data/raw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="*", help="only these tables (default: all)")
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3", config=Config(max_pool_connections=32, retries={"max_attempts": 5}))

    objs = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=PREFIX):
        for o in page.get("Contents", []):
            table = o["Key"][len(PREFIX):].split("/")[0].removesuffix(".csv")
            if args.tables and table not in args.tables:
                continue
            if table in args.exclude:
                continue
            objs.append(o)

    def fetch(o):
        dest = RAW / o["Key"][len(PREFIX):]
        if dest.exists() and dest.stat().st_size == o["Size"]:
            return o, dest, False
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, o["Key"], str(dest))
        return o, dest, True

    RAW.mkdir(parents=True, exist_ok=True)
    rows, new = [], 0
    with ThreadPoolExecutor(24) as ex:
        for i, (o, dest, downloaded) in enumerate(ex.map(fetch, objs), 1):
            new += downloaded
            rows.append([o["Key"], o["Size"], o["ETag"].strip('"'), o["LastModified"].isoformat(), str(dest)])
            if i % 500 == 0:
                print(f"{i}/{len(objs)}", flush=True)

    manifest = RAW / "_manifest.csv"
    existing = {}
    if manifest.exists():
        with manifest.open() as f:
            existing = {r[0]: r for r in csv.reader(f)}
        existing.pop("s3_key", None)
    existing.update({r[0]: r for r in rows})
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s3_key", "size", "etag", "last_modified", "local_path"])
        w.writerows(sorted(existing.values()))
    print(f"{len(objs)} objects, {new} downloaded, manifest -> {manifest}")


if __name__ == "__main__":
    main()
