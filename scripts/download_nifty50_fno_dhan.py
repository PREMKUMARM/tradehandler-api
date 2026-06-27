#!/usr/bin/env python3
"""Download Nifty 50 FnO rolling options OHLC from Dhan into local DuckDB."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from services.dhan_data_client import DhanDataClient, interval_to_str
from services.nifty_duckdb_store import (
    NIFTY_STRIKE_OFFSETS,
    STAGING_DB_PATH,
    NiftyDuckDBStore,
    parse_rolling_option_leg,
)

IST = ZoneInfo("Asia/Kolkata")
ROLLING_OPTION_MAX_DAYS = 30


def _parse_date(raw: str) -> date:
    return date.fromisoformat(raw.strip())


def iter_rolling_chunks(start: date, end: date, *, max_days: int = ROLLING_OPTION_MAX_DAYS) -> list[tuple[str, str]]:
    """Non-overlapping [from, to) windows for Dhan rollingoption (max ~30 days per call)."""
    chunks: list[tuple[str, str]] = []
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=max_days), end)
        chunks.append((cur.isoformat(), chunk_end.isoformat()))
        cur = chunk_end
    return chunks


def download_nifty_fno(
    *,
    days: int = 30,
    start_date: date | None = None,
    end_date: date | None = None,
    interval_min: int = 5,
    expiry_code: int = 1,
    expiry_flag: str = "WEEK",
    db_path: Path | None = None,
    force: bool = False,
    offsets: list[str] | None = None,
) -> dict:
    end = end_date or date.today()
    start = start_date or (end - timedelta(days=days))
    api_end = end + timedelta(days=1)

    client = DhanDataClient()
    prof = client.profile()
    if str(prof.get("dataPlan") or "").lower() != "active":
        raise RuntimeError("Dhan Data API is not active. Set DHAN_ACCESS_TOKEN in .env")

    store = NiftyDuckDBStore(db_path)
    staging_path: Path | None = None
    try:
        store.ensure_schema()
    except RuntimeError:
        staging_path = Path(db_path) if db_path else STAGING_DB_PATH
        print(f"Main DB locked by DuckDB extension — writing to staging: {staging_path}")
        store = NiftyDuckDBStore(staging_path)
        store.ensure_schema()

    strike_offsets = offsets or list(NIFTY_STRIKE_OFFSETS)
    interval = interval_to_str(interval_min)
    chunks = iter_rolling_chunks(start, api_end)
    kinds = ("CE", "PE")
    total_calls = len(chunks) * len(strike_offsets) * len(kinds)
    total_bars = 0
    call_no = 0

    print(
        f"Nifty50 FnO ingest: {start} -> {end} "
        f"({len(chunks)} chunks x {len(strike_offsets)} offsets x 2 kinds = {total_calls} API calls, {interval_min}m)"
    )
    print(f"DuckDB: {store.db_path}")

    if not force and store.has_fno_range(start.isoformat(), end.isoformat(), interval_min=interval_min, expiry_code=expiry_code):
        print("Range already cached (ATM CE coverage). Use --force to re-fetch.")
        summary = store.fno_summary()
        summary["bars_upserted_this_run"] = 0
        return summary

    conn = store.connect()
    try:
        for from_date, to_date in chunks:
            for offset in strike_offsets:
                for kind in kinds:
                    call_no += 1
                    print(
                        f"  [{call_no}/{total_calls}] {from_date}->{to_date} {offset} {kind} ...",
                        end=" ",
                        flush=True,
                    )
                    leg = client.nifty_rolling_option_range(
                        from_date=from_date,
                        to_date=to_date,
                        kind=kind,
                        offset=offset,
                        interval=interval,
                        expiry_code=expiry_code,
                        expiry_flag=expiry_flag,
                    )
                    rows = parse_rolling_option_leg(
                        leg,
                        kind=kind,
                        strike_offset=offset,
                        expiry_code=expiry_code,
                        expiry_flag=expiry_flag,
                        interval_min=interval_min,
                        fetched_at=datetime.now(IST),
                        start_date=start,
                        end_date=end,
                    )
                    if not rows:
                        print("0 bars")
                        continue
                    n = store.upsert_fno_bars(rows, conn=conn)
                    total_bars += n
                    print(f"{n} bars")
    finally:
        conn.close()

    summary = store.fno_summary()
    summary["bars_upserted_this_run"] = total_bars
    summary["api_calls"] = total_calls
    if staging_path:
        summary["staging_path"] = str(staging_path)
        summary["merge_hint"] = (
            f"Close DuckDB in Cursor, then run: "
            f"python3 scripts/merge_nifty50_staging.py"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Nifty50 FnO OHLC from Dhan into DuckDB")
    parser.add_argument("--days", type=int, default=30, help="Lookback calendar days (default: 30)")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--interval", type=int, default=5, choices=[1, 5, 15, 25, 60])
    parser.add_argument("--expiry-code", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--expiry-flag", type=str, default="WEEK", choices=["WEEK", "MONTH"])
    parser.add_argument("--db", type=str, default=None, help="DuckDB file path")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args()

    summary = download_nifty_fno(
        days=args.days,
        start_date=_parse_date(args.start) if args.start else None,
        end_date=_parse_date(args.end) if args.end else None,
        interval_min=args.interval,
        expiry_code=args.expiry_code,
        expiry_flag=args.expiry_flag,
        db_path=Path(args.db) if args.db else None,
        force=args.force,
    )

    print("\nDone.")
    print(f"  FnO bars in DB:  {summary['bar_count']}")
    print(f"  trading days:    {summary['trading_days']}")
    print(f"  strike offsets:  {summary['offsets']}")
    print(f"  date range:      {summary['first_date']} -> {summary['last_date']}")
    print(f"  upserted (run):  {summary['bars_upserted_this_run']}")
    print(f"  database:        {summary['db_path']}")
    if summary.get("staging_path"):
        print(f"\n  STAGING FILE:    {summary['staging_path']}")
        print(f"  {summary['merge_hint']}")


if __name__ == "__main__":
    main()
