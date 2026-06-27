#!/usr/bin/env python3
"""Download Nifty 50 index 5m OHLC from Dhan into local DuckDB."""
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

from services.dhan_data_client import DhanDataClient, INTRADAY_MAX_DAYS, interval_to_str
from services.nifty_duckdb_store import NiftyDuckDBStore, iter_intraday_chunks, parse_intraday_response

IST = ZoneInfo("Asia/Kolkata")


def _parse_date(raw: str) -> date:
    return date.fromisoformat(raw.strip())


def download_nifty_index(
    *,
    years: int = 5,
    start_date: date | None = None,
    end_date: date | None = None,
    interval_min: int = 5,
    db_path: Path | None = None,
    force: bool = False,
) -> dict:
    end = end_date or date.today()
    start = start_date or (end - timedelta(days=int(years * 365.25)))

    client = DhanDataClient()
    prof = client.profile()
    if str(prof.get("dataPlan") or "").lower() != "active":
        raise RuntimeError("Dhan Data API is not active. Set DHAN_ACCESS_TOKEN in .env")

    store = NiftyDuckDBStore(db_path)
    store.ensure_schema()

    interval = interval_to_str(interval_min)
    chunks = iter_intraday_chunks(start, end, max_days=INTRADAY_MAX_DAYS)
    total_bars = 0
    empty_chunks = 0

    print(f"Nifty50 index ingest: {start} -> {end} ({len(chunks)} API chunks, {interval_min}m)")
    print(f"DuckDB: {store.db_path}")

    conn = store.connect()
    try:
        for idx, (from_date, to_date) in enumerate(chunks, start=1):
            if not force:
                # Skip chunk if we already have the last weekday in range covered.
                chunk_end_day = _parse_date(to_date) - timedelta(days=1)
                while chunk_end_day >= _parse_date(from_date) and chunk_end_day.weekday() >= 5:
                    chunk_end_day -= timedelta(days=1)
                if chunk_end_day >= _parse_date(from_date) and store.has_session(
                    chunk_end_day.isoformat(), interval_min
                ):
                    print(f"  [{idx}/{len(chunks)}] skip {from_date} -> {to_date} (cached)")
                    continue

            print(f"  [{idx}/{len(chunks)}] fetch {from_date} -> {to_date} ...", end=" ", flush=True)
            resp = client.nifty_index_intraday(
                from_date=from_date,
                to_date=to_date,
                interval=interval,
            )
            rows = parse_intraday_response(resp, interval_min=interval_min, fetched_at=datetime.now(IST))
            if not rows:
                empty_chunks += 1
                print("0 bars")
                continue

            n = store.upsert_bars(rows, conn=conn)
            total_bars += n
            first_day = rows[0]["session_date"]
            last_day = rows[-1]["session_date"]
            print(f"{n} bars ({first_day} .. {last_day})")
    finally:
        conn.close()

    summary = store.summary()
    summary["chunks_fetched"] = len(chunks) - empty_chunks
    summary["empty_chunks"] = empty_chunks
    summary["bars_upserted_this_run"] = total_bars
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Nifty50 index OHLC from Dhan into DuckDB")
    parser.add_argument("--years", type=int, default=5, help="Lookback years (default: 5)")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--interval", type=int, default=5, choices=[1, 5, 15, 25, 60])
    parser.add_argument("--db", type=str, default=None, help="DuckDB file path")
    parser.add_argument("--force", action="store_true", help="Re-fetch all chunks even if cached")
    args = parser.parse_args()

    summary = download_nifty_index(
        years=args.years,
        start_date=_parse_date(args.start) if args.start else None,
        end_date=_parse_date(args.end) if args.end else None,
        interval_min=args.interval,
        db_path=Path(args.db) if args.db else None,
        force=args.force,
    )

    print("\nDone.")
    print(f"  bars in DB:      {summary['bar_count']}")
    print(f"  trading days:    {summary['trading_days']}")
    print(f"  date range:      {summary['first_date']} -> {summary['last_date']}")
    print(f"  upserted (run):  {summary['bars_upserted_this_run']}")
    print(f"  database:        {summary['db_path']}")


if __name__ == "__main__":
    main()
