#!/usr/bin/env python3
"""Download fixed-strike Sensex expired option 5m OHLC from Dhan rollingoption API."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.dhan_data_client import DhanDataClient

DEFAULT_OUT_DIR = ROOT / "data" / "sensex" / "contracts"

CSV_COLUMNS = [
    "symbol",
    "expiry_date",
    "kind",
    "strike",
    "datetime_ist",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "oi",
    "spot",
    "rolling_offset",
    "data_source",
]


def download_contract(
    *,
    expiry_date: str,
    strike: int,
    kind: str,
    out_dir: Path,
    interval: str = "5",
) -> Path:
    client = DhanDataClient()
    prof = client.profile()
    if str(prof.get("dataPlan") or "").lower() != "active":
        raise RuntimeError("Dhan Data API is not active. Set DHAN_ACCESS_TOKEN in .env")

    kind = kind.upper()
    symbol = f"SENSEX-{strike}-{kind}"
    bars = client.fetch_expired_fixed_strike_bars(
        session_date=expiry_date,
        strike=strike,
        kind=kind,
        interval=interval,
    )
    if not bars:
        raise RuntimeError(
            f"No Dhan bars found for {symbol} on {expiry_date}. "
            "Expired contracts use rollingoption (ATM±offsets), not absolute securityId."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = expiry_date.replace("-", "")
    csv_path = out_dir / f"{tag}_{symbol}_dhan_{interval}m.csv"
    json_path = out_dir / f"{tag}_{symbol}_dhan_{interval}m.json"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for bar in bars:
            row = {k: v for k, v in bar.items() if k != "timestamp"}
            writer.writerow(
                {
                    "symbol": symbol,
                    "expiry_date": expiry_date,
                    "kind": kind,
                    "strike": strike,
                    "data_source": "dhan_rollingoption_fixed_strike_merge",
                    **row,
                }
            )

    json_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "expiry_date": expiry_date,
                "kind": kind,
                "strike": strike,
                "interval_min": interval,
                "data_source": "dhan_rollingoption_fixed_strike_merge",
                "dhan_profile": {"dataPlan": prof.get("dataPlan"), "data_validity": prof.get("dataValidity")},
                "bars": len(bars),
                "rows": bars,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download fixed-strike Sensex option OHLC from Dhan")
    parser.add_argument("--expiry", default="2026-06-04", help="Expiry session date YYYY-MM-DD")
    parser.add_argument("--strike", type=int, default=74100)
    parser.add_argument("--kind", default="PE", choices=["CE", "PE"])
    parser.add_argument("--interval", default="5", choices=["1", "5", "15", "25", "60"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    path = download_contract(
        expiry_date=args.expiry,
        strike=args.strike,
        kind=args.kind,
        out_dir=args.out_dir,
        interval=args.interval,
    )
    with path.open(encoding="utf-8") as fh:
        n = sum(1 for _ in csv.DictReader(fh))
    print(f"Downloaded {n} bars → {path}")
    print(f"JSON cache → {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
