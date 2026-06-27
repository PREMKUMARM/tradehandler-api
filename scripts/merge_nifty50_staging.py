#!/usr/bin/env python3
"""Merge FnO bars from staging DuckDB into main nifty50.duckdb."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.nifty_duckdb_store import NiftyDuckDBStore, STAGING_DB_PATH


def main() -> None:
    staging = STAGING_DB_PATH
    if not staging.exists():
        print(f"No staging file at {staging}")
        return
    main_store = NiftyDuckDBStore()
    merged = main_store.merge_fno_from(staging)
    print(f"Merged {merged} new/updated FnO bars into {main_store.db_path}")
    staging.unlink()
    print(f"Removed staging file {staging}")
    print(main_store.fno_summary())


if __name__ == "__main__":
    main()
