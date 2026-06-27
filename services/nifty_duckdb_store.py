"""DuckDB storage for Nifty 50 index OHLC bars."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import duckdb

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "market" / "nifty50.duckdb"
STAGING_DB_PATH = DEFAULT_DB_PATH.with_suffix(".staging.duckdb")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS nifty50_index_bars (
    ts BIGINT NOT NULL,
    session_date DATE NOT NULL,
    interval_min INTEGER NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    source VARCHAR DEFAULT 'dhan',
    fetched_at TIMESTAMP,
    PRIMARY KEY (ts, interval_min)
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_nifty50_session_date
ON nifty50_index_bars(session_date, interval_min);
"""

CREATE_FNO_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS nifty50_fno_bars (
    ts BIGINT NOT NULL,
    session_date DATE NOT NULL,
    kind VARCHAR NOT NULL,
    strike_offset VARCHAR NOT NULL,
    expiry_code INTEGER NOT NULL,
    expiry_flag VARCHAR NOT NULL,
    interval_min INTEGER NOT NULL,
    strike DOUBLE,
    spot DOUBLE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    oi DOUBLE,
    volume BIGINT,
    source VARCHAR DEFAULT 'dhan',
    fetched_at TIMESTAMP,
    PRIMARY KEY (ts, kind, strike_offset, expiry_code, interval_min)
);
"""

CREATE_FNO_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_nifty50_fno_session
ON nifty50_fno_bars(session_date, kind, strike_offset, interval_min);
"""

NIFTY_STRIKE_OFFSETS: Tuple[str, ...] = (
    "ATM",
    *(f"ATM+{i}" for i in range(1, 11)),
    *(f"ATM-{i}" for i in range(1, 11)),
)


def parse_intraday_response(
    resp: Dict[str, Any],
    *,
    interval_min: int = 5,
    source: str = "dhan",
    fetched_at: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Flatten Dhan /charts/intraday array response into row dicts."""
    timestamps = resp.get("timestamp") or []
    if not timestamps:
        return []

    opens = resp.get("open") or []
    highs = resp.get("high") or []
    lows = resp.get("low") or []
    closes = resp.get("close") or []
    volumes = resp.get("volume") or [0] * len(timestamps)
    fetched = fetched_at or datetime.now(IST)

    rows: List[Dict[str, Any]] = []
    for idx, raw_ts in enumerate(timestamps):
        ts = int(raw_ts)
        session_day = datetime.fromtimestamp(ts, tz=IST).date()
        rows.append(
            {
                "ts": ts,
                "session_date": session_day,
                "interval_min": int(interval_min),
                "open": float(opens[idx]),
                "high": float(highs[idx]),
                "low": float(lows[idx]),
                "close": float(closes[idx]),
                "volume": int(volumes[idx] if idx < len(volumes) else 0),
                "source": source,
                "fetched_at": fetched,
            }
        )
    return rows


def parse_rolling_option_leg(
    leg: Dict[str, Any],
    *,
    kind: str,
    strike_offset: str,
    expiry_code: int,
    expiry_flag: str,
    interval_min: int = 5,
    source: str = "dhan",
    fetched_at: Optional[datetime] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """Flatten Dhan rollingoption CE/PE leg arrays into row dicts."""
    timestamps = leg.get("timestamp") or []
    if not timestamps:
        return []

    opens = leg.get("open") or []
    highs = leg.get("high") or []
    lows = leg.get("low") or []
    closes = leg.get("close") or []
    ois = leg.get("oi") or [0] * len(timestamps)
    spots = leg.get("spot") or [0] * len(timestamps)
    strikes = leg.get("strike") or [0] * len(timestamps)
    volumes = leg.get("volume") or [0] * len(timestamps)
    fetched = fetched_at or datetime.now(IST)
    kind_key = kind.upper()

    rows: List[Dict[str, Any]] = []
    for idx, raw_ts in enumerate(timestamps):
        ts = int(raw_ts)
        session_day = datetime.fromtimestamp(ts, tz=IST).date()
        if start_date and session_day < start_date:
            continue
        if end_date and session_day > end_date:
            continue
        rows.append(
            {
                "ts": ts,
                "session_date": session_day,
                "kind": kind_key,
                "strike_offset": strike_offset,
                "expiry_code": int(expiry_code),
                "expiry_flag": expiry_flag,
                "interval_min": int(interval_min),
                "strike": float(strikes[idx] if idx < len(strikes) else 0),
                "spot": float(spots[idx] if idx < len(spots) else 0),
                "open": float(opens[idx]),
                "high": float(highs[idx]),
                "low": float(lows[idx]),
                "close": float(closes[idx]),
                "oi": float(ois[idx] if idx < len(ois) else 0),
                "volume": int(volumes[idx] if idx < len(volumes) else 0),
                "source": source,
                "fetched_at": fetched,
            }
        )
    return rows


class NiftyDuckDBStore:
    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        try:
            return duckdb.connect(str(self.db_path), read_only=read_only)
        except duckdb.IOException as exc:
            if "Conflicting lock" in str(exc):
                raise RuntimeError(
                    f"Cannot open {self.db_path} — close the DuckDB extension in Cursor "
                    f"(disconnect database / reload window), then retry."
                ) from exc
            raise

    @classmethod
    def connect_path(cls, db_path: Path | str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(db_path), read_only=read_only)

    def merge_fno_from(self, staging_path: Path | str) -> int:
        """Merge FnO bars from a staging DuckDB file into this store."""
        staging = Path(staging_path)
        if not staging.exists():
            return 0
        conn = self.connect()
        try:
            self.ensure_schema(conn)
            conn.execute(f"ATTACH '{staging}' AS staging (READ_ONLY)")
            before = conn.execute("SELECT COUNT(*) FROM nifty50_fno_bars").fetchone()[0]
            conn.execute(
                """
                INSERT INTO nifty50_fno_bars
                SELECT * FROM staging.nifty50_fno_bars
                ON CONFLICT (ts, kind, strike_offset, expiry_code, interval_min) DO UPDATE SET
                    session_date = excluded.session_date,
                    expiry_flag = excluded.expiry_flag,
                    strike = excluded.strike,
                    spot = excluded.spot,
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    oi = excluded.oi,
                    volume = excluded.volume,
                    source = excluded.source,
                    fetched_at = excluded.fetched_at
                """
            )
            after = conn.execute("SELECT COUNT(*) FROM nifty50_fno_bars").fetchone()[0]
            conn.execute("DETACH staging")
            return int(after - before)
        finally:
            conn.close()

    def ensure_schema(self, conn: Optional[duckdb.DuckDBPyConnection] = None) -> None:
        own = conn is None
        conn = conn or self.connect()
        try:
            conn.execute(CREATE_TABLE_SQL)
            conn.execute(CREATE_INDEX_SQL)
            conn.execute(CREATE_FNO_TABLE_SQL)
            conn.execute(CREATE_FNO_INDEX_SQL)
        finally:
            if own:
                conn.close()

    def upsert_bars(
        self,
        rows: List[Dict[str, Any]],
        conn: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> int:
        if not rows:
            return 0

        own = conn is None
        conn = conn or self.connect()
        try:
            self.ensure_schema(conn)
            conn.executemany(
                """
                INSERT INTO nifty50_index_bars AS t (
                    ts, session_date, interval_min, open, high, low, close, volume, source, fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ts, interval_min) DO UPDATE SET
                    session_date = excluded.session_date,
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    source = excluded.source,
                    fetched_at = excluded.fetched_at
                """,
                [
                    (
                        row["ts"],
                        row["session_date"],
                        row["interval_min"],
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"],
                        row["source"],
                        row["fetched_at"],
                    )
                    for row in rows
                ],
            )
            return len(rows)
        finally:
            if own:
                conn.close()

    def summary(self, conn: Optional[duckdb.DuckDBPyConnection] = None) -> Dict[str, Any]:
        own = conn is None
        conn = conn or self.connect()
        try:
            self.ensure_schema(conn)
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS bar_count,
                    COUNT(DISTINCT session_date) AS trading_days,
                    MIN(session_date) AS first_date,
                    MAX(session_date) AS last_date,
                    MIN(interval_min) AS interval_min
                FROM nifty50_index_bars
                """
            ).fetchone()
            if not row or row[0] == 0:
                return {
                    "bar_count": 0,
                    "trading_days": 0,
                    "first_date": None,
                    "last_date": None,
                    "interval_min": None,
                    "db_path": str(self.db_path),
                }
            return {
                "bar_count": int(row[0]),
                "trading_days": int(row[1]),
                "first_date": str(row[2]),
                "last_date": str(row[3]),
                "interval_min": int(row[4]),
                "db_path": str(self.db_path),
            }
        finally:
            if own:
                conn.close()

    def has_session(self, session_date: str, interval_min: int = 5) -> bool:
        conn = self.connect()
        try:
            self.ensure_schema(conn)
            row = conn.execute(
                """
                SELECT 1 FROM nifty50_index_bars
                WHERE session_date = ? AND interval_min = ?
                LIMIT 1
                """,
                [session_date, interval_min],
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def load_bars(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval_min: int = 5,
    ) -> List[Dict[str, Any]]:
        conn = self.connect()
        try:
            self.ensure_schema(conn)
            clauses = ["interval_min = ?"]
            params: List[Any] = [interval_min]
            if start_date:
                clauses.append("session_date >= ?")
                params.append(start_date)
            if end_date:
                clauses.append("session_date <= ?")
                params.append(end_date)
            where = " AND ".join(clauses)
            cur = conn.execute(
                f"""
                SELECT ts, session_date, open, high, low, close, volume
                FROM nifty50_index_bars
                WHERE {where}
                ORDER BY ts
                """,
                params,
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        finally:
            conn.close()

    def upsert_fno_bars(
        self,
        rows: List[Dict[str, Any]],
        conn: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> int:
        if not rows:
            return 0

        own = conn is None
        conn = conn or self.connect()
        try:
            self.ensure_schema(conn)
            conn.executemany(
                """
                INSERT INTO nifty50_fno_bars (
                    ts, session_date, kind, strike_offset, expiry_code, expiry_flag,
                    interval_min, strike, spot, open, high, low, close, oi, volume,
                    source, fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ts, kind, strike_offset, expiry_code, interval_min) DO UPDATE SET
                    session_date = excluded.session_date,
                    expiry_flag = excluded.expiry_flag,
                    strike = excluded.strike,
                    spot = excluded.spot,
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    oi = excluded.oi,
                    volume = excluded.volume,
                    source = excluded.source,
                    fetched_at = excluded.fetched_at
                """,
                [
                    (
                        row["ts"],
                        row["session_date"],
                        row["kind"],
                        row["strike_offset"],
                        row["expiry_code"],
                        row["expiry_flag"],
                        row["interval_min"],
                        row["strike"],
                        row["spot"],
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["oi"],
                        row["volume"],
                        row["source"],
                        row["fetched_at"],
                    )
                    for row in rows
                ],
            )
            return len(rows)
        finally:
            if own:
                conn.close()

    def fno_summary(self, conn: Optional[duckdb.DuckDBPyConnection] = None) -> Dict[str, Any]:
        own = conn is None
        conn = conn or self.connect()
        try:
            self.ensure_schema(conn)
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS bar_count,
                    COUNT(DISTINCT session_date) AS trading_days,
                    COUNT(DISTINCT strike_offset) AS offsets,
                    MIN(session_date) AS first_date,
                    MAX(session_date) AS last_date,
                    MIN(interval_min) AS interval_min
                FROM nifty50_fno_bars
                """
            ).fetchone()
            if not row or row[0] == 0:
                return {
                    "bar_count": 0,
                    "trading_days": 0,
                    "offsets": 0,
                    "first_date": None,
                    "last_date": None,
                    "interval_min": None,
                    "db_path": str(self.db_path),
                }
            return {
                "bar_count": int(row[0]),
                "trading_days": int(row[1]),
                "offsets": int(row[2]),
                "first_date": str(row[3]),
                "last_date": str(row[4]),
                "interval_min": int(row[5]),
                "db_path": str(self.db_path),
            }
        finally:
            if own:
                conn.close()

    def has_fno_range(
        self,
        start_date: str,
        end_date: str,
        *,
        interval_min: int = 5,
        expiry_code: int = 1,
    ) -> bool:
        conn = self.connect()
        try:
            self.ensure_schema(conn)
            row = conn.execute(
                """
                SELECT COUNT(DISTINCT session_date) AS days
                FROM nifty50_fno_bars
                WHERE session_date BETWEEN ? AND ?
                  AND interval_min = ?
                  AND expiry_code = ?
                  AND strike_offset = 'ATM'
                  AND kind = 'CE'
                """,
                [start_date, end_date, interval_min, expiry_code],
            ).fetchone()
            if not row or row[0] == 0:
                return False
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
            expected = sum(
                1
                for i in range((end - start).days + 1)
                if (start + timedelta(days=i)).weekday() < 5
            )
            return int(row[0]) >= expected
        finally:
            conn.close()


def iter_weekdays(start: date, end: date) -> List[date]:
    out: List[date] = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def iter_intraday_chunks(
    start: date,
    end: date,
    *,
    max_days: int = 90,
) -> List[Tuple[str, str]]:
    """Yield [from_date, to_date) windows for Dhan intraday (max 90 days per call)."""
    chunks: List[Tuple[str, str]] = []
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=max_days), end)
        chunks.append((cur.isoformat(), chunk_end.isoformat()))
        cur = chunk_end
    return chunks
