"""Export Dhan backtest JSON results to trade CSV."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


CSV_COLUMNS = [
    "segment",
    "session_date",
    "symbol",
    "kind",
    "strike",
    "strike_source",
    "entry_datetime_ist",
    "exit_datetime_ist",
    "hold_minutes",
    "entry_premium",
    "sl_premium",
    "target_premium",
    "exit_premium",
    "pnl_inr",
    "r_multiple",
    "exit_reason",
    "num_lots",
    "quantity",
    "risk_at_sl_inr",
    "capital_before",
    "capital_after",
    "index_open",
]


def _hold_minutes(entry_ist: str, exit_ist: str) -> int:
    try:
        eh, em = map(int, entry_ist.split()[1].split(":")[:2])
        xh, xm = map(int, exit_ist.split()[1].split(":")[:2])
        return max(0, (xh * 60 + xm) - (eh * 60 + em))
    except (IndexError, ValueError):
        return 0


def trades_from_report(segment: str, report: Dict[str, Any], *, lot_size: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in report.get("trades") or []:
        entry_ist = str(t.get("entry_datetime_ist") or "")
        exit_ist = str(t.get("exit_datetime_ist") or "")
        lots = int(t.get("num_lots") or 0)
        rows.append(
            {
                "segment": segment.upper(),
                "session_date": t.get("expiry_date") or t.get("session_date"),
                "symbol": t.get("symbol") or "",
                "kind": t.get("kind") or t.get("direction"),
                "strike": t.get("strike"),
                "strike_source": t.get("strike_source") or "",
                "entry_datetime_ist": entry_ist,
                "exit_datetime_ist": exit_ist,
                "hold_minutes": _hold_minutes(entry_ist, exit_ist),
                "entry_premium": t.get("entry"),
                "sl_premium": t.get("sl"),
                "target_premium": t.get("target"),
                "exit_premium": t.get("exit"),
                "pnl_inr": t.get("pnl_inr"),
                "r_multiple": t.get("r_multiple"),
                "exit_reason": t.get("exit_reason"),
                "num_lots": lots,
                "quantity": lots * lot_size if lots else "",
                "risk_at_sl_inr": t.get("risk_at_sl_inr"),
                "capital_before": t.get("capital_before"),
                "capital_after": t.get("capital_after"),
                "index_open": t.get("index_open"),
            }
        )
    return rows


def write_trades_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def export_backtest_result_csv(
    result: Dict[str, Any],
    path: Path,
    *,
    segment: str,
    lot_size: int,
    mode: str = "5m",
) -> int:
    report = (result.get("reports") or {}).get(mode) or {}
    rows = trades_from_report(segment, report, lot_size=lot_size)
    write_trades_csv(path, rows)
    return len(rows)
