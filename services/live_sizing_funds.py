"""
Virtual fund amounts for position sizing in live mode (qty from risk % / entry / SL).

Does not replace Kite margin checks for order placement — only sizes quantity.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from services.paper_trading import normalize_segment
from utils.logger import log_info, log_warning

LIVE_SIZING_FUNDS_PATH = Path(
    os.getenv("LIVE_SIZING_FUNDS_FILE", "data/live_sizing_funds.json")
)

DEFAULT_LIVE_SIZING: Dict[str, Dict[str, Any]] = {
    "nifty50": {"allocated": 500_000.0, "currency": "INR"},
    "commodity": {"allocated": 200_000.0, "currency": "INR"},
    "crypto": {"allocated": 10_000.0, "currency": "USDT"},
    "sensex": {"allocated": 500_000.0, "currency": "INR"},
}


def _read_file() -> Dict[str, Dict[str, Any]]:
    out = {k: dict(v) for k, v in DEFAULT_LIVE_SIZING.items()}
    try:
        if LIVE_SIZING_FUNDS_PATH.exists():
            raw = json.loads(LIVE_SIZING_FUNDS_PATH.read_text(encoding="utf-8"))
            for seg in ("nifty50", "commodity", "crypto", "sensex"):
                if seg in raw and isinstance(raw[seg], dict) and "allocated" in raw[seg]:
                    out[seg]["allocated"] = float(raw[seg]["allocated"])
                    if raw[seg].get("currency"):
                        out[seg]["currency"] = str(raw[seg]["currency"]).upper()
    except Exception as e:
        log_warning(f"[LiveSizingFunds] read failed: {e}")
    return out


def _write_file(data: Dict[str, Dict[str, Any]]) -> None:
    LIVE_SIZING_FUNDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_SIZING_FUNDS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_live_sizing_amount(segment: str) -> float:
    """Virtual capital for qty sizing in live mode; 0 = fall back to Kite margin."""
    seg = normalize_segment(segment)
    return float(_read_file().get(seg, DEFAULT_LIVE_SIZING["nifty50"])["allocated"])


def get_live_sizing_snapshot(segment: str) -> Dict[str, Any]:
    seg = normalize_segment(segment)
    cfg = _read_file().get(seg, DEFAULT_LIVE_SIZING["nifty50"])
    return {
        "segment": seg,
        "allocated": round(float(cfg["allocated"]), 2),
        "currency": str(cfg.get("currency") or "INR"),
        "used_for": "live_qty_sizing_only",
    }


def get_all_live_sizing_snapshots() -> Dict[str, Dict[str, Any]]:
    return {seg: get_live_sizing_snapshot(seg) for seg in ("nifty50", "commodity", "crypto", "sensex")}


def set_live_sizing_allocated(segment: str, allocated: float) -> Dict[str, Any]:
    seg = normalize_segment(segment)
    amt = float(allocated)
    if amt < 1000:
        raise ValueError("Live sizing fund must be at least 1000")
    data = _read_file()
    cur = data.get(seg, dict(DEFAULT_LIVE_SIZING.get(seg, DEFAULT_LIVE_SIZING["nifty50"])))
    cur["allocated"] = amt
    data[seg] = cur
    _write_file(data)
    log_info(f"[LiveSizingFunds] {seg} live sizing capital={amt:,.2f}")
    return get_live_sizing_snapshot(seg)
