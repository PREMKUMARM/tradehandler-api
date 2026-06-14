"""Sensex BFO chain builder for executor and backtests."""
from typing import Any, Dict, List


def build_sensex_options_universe(kite) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        for inst in kite.instruments("BFO"):
            if inst.get("name") != "SENSEX":
                continue
            if inst.get("instrument_type") not in ("CE", "PE"):
                continue
            out.append(
                {
                    "strike": inst.get("strike"),
                    "instrument_type": inst.get("instrument_type"),
                    "expiry": inst.get("expiry"),
                    "tradingsymbol": inst.get("tradingsymbol"),
                    "instrument_token": inst.get("instrument_token"),
                }
            )
    except Exception:
        pass
    return out


def sensex_index_token(kite) -> int:
    try:
        for inst in kite.instruments("BSE"):
            if inst.get("tradingsymbol") == "SENSEX" and inst.get("instrument_type") == "INDEX":
                return int(inst["instrument_token"])
    except Exception:
        pass
    return 265
