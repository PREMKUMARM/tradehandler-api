"""Sensex BFO chain builder for executor and backtests."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from services.push.option_contract_resolver import OptionContract


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


def _nearest_expiry(universe: List[Dict[str, Any]]) -> Optional[date]:
    today = date.today()
    expiries: List[date] = []
    for row in universe:
        exp = row.get("expiry")
        if isinstance(exp, datetime):
            exp = exp.date()
        elif isinstance(exp, str):
            try:
                exp = date.fromisoformat(exp[:10])
            except ValueError:
                continue
        if isinstance(exp, date) and exp >= today:
            expiries.append(exp)
    if not expiries:
        return None
    return min(expiries)


def resolve_sensex_contract(
    kite,
    *,
    strike: int,
    kind: str,
    expiry: Optional[date] = None,
    universe: Optional[List[Dict[str, Any]]] = None,
) -> Optional[OptionContract]:
    """Resolve nearest weekly SENSEX BFO option at strike/kind."""
    kind = (kind or "").upper()
    if kind not in ("CE", "PE") or strike <= 0:
        return None
    rows = universe if universe is not None else build_sensex_options_universe(kite)
    if not rows:
        return None
    target_expiry = expiry or _nearest_expiry(rows)
    if target_expiry is None:
        return None
    for row in rows:
        exp = row.get("expiry")
        if isinstance(exp, datetime):
            exp = exp.date()
        elif isinstance(exp, str):
            try:
                exp = date.fromisoformat(str(exp)[:10])
            except ValueError:
                continue
        if exp != target_expiry:
            continue
        if int(row.get("strike") or 0) != int(strike):
            continue
        if (row.get("instrument_type") or "").upper() != kind:
            continue
        tok = int(row.get("instrument_token") or 0)
        sym = row.get("tradingsymbol") or ""
        if not sym or tok <= 0:
            continue
        return OptionContract(
            tradingsymbol=sym,
            exchange="BFO",
            instrument_token=tok,
            strike=int(strike),
            expiry=target_expiry,
            instrument_type=kind,
        )
    same_kind = [
        int(r.get("strike") or 0)
        for r in rows
        if (r.get("instrument_type") or "").upper() == kind
        and (
            (isinstance(r.get("expiry"), date) and r.get("expiry") == target_expiry)
            or str(r.get("expiry"))[:10] == target_expiry.isoformat()
        )
    ]
    if not same_kind:
        return None
    nearest = min(same_kind, key=lambda s: abs(s - int(strike)))
    return resolve_sensex_contract(
        kite, strike=nearest, kind=kind, expiry=target_expiry, universe=rows
    )


def sensex_index_token(kite) -> int:
    try:
        for inst in kite.instruments("BSE"):
            if inst.get("tradingsymbol") == "SENSEX" and inst.get("instrument_type") == "INDEX":
                return int(inst["instrument_token"])
    except Exception:
        pass
    return 265
