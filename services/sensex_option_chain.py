"""Sensex BFO chain builder for executor and backtests."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from services.push.option_contract_resolver import OptionContract
from services.sensex_constants import sensex_premium_band_scan_points


def _expiry_key(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(str(value).strip()[:10])
        except ValueError:
            return None
    return None


def _same_expiry(left: Any, right: Any) -> bool:
    a, b = _expiry_key(left), _expiry_key(right)
    return a is not None and a == b


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
        exp = _expiry_key(row.get("expiry"))
        if exp is not None and exp >= today:
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
    max_dist = sensex_premium_band_scan_points()
    for row in rows:
        if not _same_expiry(row.get("expiry"), target_expiry):
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
        and _same_expiry(r.get("expiry"), target_expiry)
    ]
    if not same_kind:
        return None
    nearest = min(same_kind, key=lambda s: abs(s - int(strike)))
    if abs(nearest - int(strike)) > max_dist:
        return None
    return resolve_sensex_contract(
        kite, strike=nearest, kind=kind, expiry=target_expiry, universe=rows
    )


def resolve_sensex_contract_by_symbol(
    kite,
    tradingsymbol: str,
    *,
    universe: Optional[List[Dict[str, Any]]] = None,
) -> Optional[OptionContract]:
    """Resolve BFO contract from tradingsymbol (preferred when strategy already picked strike)."""
    sym = (tradingsymbol or "").strip().upper()
    if not sym:
        return None
    rows = universe if universe is not None else build_sensex_options_universe(kite)
    for row in rows:
        if str(row.get("tradingsymbol") or "").upper() != sym:
            continue
        exp = _expiry_key(row.get("expiry"))
        kind = (row.get("instrument_type") or "").upper()
        tok = int(row.get("instrument_token") or 0)
        strike = int(row.get("strike") or 0)
        if not exp or kind not in ("CE", "PE") or tok <= 0 or strike <= 0:
            continue
        return OptionContract(
            tradingsymbol=sym,
            exchange="BFO",
            instrument_token=tok,
            strike=strike,
            expiry=exp,
            instrument_type=kind,
        )
    return None


def sensex_index_token(kite) -> int:
    try:
        for inst in kite.instruments("BSE"):
            if inst.get("tradingsymbol") == "SENSEX" and inst.get("instrument_type") == "INDEX":
                return int(inst["instrument_token"])
    except Exception:
        pass
    return 265
