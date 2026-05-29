"""
Green Bar Sentinel — user's OI-buildup strike selection for Nifty50.

Kite option chain shows green bars under OI change. This module:
1. Snapshots opening OI at session start (baseline).
2. Ranks strikes by intraday OI buildup (proxy for green underline length).
3. Picks the **2nd-ranked** anchor strike on the active side (CE/PE).
4. Confirms reversal at that anchor before entry.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from utils.logger import log_info, log_warning

IST = ZoneInfo("Asia/Kolkata")
BASELINE_PATH = Path("data/nifty_oi_baseline.json")

STRATEGY_ID = "green_bar_sentinel_2nd_oi"
STRATEGY_NAME = "Green Bar Sentinel — 2nd OI Anchor Reversal"
STRATEGY_DESC = (
    "After open, rank strikes by OI buildup (Kite green-bar proxy). "
    "Monitor the 2nd-highest OI-anchor strike, wait for reversal confirmation, then enter."
)


def _today_key() -> str:
    return datetime.now(IST).date().isoformat()


def _load_baseline() -> Dict[str, Any]:
    if not BASELINE_PATH.exists():
        return {}
    try:
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_baseline(data: Dict[str, Any]) -> None:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _quote_rows(kite, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [f"NFO:{r['tradingsymbol']}" for r in rows if r.get("tradingsymbol")]
    if not keys:
        return {}
    try:
        return kite.quote(keys) or {}
    except Exception as exc:
        log_warning(f"[OI Sentinel] quote failed: {exc}")
        return {}


def enrich_chain_with_oi_buildup(
    kite,
    spot: float,
    universe: List[Dict[str, Any]],
    *,
    strike_window: int = 500,
) -> Dict[str, Any]:
    """
    Extend chain payload with OI buildup ranks and 2nd-anchor strikes.
    """
    if spot <= 0 or not universe:
        return {}

    atm = int(round(spot / 50) * 50)
    expiries = sorted({u["expiry"] for u in universe if u.get("expiry")})
    if not expiries:
        return {}
    expiry = expiries[0]

    rows = [
        u
        for u in universe
        if u.get("expiry") == expiry
        and abs(int(u.get("strike") or 0) - atm) <= strike_window
    ]
    quotes = _quote_rows(kite, rows)

    baseline_doc = _load_baseline()
    baseline_day = baseline_doc.get("date")
    baseline_map: Dict[str, float] = baseline_doc.get("oi_by_symbol") or {}

    # First snapshot of the session becomes baseline (post 9:15).
    now = datetime.now(IST)
    session_minutes = now.hour * 60 + now.minute
    need_baseline = baseline_day != _today_key() and session_minutes >= 9 * 60 + 15

    current_oi: Dict[str, float] = {}
    by_strike: Dict[int, Dict[str, Any]] = {}

    for r in rows:
        sym = r.get("tradingsymbol") or ""
        key = f"NFO:{sym}"
        qd = quotes.get(key, {}) or {}
        oi = float(qd.get("oi") or 0)
        ltp = float(qd.get("last_price") or 0)
        current_oi[sym] = oi
        strike = int(r.get("strike") or 0)
        kind = (r.get("instrument_type") or "").upper()
        if strike not in by_strike:
            by_strike[strike] = {"strike": strike, "ce_oi": 0.0, "pe_oi": 0.0, "ce_ltp": 0.0, "pe_ltp": 0.0}
        if kind == "CE":
            by_strike[strike]["ce_oi"] = oi
            by_strike[strike]["ce_ltp"] = ltp
            by_strike[strike]["ce_symbol"] = sym
        elif kind == "PE":
            by_strike[strike]["pe_oi"] = oi
            by_strike[strike]["pe_ltp"] = ltp
            by_strike[strike]["pe_symbol"] = sym

    if need_baseline and current_oi:
        _save_baseline({"date": _today_key(), "oi_by_symbol": current_oi, "captured_at": now.isoformat()})
        baseline_map = dict(current_oi)
        log_info(f"[OI Sentinel] Opening OI baseline saved ({len(current_oi)} symbols)")

    ranked_ce: List[Dict[str, Any]] = []
    ranked_pe: List[Dict[str, Any]] = []

    for strike, row in sorted(by_strike.items()):
        ce_sym = row.get("ce_symbol") or ""
        pe_sym = row.get("pe_symbol") or ""
        ce_base = float(baseline_map.get(ce_sym, 0) or 0)
        pe_base = float(baseline_map.get(pe_sym, 0) or 0)
        ce_chg = max(0.0, float(row.get("ce_oi") or 0) - ce_base)
        pe_chg = max(0.0, float(row.get("pe_oi") or 0) - pe_base)
        ce_pct = (ce_chg / ce_base * 100.0) if ce_base > 0 else (100.0 if ce_chg > 0 else 0.0)
        pe_pct = (pe_chg / pe_base * 100.0) if pe_base > 0 else (100.0 if pe_chg > 0 else 0.0)
        if ce_chg > 0 or float(row.get("ce_oi") or 0) > 0:
            ranked_ce.append(
                {
                    "strike": strike,
                    "oi": float(row.get("ce_oi") or 0),
                    "oi_change": ce_chg,
                    "oi_change_pct": round(ce_pct, 1),
                    "ltp": float(row.get("ce_ltp") or 0),
                    "symbol": ce_sym,
                }
            )
        if pe_chg > 0 or float(row.get("pe_oi") or 0) > 0:
            ranked_pe.append(
                {
                    "strike": strike,
                    "oi": float(row.get("pe_oi") or 0),
                    "oi_change": pe_chg,
                    "oi_change_pct": round(pe_pct, 1),
                    "ltp": float(row.get("pe_ltp") or 0),
                    "symbol": pe_sym,
                }
            )

    ranked_ce.sort(key=lambda x: (x["oi_change"], x["oi"]), reverse=True)
    ranked_pe.sort(key=lambda x: (x["oi_change"], x["oi"]), reverse=True)

    def _pick_second(ranked: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if len(ranked) >= 2:
            return ranked[1]
        return ranked[0] if ranked else None

    second_ce = _pick_second(ranked_ce)
    second_pe = _pick_second(ranked_pe)

    # Active side: stronger 2nd-anchor OI buildup; tie-break with spot bias.
    ce_score = float(second_ce.get("oi_change") or 0) if second_ce else 0.0
    pe_score = float(second_pe.get("oi_change") or 0) if second_pe else 0.0
    if ce_score >= pe_score:
        active_kind = "CE"
        active_anchor = second_ce
    else:
        active_kind = "PE"
        active_anchor = second_pe

    return {
        "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
        "atm": atm,
        "ranked_ce_oi": ranked_ce[:8],
        "ranked_pe_oi": ranked_pe[:8],
        "second_ce_anchor": second_ce,
        "second_pe_anchor": second_pe,
        "active_kind": active_kind,
        "active_anchor": active_anchor,
        "baseline_ready": baseline_day == _today_key() or need_baseline,
    }


def pick_sentinel_anchor(chain_oi: Dict[str, Any], direction_pref: str = "AUTO") -> Tuple[str, Optional[int], Dict[str, Any]]:
    """Return (kind, anchor_strike, meta)."""
    pref = (direction_pref or "AUTO").upper()
    if pref in ("CE", "PE"):
        anchor = chain_oi.get("second_ce_anchor") if pref == "CE" else chain_oi.get("second_pe_anchor")
        if anchor:
            return pref, int(anchor["strike"]), anchor
    active = chain_oi.get("active_anchor") or {}
    kind = chain_oi.get("active_kind") or "CE"
    if active.get("strike"):
        return kind, int(active["strike"]), active
    return kind, None, {}


def reversal_confirmed_at_anchor(
    *,
    spot: float,
    anchor_strike: int,
    kind: str,
    intra: Dict[str, Any],
    last_5m_close: Optional[float] = None,
) -> Tuple[bool, int, str]:
    """
    Reversal confirmation at the 2nd OI anchor:
    - CE: spot tested near/below anchor, then reclaimed above (V-reversal).
    - PE: spot tested near/above anchor, then rejected below (inverted-V).
    """
    if anchor_strike <= 0 or spot <= 0:
        return False, 35, "Anchor strike unavailable"

    kind = (kind or "CE").upper()
    buf = 25.0
    day_low = float(intra.get("day_low") or spot)
    day_high = float(intra.get("day_high") or spot)
    l5 = float(last_5m_close) if last_5m_close is not None else spot
    score = 40

    if kind == "CE":
        tested = day_low <= anchor_strike + buf
        reclaimed = spot >= anchor_strike - 10 and l5 >= anchor_strike - 15
        if tested and reclaimed:
            score += 35
            return True, min(95, score), f"CE reversal: tested ≤{anchor_strike + buf:.0f}, reclaimed above anchor"
        if tested:
            score += 15
            return False, score, f"CE anchor {anchor_strike}: tested — wait for reclaim above anchor"
        return False, score, f"CE anchor {anchor_strike}: waiting for dip toward anchor"

    tested = day_high >= anchor_strike - buf
    rejected = spot <= anchor_strike + 10 and l5 <= anchor_strike + 15
    if tested and rejected:
        score += 35
        return True, min(95, score), f"PE reversal: tested ≥{anchor_strike - buf:.0f}, rejected below anchor"
    if tested:
        score += 15
        return False, score, f"PE anchor {anchor_strike}: tested — wait for rejection below anchor"
    return False, score, f"PE anchor {anchor_strike}: waiting for rally toward anchor"
