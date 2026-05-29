"""
Paper trade row detail — live segment indicators + entry/SL/TP reasoning.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from services.paper_trading import infer_segment_from_order, normalize_segment

IST = ZoneInfo("Asia/Kolkata")


def _parse_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def slim_trade_plan_for_paper(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Compact snapshot stored on paper order payload for journal expand view."""
    ind = dict(plan.get("indicators") or {})
    keys = (
        "nifty_spot",
        "underlying_spot",
        "prev_close",
        "vix",
        "pdh",
        "pdl",
        "or_high",
        "or_low",
        "ema9",
        "bb_lower",
        "bb_middle",
        "bb_upper",
        "bb_zone",
        "bb_on_contract",
        "spot_entry",
        "spot_stop_loss",
        "spot_target",
        "option_bid",
        "option_ask",
        "option_ltp",
        "level_note",
        "indicator_window",
        "strategy_id",
        "pattern_tag",
        "margin",
        "risk_pct",
    )
    slim_ind = {k: ind[k] for k in keys if ind.get(k) is not None}
    return {
        "strategy_id": plan.get("strategy_id"),
        "strategy_name": plan.get("strategy_name"),
        "option_type": plan.get("option_type"),
        "strike": plan.get("strike"),
        "expiry": plan.get("expiry"),
        "entry_limit_price": plan.get("entry_limit_price"),
        "entry_premium": plan.get("entry_premium"),
        "entry_fair_premium": plan.get("entry_fair_premium"),
        "entry_style": plan.get("entry_style"),
        "entry_ready": plan.get("entry_ready"),
        "entry_confirmation_score": plan.get("entry_confirmation_score"),
        "entry_spot_trigger": plan.get("entry_spot_trigger"),
        "entry_block_reason": plan.get("entry_block_reason"),
        "stop_loss_premium": plan.get("stop_loss_premium"),
        "target_premium": plan.get("target_premium"),
        "spot_stop_loss": plan.get("spot_stop_loss"),
        "spot_target": plan.get("spot_target"),
        "delta_used": plan.get("delta_used"),
        "pattern_tag": plan.get("pattern_tag"),
        "strike_moneyness": plan.get("strike_moneyness"),
        "note": plan.get("note"),
        "level_note": ind.get("level_note") or plan.get("level_note"),
        "bb_on_contract": plan.get("bb_on_contract") or ind.get("bb_on_contract"),
        "indicators": slim_ind,
        "captured_at": datetime.now(IST).isoformat(),
    }


def _fmt_num(v: Any, digits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
        if abs(f) >= 1000:
            return f"{f:,.{digits}f}"
        return f"{f:.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _merge_option_bb(live: Dict[str, Any], opt_bb: Dict[str, Any]) -> None:
    """Overlay contract BB + LTP onto live indicator bundle."""
    if opt_bb.get("bb_middle") is None:
        return
    live["bb_lower"] = opt_bb.get("bb_lower")
    live["bb_middle"] = opt_bb.get("bb_middle")
    live["bb_upper"] = opt_bb.get("bb_upper")
    live["last_5m_close"] = opt_bb.get("last_5m_close")
    live["option_ltp"] = opt_bb.get("option_ltp") or live.get("option_ltp")
    live["indicator_window"] = opt_bb.get("indicator_window") or live.get("indicator_window")
    live["indicator_sources"] = {
        **(live.get("indicator_sources") or {}),
        **(opt_bb.get("indicator_sources") or {}),
    }
    live["bb_bar_count"] = opt_bb.get("bar_count")


def _live_indicators_for_segment(
    segment: str,
    *,
    tradingsymbol: Optional[str] = None,
    exchange: Optional[str] = None,
) -> Dict[str, Any]:
    seg = normalize_segment(segment)
    sym = (tradingsymbol or "").strip()
    ex = (exchange or ("MCX" if seg == "commodity" else "NFO")).upper()

    if seg == "nifty50":
        from services.kite_live_indicators import get_nifty_bundle_for_v2, get_option_bollinger_snapshot

        bundle = get_nifty_bundle_for_v2()
        live: Dict[str, Any] = {
            "segment": seg,
            "underlying_label": "Nifty 50 (index)",
            "contract_label": sym or "—",
            "spot": bundle.get("nifty_spot"),
            "spot_source": bundle.get("spot_source"),
            "prev_close": bundle.get("prev_close"),
            "vix": bundle.get("vix"),
            "pdh": bundle.get("pdh"),
            "pdl": bundle.get("pdl"),
            "or_high": bundle.get("or_high"),
            "or_low": bundle.get("or_low"),
            "ema9": bundle.get("ema9"),
            "bb_lower": None,
            "bb_middle": None,
            "bb_upper": None,
            "last_5m_close": None,
            "option_ltp": None,
            "indicator_window": None,
            "indicator_sources": bundle.get("indicator_sources") or {},
            "last_tick_at": bundle.get("last_tick_at"),
            "updated_at": datetime.now(IST).isoformat(),
            "compare_hint": (
                "Bollinger (5m, 20, 2σ) is on the **option contract** chart in Zerodha — "
                "open the same symbol as this trade. Nifty index levels below are context only."
            ),
        }
        if sym:
            opt_bb = get_option_bollinger_snapshot(sym, ex)
            _merge_option_bb(live, opt_bb)
            live["contract_label"] = sym
            if opt_bb.get("error"):
                live["bb_error"] = opt_bb.get("error")
        return live

    if seg == "commodity":
        from services.commodity_live_indicators import recalculate_from_ticker
        from services.kite_live_indicators import get_option_bollinger_snapshot

        bundle = recalculate_from_ticker()
        live = {
            "segment": seg,
            "underlying_label": "MCX Crude (future)",
            "contract_label": sym or "—",
            "spot": bundle.get("underlying_spot") or bundle.get("nifty_spot"),
            "spot_source": bundle.get("spot_source"),
            "prev_close": bundle.get("prev_close"),
            "vix": None,
            "pdh": bundle.get("pdh"),
            "pdl": bundle.get("pdl"),
            "or_high": bundle.get("or_high"),
            "or_low": bundle.get("or_low"),
            "ema9": bundle.get("ema9"),
            "bb_lower": None,
            "bb_middle": None,
            "bb_upper": None,
            "last_5m_close": None,
            "option_ltp": None,
            "indicator_window": None,
            "indicator_sources": bundle.get("indicator_sources") or {},
            "last_tick_at": bundle.get("last_tick_at"),
            "updated_at": datetime.now(IST).isoformat(),
            "compare_hint": (
                "Compare 5m BB on the **same MCX contract** as this trade in Zerodha Charts."
            ),
        }
        if sym:
            _merge_option_bb(live, get_option_bollinger_snapshot(sym, ex))
            live["contract_label"] = sym
        return live

    return {
        "segment": seg,
        "underlying_label": "Crypto",
        "spot": None,
        "compare_hint": "Crypto paper uses Binance marks — no Nifty-style BB on this journal row.",
        "updated_at": datetime.now(IST).isoformat(),
    }


def _indicator_table(live: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    def add(name: str, key: str, *, suffix: str = "") -> None:
        val = live.get(key)
        if val is None:
            return
        rows.append({"name": name, "value": _fmt_num(val) + suffix})

    contract = live.get("contract_label")
    if contract and contract != "—":
        rows.append({"name": "Contract", "value": str(contract)})

    add("Contract LTP", "option_ltp")
    add(live.get("underlying_label") or "Index spot", "spot")
    add("Prev close (index)", "prev_close")
    add("VIX", "vix")
    add("PDH (index)", "pdh")
    add("PDL (index)", "pdl")
    add("OR high (index)", "or_high")
    add("OR low (index)", "or_low")
    add("9 EMA (index 5m)", "ema9")
    add("BB lower (contract 5m)", "bb_lower")
    add("BB middle (contract 5m)", "bb_middle")
    add("BB upper (contract 5m)", "bb_upper")
    add("Last 5m close (contract)", "last_5m_close")
    win = live.get("indicator_window")
    if win:
        if isinstance(win, dict):
            wtxt = (
                f"{win.get('period', 20)} bars · {win.get('timeframe', '5m')} · "
                f"σ={win.get('std_dev', 2)}"
            )
        else:
            wtxt = str(win)
        rows.append({"name": "BB window", "value": wtxt})
    bc = live.get("bb_bar_count")
    if bc:
        rows.append({"name": "BB bars used", "value": str(bc)})
    src = (live.get("indicator_sources") or {}).get("bb_middle")
    if src:
        rows.append({"name": "BB source", "value": str(src)})
    if live.get("bb_error"):
        rows.append({"name": "BB status", "value": str(live["bb_error"])})
    idx_src = live.get("spot_source")
    if idx_src:
        rows.append({"name": "Index spot source", "value": str(idx_src)})
    return rows


def _build_pricing_reason(
    plan: Optional[Dict[str, Any]],
    *,
    entry_price: Optional[float],
    stoploss: Optional[float],
    target: Optional[float],
    symbol: str,
) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = []
    if not plan:
        steps.append(
            {
                "title": "Placement snapshot",
                "detail": (
                    "No trade-plan snapshot was stored when this order was placed. "
                    "Entry/SL/target below are from the paper ledger only. "
                    "New paper entries will include full reasoning."
                ),
            }
        )
        if entry_price is not None:
            steps.append(
                {
                    "title": "Entry (filled)",
                    "detail": f"Paper fill at ₹{_fmt_num(entry_price)} for {symbol}.",
                }
            )
        if stoploss is not None and target is not None:
            steps.append(
                {
                    "title": "Exit levels",
                    "detail": f"Stop-loss ₹{_fmt_num(stoploss)} · Target ₹{_fmt_num(target)}.",
                }
            )
        return steps

    sid = str(plan.get("strategy_id") or "strategy")
    opt = (plan.get("option_type") or "").upper()
    ind = plan.get("indicators") or {}
    contract_sym = plan.get("bb_on_contract") or ind.get("bb_on_contract") or symbol
    prem = ind.get("option_ltp") or plan.get("entry_premium") or entry_price
    spot_sl = plan.get("spot_stop_loss") or ind.get("spot_stop_loss")
    spot_tp = plan.get("spot_target") or ind.get("spot_target")
    delta = plan.get("delta_used")
    style = plan.get("entry_style") or ""
    fair = plan.get("entry_fair_premium")
    limit = plan.get("entry_limit_price") or entry_price
    sl_p = plan.get("stop_loss_premium") or stoploss
    tp_p = plan.get("target_premium") or target
    score = plan.get("entry_confirmation_score")
    trig = plan.get("entry_spot_trigger")

    steps.append(
        {
            "title": "Strategy",
            "detail": (
                f"{sid}"
                + (f" · {plan.get('pattern_tag')}" if plan.get("pattern_tag") else "")
                + (f" · {opt} strike {plan.get('strike')}" if plan.get("strike") else "")
                + (f" · {plan.get('strike_moneyness', '')}".rstrip())
            ).strip(" ·"),
        }
    )

    bb_parts = []
    for k, label in (
        ("bb_lower", "lower"),
        ("bb_middle", "middle"),
        ("bb_upper", "upper"),
    ):
        if ind.get(k) is not None:
            bb_parts.append(f"{label} ₹{_fmt_num(ind[k])}")
    if bb_parts:
        zone = ind.get("bb_zone") or ""
        steps.append(
            {
                "title": f"Bollinger (5m on {contract_sym})",
                "detail": " · ".join(bb_parts)
                + (f" · zone {zone}" if zone else "")
                + (f" · LTP ₹{_fmt_num(prem)}" if prem else "")
                + (f" · {plan.get('level_note')}" if plan.get("level_note") else ""),
            }
        )

    entry_detail = (
        f"LIMIT ₹{_fmt_num(limit)} ({style or 'patient limit'})"
        + (f" · fair ₹{_fmt_num(fair)}" if fair else "")
        + (f" · LTP ₹{_fmt_num(prem)}" if prem else "")
        + (
            f" · book bid/ask ₹{_fmt_num(ind.get('option_bid'))}/₹{_fmt_num(ind.get('option_ask'))}"
            if ind.get("option_bid")
            else ""
        )
    )
    if trig is not None:
        entry_detail += f" · band trigger ₹{_fmt_num(trig)}"
    if score is not None:
        entry_detail += f" · score {score}"
    steps.append({"title": "Entry price", "detail": entry_detail})

    if spot_sl is not None and spot_tp is not None and (
        sid == "bb_5m_mean_reversion" or ind.get("bb_on_contract")
    ):
        steps.append(
            {
                "title": "SL / target (option premium from contract BB)",
                "detail": (
                    f"PE: SL above upper band + buffer → ₹{_fmt_num(spot_sl)} · "
                    f"target toward middle → ₹{_fmt_num(spot_tp)}."
                    if opt == "PE"
                    else (
                        f"CE: SL below lower band + buffer → ₹{_fmt_num(spot_sl)} · "
                        f"target toward middle → ₹{_fmt_num(spot_tp)}."
                    )
                )
                + (f" Entry LTP ₹{_fmt_num(prem)}." if prem else ""),
            }
        )
    elif spot_sl is not None and spot_tp is not None:
        steps.append(
            {
                "title": "Spot SL / target",
                "detail": (
                    f"SL {_fmt_num(spot_sl, 0)} → target {_fmt_num(spot_tp, 0)}."
                    + (f" {plan.get('level_note')}" if plan.get("level_note") else "")
                ),
            }
        )

    if (
        delta is not None
        and sid != "bb_5m_mean_reversion"
        and spot_sl is not None
        and spot_tp is not None
    ):
        try:
            nifty = ind.get("nifty_spot")
            if nifty:
                spot_risk = abs(float(nifty) - float(spot_sl))
                spot_reward = abs(float(spot_tp) - float(nifty))
                steps.append(
                    {
                        "title": "Option SL / target (delta map)",
                        "detail": (
                            f"δ ≈ {_fmt_num(delta, 3)} · index risk {spot_risk:.1f} pts → "
                            f"premium SL ₹{_fmt_num(sl_p)} · reward {spot_reward:.1f} pts → "
                            f"premium TP ₹{_fmt_num(tp_p)}."
                        ),
                    }
                )
        except (TypeError, ValueError):
            pass
    elif sl_p is not None and tp_p is not None:
        steps.append(
            {
                "title": "Option SL / target",
                "detail": f"Stop-loss ₹{_fmt_num(sl_p)} · Target ₹{_fmt_num(tp_p)}.",
            }
        )

    if plan.get("note"):
        steps.append({"title": "Note", "detail": str(plan["note"])})
    if plan.get("captured_at"):
        steps.append(
            {
                "title": "Snapshot time",
                "detail": f"Indicators/plan captured at placement: {plan['captured_at']}.",
            }
        )
    return steps


def get_paper_trade_detail(order_id: str) -> Dict[str, Any]:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    cur = conn.execute(
        """
        SELECT id, created_at, order_id, payload, status,
               stoploss, target, trailing_stoploss,
               exit_reason, exit_price, exit_at
        FROM paper_orders
        WHERE order_id = ?
        LIMIT 1
        """,
        (order_id,),
    )
    row = cur.fetchone()
    if not row:
        return {"error": "not_found", "message": f"No paper order {order_id!r}"}

    d = {k: row[k] for k in row.keys()}
    payload = _parse_payload(d.get("payload"))
    if payload.get("paper_exit_leg"):
        return {"error": "exit_leg", "message": "Detail is only available for entry orders."}

    segment = payload.get("segment")
    if not segment:
        segment = infer_segment_from_order(
            str(payload.get("exchange") or ""),
            str(payload.get("tradingsymbol") or ""),
        )
    segment = normalize_segment(str(segment))
    symbol = str(payload.get("tradingsymbol") or "")
    exchange = str(payload.get("exchange") or "")

    entry_px = payload.get("paper_fill_price") or payload.get("price")
    try:
        entry_px = float(entry_px) if entry_px is not None else None
    except (TypeError, ValueError):
        entry_px = None

    at_entry = payload.get("paper_trade_plan")
    live = _live_indicators_for_segment(
        segment,
        tradingsymbol=symbol,
        exchange=exchange,
    )
    at_entry_table: List[Dict[str, str]] = []
    if at_entry:
        ind = at_entry.get("indicators") or {}
        at_entry_table = _indicator_table(
            {
                **ind,
                "underlying_label": live.get("underlying_label"),
                "contract_label": symbol,
                "spot": ind.get("nifty_spot") or ind.get("underlying_spot"),
                "option_ltp": ind.get("option_ltp") or entry_px,
                "indicator_window": ind.get("indicator_window")
                or {"period": 20, "timeframe": "5m", "std_dev": 2},
            }
        )

    return {
        "order_id": order_id,
        "segment": segment,
        "symbol": symbol,
        "entry_price": entry_px,
        "stoploss": d.get("stoploss"),
        "target": d.get("target"),
        "live_indicators": live,
        "live_indicator_rows": _indicator_table(live),
        "at_entry_plan": at_entry,
        "at_entry_indicator_rows": at_entry_table,
        "pricing_reason": _build_pricing_reason(
            at_entry,
            entry_price=entry_px,
            stoploss=d.get("stoploss"),
            target=d.get("target"),
            symbol=symbol,
        ),
        "compare_hint": live.get("compare_hint"),
    }
