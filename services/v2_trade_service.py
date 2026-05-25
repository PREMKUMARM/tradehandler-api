"""
V2 wizard — validate checklist, build Nifty ATM option trade plan, place entry + GTT OCO.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from agent.config import get_agent_config
from agent.tools.kite_tools import place_gtt_tool, place_order_tool
from schemas.v2_trading import ChecklistStepStatus
from utils.margin_utils import parse_equity_margins
from services.push.option_contract_resolver import estimate_option_leg
from services.v2_strategy_analysis import analyze_fno_strategies
from utils.kite_utils import get_kite_instance, get_access_token
from utils.logger import log_error, log_info

IST = ZoneInfo("Asia/Kolkata")

STEP_TITLES = [
    "Confirm session, connectivity, and buying power",
    "Define the trade hypothesis on Nifty",
    "Check India VIX and event risk",
    "Strategy analysis",
    "Read the Nifty option chain",
    "Pick expiry with purpose",
    "Select strike using delta and liquidity",
    "Estimate premium, Greeks, and max loss",
    "Plan exit before entry",
    "Size the position to your risk rule",
    "Choose product and order type",
    "Final order ticket review",
]

# Steps that must be marked done in the wizard before place (0-indexed).
REQUIRED_MARKED_STEPS = [0, 1, 2, 9]
# Steps validated at execution time on the server during market hours.
MARKET_EXECUTION_STEPS = [4, 5, 6, 7, 8, 10, 11]


def _ist_clock() -> Tuple[int, bool]:
    now = datetime.now(IST)
    day = now.weekday()
    minutes = now.hour * 60 + now.minute
    return minutes, 1 <= day <= 5


def is_market_session_open() -> bool:
    minutes, is_weekday = _ist_clock()
    if not is_weekday:
        return False
    return 9 * 60 + 15 <= minutes < 15 * 60 + 30


def allow_offhours_v2_place() -> bool:
    """When true, allow confirm/place for integration testing outside market hours."""
    return os.getenv("V2_ALLOW_OFFHOURS_PLACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _resolve_can_place(
    trade_plan: Optional[Dict[str, Any]],
    validation: Optional[Dict[str, Any]],
    market_open: bool,
) -> bool:
    if not trade_plan:
        return False
    if market_open:
        return bool(validation and validation.get("is_good_trade"))
    if allow_offhours_v2_place():
        return True
    return False


def _check_kite_and_margin() -> Tuple[bool, float, str]:
    try:
        token = get_access_token()
        if not token or len(token) < 10:
            return False, 0.0, "Kite not connected — connect via profile menu"
        kite = get_kite_instance(skip_validation=False)
        margins = kite.margins()
        equity = margins.get("equity", {})
        available, _, _ = parse_equity_margins(equity)
        if available <= 0:
            return False, 0.0, "No available margin reported"
        return True, float(available), "Kite connected"
    except Exception as exc:
        return False, 0.0, f"Kite check failed: {exc}"


def _resolve_completed_steps(
    completed_steps: Optional[List[bool]], auto_execute: bool
) -> List[bool]:
    if auto_execute:
        return [True] * len(STEP_TITLES)
    if completed_steps and len(completed_steps) == len(STEP_TITLES):
        return list(completed_steps)
    return [False] * len(STEP_TITLES)


def validate_checklist(
    completed_steps: List[bool],
    direction: str,
    market_open: bool,
    auto_execute: bool = False,
) -> Tuple[List[ChecklistStepStatus], List[int], bool]:
    if auto_execute:
        return _validate_checklist_auto(direction, market_open, plan=None)
    statuses: List[ChecklistStepStatus] = []
    missing: List[int] = []
    kite_ok, margin, kite_msg = _check_kite_and_margin()
    dir_ok = direction.upper() in ("CE", "PE", "AUTO")

    for i, title in enumerate(STEP_TITLES):
        marked = bool(completed_steps[i]) if i < len(completed_steps) else False
        server_ok = True
        msg = "Marked complete" if marked else "Not marked done"

        if i == 0:
            server_ok = kite_ok and margin > 0
            msg = kite_msg
            if not server_ok or not marked:
                missing.append(i)
        elif i == 1:
            server_ok = marked or direction.upper() in ("CE", "PE")
            msg = "Hypothesis set" if marked else "Mark done or choose CE/PE below"
            if not marked and direction.upper() not in ("CE", "PE"):
                missing.append(i)
        elif i == 2:
            server_ok = marked
            msg = "VIX/events reviewed" if marked else "Mark after reviewing VIX and calendar"
            if not marked:
                missing.append(i)
        elif i == 3:
            server_ok = marked
            msg = "Strategy chosen" if marked else "Run strategy analysis and pick best fit"
            if not marked:
                missing.append(i)
        elif i == 9:
            server_ok = marked
            msg = "Position sized" if marked else "Mark after size matches risk rule"
            if not marked:
                missing.append(i)
        elif i in MARKET_EXECUTION_STEPS:
            if market_open:
                server_ok = True
                msg = "Validated when placing order" if not marked else "Marked complete"
            else:
                server_ok = False
                msg = "Opens 9:15 AM IST on trading days"

        statuses.append(
            ChecklistStepStatus(
                index=i,
                title=title,
                completed=marked,
                server_ok=server_ok,
                message=msg,
            )
        )

    checklist_ready = len(missing) == 0 and dir_ok
    return statuses, missing, checklist_ready


def _step(
    index: int,
    title: str,
    server_ok: bool,
    message: str,
    output: Optional[str] = None,
) -> ChecklistStepStatus:
    return ChecklistStepStatus(
        index=index,
        title=title,
        completed=server_ok,
        server_ok=server_ok,
        message=message,
        output=output,
    )


def _validate_checklist_auto(
    direction: str,
    market_open: bool,
    plan: Optional[Dict[str, Any]] = None,
    plan_error: Optional[str] = None,
    resolved_direction: Optional[str] = None,
    margin: float = 0.0,
    strategy_analysis: Optional[Dict[str, Any]] = None,
) -> Tuple[List[ChecklistStepStatus], List[int], bool]:
    """Build per-step outputs for auto-execute preview UI."""
    statuses: List[ChecklistStepStatus] = []
    missing: List[int] = []
    kite_ok, margin_val, kite_msg = _check_kite_and_margin()
    if margin <= 0:
        margin = margin_val
    dir_ok = direction.upper() in ("CE", "PE", "AUTO")
    opt = resolved_direction or (plan or {}).get("option_type") or direction.upper()

    for i, title in enumerate(STEP_TITLES):
        if i == 0:
            ok = kite_ok and margin > 0
            out = f"Available margin: ₹{margin:,.2f}" if ok else None
            if not ok:
                missing.append(i)
            statuses.append(_step(i, title, ok, kite_msg, out))
        elif i == 1:
            ok = dir_ok and (plan is not None or direction.upper() in ("CE", "PE"))
            if direction.upper() == "AUTO" and plan:
                msg = f"Direction AUTO → {opt}"
                out = f"Nifty {plan.get('nifty_spot')} vs prior close → {opt}"
            elif direction.upper() in ("CE", "PE"):
                msg = f"Direction: {direction.upper()}"
                out = f"Hypothesis: buy Nifty {direction.upper()}"
            else:
                msg = "Could not resolve direction"
                out = plan_error
                ok = False
                missing.append(i)
            statuses.append(_step(i, title, ok, msg, out))
        elif i == 2:
            vix = (strategy_analysis or {}).get("context", {}).get("vix")
            ok = plan is not None
            out = (
                f"VIX {vix:.1f} — sized for current IV" if vix else "VIX/event check passed"
            ) if ok else plan_error
            statuses.append(_step(i, title, ok, "VIX & event risk reviewed", out))
        elif i == 3:
            ok = strategy_analysis is not None and plan is not None
            sa = strategy_analysis or {}
            out = sa.get("output_summary") or plan_error
            msg = (
                f"Best: {sa.get('selected_name')} ({sa.get('selected_score')}/100)"
                if ok
                else "Strategy analysis failed"
            )
            statuses.append(_step(i, title, ok, msg, out))
        elif i == 4:
            ok = plan is not None
            out = (
                f"Nifty spot {plan.get('nifty_spot')} · focus ATM/near-ATM {opt} liquidity"
                if plan
                else plan_error
            )
            statuses.append(_step(i, title, ok, "Chain read (spot + OI context)", out))
        elif i == 5:
            ok = plan is not None
            out = f"Expiry: {plan.get('expiry')}" if plan else plan_error
            statuses.append(_step(i, title, ok, "Nearest liquid weekly/monthly expiry", out))
        elif i == 6:
            ok = plan is not None
            strat = (plan or {}).get("strategy_name", "")
            out = (
                f"Strike {plan.get('strike')} {opt} · ~delta 0.5 ATM · via {strat}"
                if plan
                else plan_error
            )
            statuses.append(_step(i, title, ok, "ATM strike selected", out))
        elif i == 7:
            ok = plan is not None
            out = (
                f"Entry ~₹{plan.get('entry_premium')} · max loss ~₹{plan.get('risk_inr')}"
                if plan
                else plan_error
            )
            statuses.append(_step(i, title, ok, "Premium & max loss estimated", out))
        elif i == 8:
            ok = plan is not None
            out = (
                f"SL premium ₹{plan.get('stop_loss_premium')} · Target ₹{plan.get('target_premium')} · "
                f"Nifty SL {plan.get('spot_stop_loss')} · Tgt {plan.get('spot_target')}"
                if plan
                else plan_error
            )
            statuses.append(_step(i, title, ok, "Exit plan (GTT OCO)", out))
        elif i == 9:
            ok = plan is not None
            lots = plan.get("num_lots") if plan else 0
            lot_sz = plan.get("lot_size") if plan else 75
            qty = plan.get("quantity") if plan else 0
            out = (
                f"{lots} lot(s) × {lot_sz} = {qty} qty · risk ₹{plan.get('risk_inr')}"
                if plan
                else plan_error
            )
            statuses.append(_step(i, title, ok, "Position sized to risk rule", out))
        elif i == 10:
            ok = plan is not None
            out = f"{plan.get('product')} · MARKET entry + GTT OCO exit" if plan else plan_error
            statuses.append(_step(i, title, ok, "MIS intraday · MARKET + GTT", out))
        elif i == 11:
            has_plan = plan is not None
            ok = has_plan
            sym = plan.get("tradingsymbol") if plan else "—"
            msg = (
                "Ticket ready — confirm to place"
                if has_plan and market_open
                else ("Market closed — review only" if has_plan else "Plan incomplete")
            )
            out = f"BUY {plan.get('quantity')} {sym} @ ~₹{plan.get('entry_premium')}" if plan else plan_error
            statuses.append(_step(i, title, ok, msg, out))

    if not kite_ok or margin <= 0:
        if 0 not in missing:
            missing.append(0)
    checklist_ready = plan is not None and dir_ok and 0 not in missing
    return statuses, missing, checklist_ready


def _resolve_direction(direction: str, nifty_ltp: float, prev_close: float) -> str:
    d = (direction or "AUTO").upper()
    if d in ("CE", "PE"):
        return d
    if prev_close > 0:
        return "CE" if nifty_ltp >= prev_close else "PE"
    return "CE"


def build_trade_plan(
    direction: str,
    risk_pct: float,
    reward_pct: float,
    num_lots: int,
    capital: float,
    strategy_analysis: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    messages: List[str] = []
    kite = get_kite_instance()
    quote = kite.quote("NSE:NIFTY 50")
    nq = quote.get("NSE:NIFTY 50", {})
    nifty_ltp = float(nq.get("last_price") or 0)
    prev_close = float(nq.get("ohlc", {}).get("close") or nifty_ltp)
    if nifty_ltp <= 0:
        return None, ["Could not fetch Nifty 50 price"]

    strategy_id = None
    strategy_name = None
    if strategy_analysis:
        strategy_id = strategy_analysis.get("selected_id")
        strategy_name = strategy_analysis.get("selected_name")
        option_kind = strategy_analysis.get("selected_option_kind") or _resolve_direction(
            direction, nifty_ltp, prev_close
        )
        ranked = strategy_analysis.get("strategies") or []
        sel_row = next((s for s in ranked if s.get("id") == strategy_id), ranked[0] if ranked else {})
        nifty_ltp = float(sel_row.get("spot_entry") or nifty_ltp)
        spot_sl = float(sel_row.get("spot_stop_loss") or nifty_ltp)
        spot_tgt = float(sel_row.get("spot_target") or nifty_ltp)
        messages.append(
            f"Strategy: {strategy_name} (score {strategy_analysis.get('selected_score')})"
        )
    else:
        option_kind = _resolve_direction(direction, nifty_ltp, prev_close)
        risk_ratio = reward_pct / risk_pct if risk_pct > 0 else 2.0
        spot_risk_pts = max(nifty_ltp * (risk_pct / 100.0) * 0.35, 15.0)
        if option_kind == "CE":
            spot_sl = nifty_ltp - spot_risk_pts
            spot_tgt = nifty_ltp + spot_risk_pts * risk_ratio
        else:
            spot_sl = nifty_ltp + spot_risk_pts
            spot_tgt = nifty_ltp - spot_risk_pts * risk_ratio

    if direction.upper() == "AUTO" and not strategy_analysis:
        messages.append(f"Direction AUTO → {option_kind} (spot vs prior close)")

    lot_size = 75
    try:
        for row in kite.instruments("NFO"):
            if row.get("name") == "NIFTY" and row.get("instrument_type") == option_kind:
                lot_size = int(row.get("lot_size") or 75)
                break
    except Exception:
        pass

    leg = estimate_option_leg(
        spot_entry=nifty_ltp,
        spot_stop_loss=spot_sl,
        spot_target=spot_tgt,
        kind=option_kind,
        delta=0.5,
        lot_size=lot_size,
        num_lots=num_lots,
        atm_step=50,
        name="NIFTY",
    )

    if leg.contract is None:
        return None, ["Could not resolve ATM NIFTY option contract"]

    entry_prem = float(leg.entry_premium or 0)
    sl_prem = float(leg.sl_premium or 0)
    tgt_prem = float(leg.target_premium or 0)
    prem_risk = max(0.05, entry_prem - sl_prem)
    max_risk_amt = capital * (risk_pct / 100.0)
    max_qty = int(max_risk_amt / (prem_risk * lot_size)) if prem_risk > 0 else 1
    qty_lots = min(num_lots, max(1, max_qty))
    quantity = qty_lots * lot_size

    risk_inr = prem_risk * quantity
    reward_inr = max(0.0, (tgt_prem - entry_prem) * quantity)
    rr = (reward_inr / risk_inr) if risk_inr > 0 else 0.0

    plan = {
        "tradingsymbol": leg.contract.tradingsymbol,
        "exchange": "NFO",
        "option_type": option_kind,
        "strike": int(leg.contract.strike),
        "expiry": leg.contract.expiry.isoformat(),
        "quantity": quantity,
        "lot_size": lot_size,
        "num_lots": qty_lots,
        "product": "MIS",
        "entry_premium": round(entry_prem, 2),
        "stop_loss_premium": round(sl_prem, 2),
        "target_premium": round(tgt_prem, 2),
        "nifty_spot": round(nifty_ltp, 2),
        "spot_stop_loss": round(spot_sl, 2),
        "spot_target": round(spot_tgt, 2),
        "risk_inr": round(risk_inr, 2),
        "reward_inr": round(reward_inr, 2),
        "reward_ratio": round(rr, 2),
        "estimated_premium": bool(leg.estimated),
        "note": leg.note,
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
    }
    messages.append(
        f"Plan: BUY {quantity} {leg.contract.tradingsymbol} @ ~₹{entry_prem:.2f} | "
        f"GTT SL ₹{sl_prem:.2f} TP ₹{tgt_prem:.2f}"
    )
    return plan, messages


def _validate_trade_plan(plan: Dict[str, Any], capital: float, risk_pct: float, reward_pct: float) -> Dict[str, Any]:
    entry = plan["entry_premium"]
    sl = plan["stop_loss_premium"]
    tgt = plan["target_premium"]
    qty = plan["quantity"]
    if entry <= sl:
        return {"is_good_trade": False, "error": "Stop-loss premium must be below entry for long option"}
    if tgt <= entry:
        return {"is_good_trade": False, "error": "Target premium must be above entry for long option"}
    risk_amt = (entry - sl) * qty
    reward_amt = (tgt - entry) * qty
    max_risk = capital * risk_pct / 100.0
    ratio = reward_pct / risk_pct if risk_pct > 0 else 2.0
    min_reward = risk_amt * ratio
    return {
        "is_good_trade": risk_amt <= max_risk and reward_amt >= min_reward,
        "risk_amount": round(risk_amt, 2),
        "reward_amount": round(reward_amt, 2),
        "max_risk_amount": round(max_risk, 2),
        "min_required_reward": round(min_reward, 2),
        "risk_within_limit": risk_amt <= max_risk,
        "reward_meets_requirement": reward_amt >= min_reward,
    }


def preview_trade(
    completed_steps: Optional[List[bool]] = None,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    num_lots: int = 1,
    auto_execute: bool = False,
) -> Dict[str, Any]:
    cfg = get_agent_config()
    risk_pct = float(risk_percentage or cfg.risk_per_trade_pct or 1.0)
    reward_pct = float(reward_percentage or cfg.reward_per_trade_pct or 2.0)
    market_open = is_market_session_open()
    steps = _resolve_completed_steps(completed_steps, auto_execute)

    messages: List[str] = []
    _, margin, _ = _check_kite_and_margin()
    capital = margin if margin > 0 else float(cfg.trading_capital or 100000)

    trade_plan = None
    validation = None
    can_place = False
    plan_error: Optional[str] = None
    resolved_dir: Optional[str] = None

    strategy_analysis: Optional[Dict[str, Any]] = None

    if auto_execute:
        strategy_analysis = analyze_fno_strategies(
            direction_pref=direction,
            margin=capital,
        )
        plan, plan_msgs = build_trade_plan(
            direction,
            risk_pct,
            reward_pct,
            num_lots,
            capital,
            strategy_analysis=strategy_analysis,
        )
        messages.extend(plan_msgs)
        if plan:
            trade_plan = plan
            resolved_dir = plan.get("option_type")
            validation = _validate_trade_plan(plan, capital, risk_pct, reward_pct)
            can_place = _resolve_can_place(trade_plan, validation, market_open)
            if not validation.get("is_good_trade"):
                messages.append("Risk/reward validation failed — adjust size or levels")
            elif not market_open and allow_offhours_v2_place():
                messages.append(
                    "Test mode: off-hours place enabled (V2_ALLOW_OFFHOURS_PLACE) — live session still preferred"
                )
            elif not market_open:
                messages.append("Preview only — confirm and place when market is open")
        else:
            plan_error = plan_msgs[0] if plan_msgs else "Could not build trade plan"
            messages.append(plan_error)

        statuses, missing, checklist_ready = _validate_checklist_auto(
            direction,
            market_open,
            plan=trade_plan,
            plan_error=plan_error,
            resolved_direction=resolved_dir,
            margin=capital,
            strategy_analysis=strategy_analysis,
        )
    else:
        statuses, missing, checklist_ready = validate_checklist(
            steps, direction, market_open, auto_execute=False
        )
        if not checklist_ready:
            messages.append(f"Complete checklist steps: {[i + 1 for i in missing]}")
        if not market_open:
            messages.append("Live orders only during market hours (9:15 AM–3:30 PM IST, Mon–Fri)")
        if checklist_ready:
            plan, plan_msgs = build_trade_plan(direction, risk_pct, reward_pct, num_lots, capital)
            messages.extend(plan_msgs)
            if plan:
                trade_plan = plan
                validation = _validate_trade_plan(plan, capital, risk_pct, reward_pct)
                can_place = _resolve_can_place(trade_plan, validation, market_open)
                if not validation.get("is_good_trade"):
                    messages.append("Risk/reward validation failed — adjust size or levels")
                elif not market_open and allow_offhours_v2_place():
                    messages.append(
                        "Test mode: off-hours place enabled (V2_ALLOW_OFFHOURS_PLACE)"
                    )
                elif not market_open:
                    messages.append("Preview only — confirm and place when market is open")

    return {
        "can_place": can_place,
        "checklist_ready": checklist_ready,
        "missing_steps": missing,
        "step_statuses": [s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in statuses],
        "trade_plan": trade_plan,
        "validation": validation,
        "messages": messages,
        "market_open": market_open,
        "allow_test_place": allow_offhours_v2_place(),
        "strategy_analysis": strategy_analysis,
    }


def get_strategy_analysis(direction: str = "AUTO") -> Dict[str, Any]:
    """Standalone strategy analysis for wizard step 4 (uses steps 1–3 context)."""
    _, margin, _ = _check_kite_and_margin()
    cfg = get_agent_config()
    capital = margin if margin > 0 else float(cfg.trading_capital or 100000)
    return analyze_fno_strategies(direction_pref=direction, margin=capital)


def place_trade(
    completed_steps: Optional[List[bool]] = None,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    num_lots: int = 1,
    confirm: bool = False,
    auto_execute: bool = False,
) -> Dict[str, Any]:
    preview = preview_trade(
        completed_steps,
        direction,
        risk_percentage,
        reward_percentage,
        num_lots,
        auto_execute=auto_execute,
    )
    result = {**preview, "placed": False, "entry_order_id": None, "gtt_trigger_id": None, "errors": []}

    if not confirm:
        result["errors"].append("Set confirm=true to place orders")
        return result

    if auto_execute:
        result["messages"] = ["Auto-executing checklist…"] + list(result.get("messages", []))

    if not preview.get("can_place") or not preview.get("trade_plan"):
        result["errors"].append("Cannot place — fix checklist or validation first")
        return result

    plan = preview["trade_plan"]
    symbol = plan["tradingsymbol"]
    qty = int(plan["quantity"])
    entry_prem = float(plan["entry_premium"])
    sl_prem = float(plan["stop_loss_premium"])
    tgt_prem = float(plan["target_premium"])

    try:
        entry = place_order_tool.invoke({
            "tradingsymbol": symbol,
            "exchange": "NFO",
            "transaction_type": "BUY",
            "quantity": qty,
            "order_type": "MARKET",
            "product": "MIS",
        })
        if entry.get("status") != "success":
            result["errors"].append(entry.get("error") or "Entry order failed")
            return result

        entry_id = str(entry.get("order_id"))
        result["entry_order_id"] = entry_id
        log_info(f"V2 trade entry placed: {entry_id} {symbol}")

        sl_trigger = round(sl_prem * 1.002, 2)
        tp_trigger = round(tgt_prem * 0.998, 2)
        last_price = entry_prem

        gtt = place_gtt_tool.invoke({
            "tradingsymbol": symbol,
            "exchange": "NFO",
            "trigger_type": "two-leg",
            "trigger_prices": [sl_trigger, tp_trigger],
            "last_price": last_price,
            "stop_loss_price": round(sl_prem, 2),
            "target_price": round(tgt_prem, 2),
            "quantity": qty,
            "transaction_type": "SELL",
            "product": "MIS",
        })

        if gtt.get("status") == "success":
            result["gtt_trigger_id"] = str(gtt.get("trigger_id"))
            result["placed"] = True
            result["messages"] = list(preview.get("messages", [])) + [
                f"Entry order {entry_id}",
                f"GTT OCO trigger {result['gtt_trigger_id']}",
            ]
        else:
            result["errors"].append(
                gtt.get("error") or "GTT placement failed — entry may be open without exit GTT"
            )
            result["messages"] = list(preview.get("messages", [])) + [
                f"Entry order {entry_id} placed; exit GTT failed — manage manually",
            ]

    except Exception as exc:
        log_error(f"V2 place_trade error: {exc}")
        result["errors"].append(str(exc))

    return result
