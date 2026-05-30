"""Crypto (Binance BTCUSDT perp) wizard — checklist, preview, place."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.crypto_config import (
    DEFAULT_LEVERAGE,
    MIN_LIVE_USDT_BALANCE,
    SYMBOL,
    allow_offhours_crypto_place,
    is_crypto_session_open,
)
from services.crypto_indicator_plan import build_trade_plan, refresh_plan_at_execution
from services.crypto_live_indicators import recalculate_from_ticker
from utils.binance_order_utils import (
    get_binance_credentials,
    get_usdt_balance,
    place_limit_order,
    place_stop_market,
    set_margin_type,
)
from utils.logger import log_error, log_info

STEP_TITLES = [
    "Binance API & USDT balance",
    "Market session (24/7)",
    "Direction hypothesis (LONG / SHORT)",
    "Structure: OR / PDH / PDL / VWAP",
    "Entry setup (5m BB)",
    "Position size & leverage",
    "Stop loss & take profit",
    "Final order review",
]

REQUIRED_MARKED_STEPS = [0, 2, 5, 7]


def _step_statuses_from_live(
  plan: Dict[str, Any],
  live: Dict[str, Any],
  balance: float,
  *,
  paper_mode: bool = False,
) -> List[Dict[str, Any]]:
    side = plan.get("side") or "LONG"
    spot = float(live.get("btc_spot") or 0)
    out: List[Dict[str, Any]] = []
    min_bal = MIN_LIVE_USDT_BALANCE if not paper_mode else 0.0
    checks = [
        balance >= min_bal or (paper_mode and bool(live.get("connected"))),
        is_crypto_session_open(),
        side in ("LONG", "SHORT"),
        spot > 0 and live.get("or_high"),
        bool(plan.get("entry_ready")),
        float(plan.get("quantity") or 0) > 0,
        float(plan.get("stop_loss_premium") or 0) > 0,
        float(plan.get("entry_limit_price") or 0) > 0,
    ]
    for i, title in enumerate(STEP_TITLES):
        ok = bool(checks[i]) if i < len(checks) else False
        out.append(
            {
                "index": i,
                "title": title,
                "completed": ok,
                "server_ok": ok,
                "message": "Pass" if ok else "Needs attention",
                "output": "",
            }
        )
    return out


def get_checklist_live(
    *,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    quantity_btc: Optional[float] = None,
) -> Dict[str, Any]:
    paper_mode = False
    try:
        from services.paper_trading import is_paper_mode_for_segment

        paper_mode = is_paper_mode_for_segment("crypto")
    except Exception:
        pass

    connected = False
    balance = 0.0
    msg = ""
    if paper_mode:
        try:
            from services.paper_funds import get_available_balance

            balance = float(get_available_balance("crypto") or 0)
            connected = True
            msg = f"Paper mode · ${balance:,.2f} USDT available (public Binance quotes)"
        except Exception as exc:
            balance = 0.0
            connected = True
            msg = f"Paper mode · fund lookup: {exc}"
    else:
        try:
            get_binance_credentials()
            balance = get_usdt_balance()
            connected = True
            msg = f"Binance connected · USDT available ${balance:,.2f}"
        except Exception as exc:
            balance = 0.0
            connected = False
            msg = str(exc)

    plan, messages = build_trade_plan(
        direction=direction,
        risk_percentage=risk_percentage,
        reward_percentage=reward_percentage,
        quantity_btc=quantity_btc,
    )
    live = plan.get("indicators") or recalculate_from_ticker()
    if not plan and live.get("connected"):
        connected = True
    steps = _step_statuses_from_live(plan, live, balance, paper_mode=paper_mode)
    missing = [s["index"] for s in steps if not s["server_ok"]]
    ready = len(missing) == 0

    if paper_mode:
        messages.append("Paper mode ON (Crypto) — simulated Binance fills in paper ledger")

    from services.watch_execute import resolve_can_execute

    preview_stub = {
        "checklist_ready": ready,
        "can_place": ready and connected and bool(plan) and not paper_mode,
        "paper_trading_mode": paper_mode,
        "trade_plan": plan or None,
    }
    can_execute = resolve_can_execute(preview_stub, plan)

    return {
        "connected": connected,
        "message": msg,
        "checklist_ready": ready,
        "missing_steps": missing,
        "step_statuses": steps,
        "trade_plan": plan or None,
        "can_place": ready and connected and bool(plan) and not paper_mode,
        "can_execute": can_execute,
        "market_open": is_crypto_session_open(),
        "allow_test_place": allow_offhours_crypto_place(),
        "paper_trading_mode": paper_mode,
        "messages": messages,
        "binance_balance_usdt": balance,
        "leverage": DEFAULT_LEVERAGE,
    }


def get_strategy_analysis(direction: str = "AUTO") -> Dict[str, Any]:
    plan, _ = build_trade_plan(direction=direction)
    side = plan.get("side") or "LONG"
    sid = plan.get("strategy_id") or "bb_5m_mean_reversion"
    return {
        "selected_id": sid,
        "selected_name": f"BTCUSDT {side} BB 5m",
        "selected_score": int(plan.get("entry_confirmation_score") or 60),
        "selected_fit": "good" if plan.get("entry_ready") else "wait",
        "selected_option_kind": side,
        "strategies": [
            {
                "id": sid,
                "name": f"BTCUSDT {side} BB 5m {DEFAULT_LEVERAGE}x",
                "score": int(plan.get("entry_confirmation_score") or 60),
                "fit": "good" if plan.get("entry_ready") else "wait",
                "option_kind": side,
            }
        ],
        "output_summary": plan.get("note") or f"{SYMBOL} {side} perp · 5m BB mean reversion",
    }


def get_checklist_analyze(
    *,
    step: int,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    quantity_btc: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Parity endpoint with Nifty/Commodity: return live checklist + highlight one step.
    (Crypto MVP runs auto_execute checklist; step analysis is informational.)
    """
    data = get_checklist_live(
        direction=direction,
        risk_percentage=risk_percentage,
        reward_percentage=reward_percentage,
        quantity_btc=quantity_btc,
    )
    try:
        idx = int(step)
    except Exception:
        idx = 0
    rows = list(data.get("step_statuses") or [])
    focus = next((r for r in rows if int(r.get("index", -1)) == idx), None)
    if focus:
        data["messages"] = list(data.get("messages", [])) + [
            f"Analyze step {idx}: {focus.get('title')} — {focus.get('message')}"
        ]
    return data


def preview_trade(
    completed_steps: Optional[List[bool]] = None,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    quantity_btc: Optional[float] = None,
    auto_execute: bool = False,
) -> Dict[str, Any]:
    live_data = get_checklist_live(
        direction=direction,
        risk_percentage=risk_percentage,
        reward_percentage=reward_percentage,
        quantity_btc=quantity_btc,
    )
    plan = live_data.get("trade_plan") or {}
    can_place = bool(live_data.get("can_place"))
    if completed_steps and not auto_execute:
        for idx in REQUIRED_MARKED_STEPS:
            if idx < len(completed_steps) and not completed_steps[idx]:
                can_place = False
    can_execute = bool(live_data.get("can_execute"))
    from services.watch_execute import resolve_can_execute

    can_execute = resolve_can_execute(live_data, plan)
    return {
        **live_data,
        "can_execute": can_execute,
        "validation": {
            "is_good_trade": can_execute,
            "risk_amount": plan.get("risk_inr"),
            "reward_amount": plan.get("reward_inr"),
        },
    }


def place_trade(
    completed_steps: Optional[List[bool]] = None,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    quantity_btc: Optional[float] = None,
    confirm: bool = False,
    auto_execute: bool = False,
    trade_plan_snapshot: Optional[Dict[str, Any]] = None,
    defer_exits_until_fill: bool = True,
) -> Dict[str, Any]:
    preview = preview_trade(
        completed_steps,
        direction,
        risk_percentage,
        reward_percentage,
        quantity_btc,
        auto_execute=auto_execute,
    )
    result = {
        **preview,
        "placed": False,
        "entry_order_id": None,
        "sl_order_id": None,
        "tp_order_id": None,
        "errors": [],
        "entry_paper": False,
    }

    if not confirm:
        result["errors"].append("Set confirm=true to place orders")
        return result

    plan = trade_plan_snapshot or preview.get("trade_plan")
    if not plan:
        result["errors"].append("No trade plan")
        return result

    plan = refresh_plan_at_execution(plan)
    result["trade_plan"] = plan

    if not preview.get("can_execute") and not (confirm and preview.get("checklist_ready")):
        result["errors"].append("Checklist not ready")
        return result

    side = str(plan.get("side") or "LONG").upper()
    order_side = "BUY" if side == "LONG" else "SELL"
    exit_side = "SELL" if side == "LONG" else "BUY"
    qty = float(plan.get("quantity") or 0)
    entry_limit = float(plan.get("entry_limit_price") or 0)
    sl_px = float(plan.get("stop_loss_premium") or 0)
    tp_px = float(plan.get("target_premium") or 0)

    if qty <= 0:
        result["errors"].append("Position size is zero — insufficient USDT for min BTCUSDT lot at current leverage")
        return result

    try:
        from services.paper_trading import is_paper_mode_for_segment, paper_place_order

        paper_mode = is_paper_mode_for_segment("crypto")
        if paper_mode:
            from services.paper_trade_detail import slim_trade_plan_for_paper

            payload = {
                "tradingsymbol": SYMBOL,
                "exchange": "BINANCE",
                "transaction_type": order_side,
                "quantity": qty,
                "order_type": "LIMIT",
                "price": entry_limit,
                "product": "ISOLATED",
                "segment": "crypto",
                "leverage": DEFAULT_LEVERAGE,
                "stoploss": sl_px,
                "target": tp_px,
                "paper_fill_price": entry_limit,
                "paper_trade_plan": slim_trade_plan_for_paper(plan),
            }
            entry_id = paper_place_order(payload)
            result["entry_order_id"] = entry_id
            result["entry_paper"] = True
            result["placed"] = True
            result["messages"] = list(preview.get("messages", [])) + [
                f"[PAPER] Entry {order_side} LIMIT {qty} BTC @ ${entry_limit:,.2f} ({DEFAULT_LEVERAGE}x)",
                f"Paper SL ${sl_px:,.2f} · TP ${tp_px:,.2f} (monitor)",
            ]
            from services.risk_gate import record_order_placed

            record_order_placed(qty * entry_limit)
            return result

        set_margin_type(SYMBOL, "ISOLATED")
        try:
            live_usdt = float(get_usdt_balance() or 0)
            need_margin = (qty * entry_limit) / max(1, DEFAULT_LEVERAGE)
            if live_usdt + 0.01 < need_margin:
                result["errors"].append(
                    f"Insufficient margin: need ${need_margin:,.2f} USDT @ {DEFAULT_LEVERAGE}x "
                    f"for {qty} BTC, have ${live_usdt:,.2f}"
                )
                return result
        except Exception as exc:
            result["errors"].append(f"Binance balance check failed: {exc}")
            return result

        entry = place_limit_order(
            symbol=SYMBOL,
            side=order_side,
            quantity=qty,
            price=entry_limit,
            leverage=DEFAULT_LEVERAGE,
        )
        if not entry.get("ok"):
            result["errors"].append(entry.get("error") or "Entry failed")
            return result

        entry_id = entry.get("order_id")
        result["entry_order_id"] = entry_id
        result["placed"] = True
        result["messages"] = list(preview.get("messages", [])) + [
            f"Entry {order_side} LIMIT {qty} BTC @ ${entry_limit:,.2f} ({DEFAULT_LEVERAGE}x)",
        ]

        if not defer_exits_until_fill:
            sl = place_stop_market(
                symbol=SYMBOL, side=exit_side, quantity=qty, stop_price=sl_px
            )
            if sl.get("ok"):
                result["sl_order_id"] = sl.get("order_id")
                result["messages"].append(f"SL STOP_MARKET @ ${sl_px:,.2f}")
            tp = place_limit_order(
                symbol=SYMBOL,
                side=exit_side,
                quantity=qty,
                price=tp_px,
                reduce_only=True,
            )
            if tp.get("ok"):
                result["tp_order_id"] = tp.get("order_id")
                result["messages"].append(f"TP LIMIT @ ${tp_px:,.2f}")
        else:
            result["exits_deferred"] = True
            result["messages"].append("SL/TP will attach after entry fills (watch)")

        from services.risk_gate import record_order_placed

        record_order_placed(qty * entry_limit)
    except Exception as exc:
        log_error(f"[Crypto] place_trade: {exc}")
        result["errors"].append(str(exc))

    return result
