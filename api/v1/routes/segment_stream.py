"""
Generic segment WebSocket: reduce REST polling from browser.

Client connects to `/api/v1/ws/segment/{segment}` and receives push updates:
- watch_status (server push)
- watch_events (server push)
- balance (server push — live margin / USDT + spot)

Client may send commands (server responds on same WS):
{ "op": "arm" | "disarm" | "nuclear_reset" | "refresh" | "ping", "payload": {...} }
{ "op": "checklist_live" | "checklist_analyze" | "strategy_analysis" | "preview" | "place", "payload": {...} }
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.segment_balance import get_segment_balance
from utils.logger import log_debug, log_error, log_warning

router = APIRouter(prefix="/ws", tags=["WebSocket"])

BALANCE_PUSH_SEC = 8.0


def _ws_preview_trade(
    preview_fn: Callable[..., Any],
    normalize_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Callable[..., Any]:
    """WebSocket preview: default auto_execute=True without duplicating client kw."""

    def _call(**kw: Any) -> Any:
        if normalize_fn:
            kw = normalize_fn(kw)
        kw.setdefault("auto_execute", True)
        return preview_fn(**kw)

    return _call


def _segment_handlers(segment: str) -> Dict[str, Callable[..., Any]]:
    seg = (segment or "").strip().lower()
    if seg in ("nifty", "nifty50", "v2"):
        from services import v2_trade_service
        from services.v2_trade_service import (
            get_checklist_analyze as checklist_analyze,
            get_checklist_live as checklist_live,
            get_strategy_analysis as strategy_analysis,
        )
        from services.v2_strategy_watch import (
            arm_watch,
            disarm_watch,
            get_watch_events,
            get_watch_status,
            nuclear_reset_watch,
        )

        return {
            "checklist_live": checklist_live,
            "checklist_analyze": checklist_analyze,
            "strategy_analysis": strategy_analysis,
            "preview": _ws_preview_trade(v2_trade_service.preview_trade),
            "place": lambda **kw: v2_trade_service.place_trade(**kw),
            "watch_status": get_watch_status,
            "watch_events": get_watch_events,
            "arm": arm_watch,
            "disarm": disarm_watch,
            "nuclear_reset": nuclear_reset_watch,
            "balance": lambda: get_segment_balance(seg),
        }
    if seg in ("sensex", "bfo", "bse"):
        from services import sensex_trade_service
        from services.sensex_trade_service import (
            get_checklist_analyze as checklist_analyze,
            get_checklist_live as checklist_live,
            get_strategy_analysis as strategy_analysis,
            normalize_sensex_trade_kwargs,
        )
        from services.sensex_strategy_watch import (
            arm_watch,
            disarm_watch,
            get_watch_events,
            get_watch_status,
            nuclear_reset_watch,
        )

        return {
            "checklist_live": lambda **kw: checklist_live(**normalize_sensex_trade_kwargs(kw)),
            "checklist_analyze": lambda **kw: checklist_analyze(**normalize_sensex_trade_kwargs(kw)),
            "strategy_analysis": lambda **kw: strategy_analysis(**normalize_sensex_trade_kwargs(kw)),
            "preview": _ws_preview_trade(
                sensex_trade_service.preview_trade,
                normalize_fn=normalize_sensex_trade_kwargs,
            ),
            "place": lambda **kw: sensex_trade_service.place_trade(**normalize_sensex_trade_kwargs(kw)),
            "watch_status": get_watch_status,
            "watch_events": get_watch_events,
            "arm": arm_watch,
            "disarm": disarm_watch,
            "nuclear_reset": nuclear_reset_watch,
            "balance": lambda: get_segment_balance(seg),
        }
    if seg in ("commodity", "mcx", "crude"):
        from services import commodity_trade_service
        from services.commodity_trade_service import (
            get_checklist_analyze as checklist_analyze,
            get_checklist_live as checklist_live,
            get_strategy_analysis as strategy_analysis,
        )
        from services.commodity_strategy_watch import (
            arm_watch,
            disarm_watch,
            get_watch_events,
            get_watch_status,
            nuclear_reset_watch,
        )

        return {
            "checklist_live": checklist_live,
            "checklist_analyze": checklist_analyze,
            "strategy_analysis": strategy_analysis,
            "preview": _ws_preview_trade(commodity_trade_service.preview_trade),
            "place": lambda **kw: commodity_trade_service.place_trade(**kw),
            "watch_status": get_watch_status,
            "watch_events": get_watch_events,
            "arm": arm_watch,
            "disarm": disarm_watch,
            "nuclear_reset": nuclear_reset_watch,
            "balance": lambda: get_segment_balance(seg),
        }
    if seg in ("crypto", "binance", "btc"):
        from services import crypto_trade_service
        from services.crypto_trade_service import get_checklist_analyze as checklist_analyze
        from services.crypto_trade_service import get_checklist_live as checklist_live
        from services.crypto_trade_service import get_strategy_analysis as strategy_analysis
        from services.crypto_strategy_watch import (
            arm_watch,
            disarm_watch,
            get_watch_events,
            get_watch_status,
            nuclear_reset_watch,
        )

        return {
            "checklist_live": checklist_live,
            "checklist_analyze": checklist_analyze,
            "strategy_analysis": strategy_analysis,
            "preview": _ws_preview_trade(crypto_trade_service.preview_trade),
            "place": lambda **kw: crypto_trade_service.place_trade(**kw),
            "watch_status": get_watch_status,
            "watch_events": get_watch_events,
            "arm": arm_watch,
            "disarm": disarm_watch,
            "nuclear_reset": nuclear_reset_watch,
            "balance": lambda: get_segment_balance(seg),
        }
    raise ValueError(f"Unknown segment '{segment}'")


@router.websocket("/segment/{segment}")
async def segment_stream(websocket: WebSocket, segment: str):
    await websocket.accept()
    log_debug(f"[WS] segment connected {segment}")
    try:
        h = _segment_handlers(segment)
    except Exception as exc:
        await websocket.send_text(
            json.dumps({"type": "error", "message": str(exc), "segment": segment})
        )
        await websocket.close()
        return

    last_status_at = 0.0
    last_events_at = 0.0
    last_balance_at = 0.0

    async def _send(msg: Dict[str, Any]) -> bool:
        try:
            await websocket.send_text(json.dumps(msg))
            return True
        except (WebSocketDisconnect, RuntimeError):
            return False
        except Exception as exc:
            log_warning(f"[WS] segment send failed: {exc}")
            return False

    async def _push_balance() -> None:
        fn = h.get("balance")
        if not fn:
            return
        data = await asyncio.to_thread(fn)
        await _send({"type": "balance", "ts": datetime.utcnow().isoformat(), "data": data})

    # Initial snapshot
    try:
        await _send(
            {"type": "watch_status", "ts": datetime.utcnow().isoformat(), "data": h["watch_status"]()}
        )
        await _send(
            {
                "type": "watch_events",
                "ts": datetime.utcnow().isoformat(),
                "data": list(h["watch_events"](20) or []),
            }
        )
        await _push_balance()
        last_balance_at = asyncio.get_running_loop().time()
    except Exception:
        pass

    while True:
        try:
            # Handle inbound command (non-blocking).
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                msg = json.loads(raw)
                op = str(msg.get("op") or "").strip().lower()
                payload = msg.get("payload") or {}
                if op == "ping":
                    await _send({"type": "pong", "ts": datetime.utcnow().isoformat()})
                elif op == "arm":
                    out = h["arm"](**(payload if isinstance(payload, dict) else {}))
                    await _send({"type": "watch_status", "ts": datetime.utcnow().isoformat(), "data": out})
                elif op == "disarm":
                    out = h["disarm"]()
                    await _send({"type": "watch_status", "ts": datetime.utcnow().isoformat(), "data": out})
                elif op in ("nuclear_reset", "reset"):
                    out = h["nuclear_reset"]()
                    await _send({"type": "watch_status", "ts": datetime.utcnow().isoformat(), "data": out})
                    await _send(
                        {
                            "type": "watch_events",
                            "ts": datetime.utcnow().isoformat(),
                            "data": list(h["watch_events"](20) or []),
                        }
                    )
                elif op in ("balance", "refresh_balance"):
                    await _push_balance()
                    last_balance_at = asyncio.get_running_loop().time()
                elif op == "refresh":
                    last_status_at = 0.0
                    last_events_at = 0.0
                    last_balance_at = 0.0
                elif op in ("checklist_live", "checklist_analyze", "strategy_analysis", "preview", "place"):
                    fn = h.get(op)
                    if not fn:
                        await _send(
                            {
                                "type": "error",
                                "ts": datetime.utcnow().isoformat(),
                                "message": f"Unsupported op '{op}'",
                            }
                        )
                    else:
                        kw = payload if isinstance(payload, dict) else {}
                        out = await asyncio.to_thread(fn, **kw)
                        if not await _send({"type": op, "ts": datetime.utcnow().isoformat(), "data": out}):
                            break
                else:
                    if op:
                        await _send({"type": "error", "ts": datetime.utcnow().isoformat(), "message": f"Unknown op '{op}'"})
            except asyncio.TimeoutError:
                pass
            except Exception as exc:
                await _send({"type": "error", "ts": datetime.utcnow().isoformat(), "message": str(exc)[:200]})

            now = asyncio.get_running_loop().time()
            if now - last_status_at >= 2.0:
                last_status_at = now
                if not await _send(
                    {"type": "watch_status", "ts": datetime.utcnow().isoformat(), "data": h["watch_status"]()}
                ):
                    break

            if now - last_events_at >= 4.0:
                last_events_at = now
                if not await _send(
                    {
                        "type": "watch_events",
                        "ts": datetime.utcnow().isoformat(),
                        "data": list(h["watch_events"](20) or []),
                    }
                ):
                    break

            if now - last_balance_at >= BALANCE_PUSH_SEC:
                last_balance_at = now
                try:
                    await _push_balance()
                except Exception:
                    pass

        except WebSocketDisconnect:
            break
        except RuntimeError as exc:
            if "websocket.send" in str(exc).lower() or "websocket.close" in str(exc).lower():
                break
            log_error(f"[WS] segment loop error: {exc}")
            break
        except Exception as exc:
            log_error(f"[WS] segment loop error: {exc}")
            break

    log_debug(f"[WS] segment disconnected {segment}")

