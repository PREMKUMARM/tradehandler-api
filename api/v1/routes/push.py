from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from database.repositories import (
    get_kite_push_reminder_repository,
    get_push_device_repository,
    get_strategy_alert_repository,
)
from services.push.kite_reminder_config import get_merged_config, next_run_payload
from services.push.market_open_gap_alert import (
    compute_gap_alert,
    next_run_info as market_open_gap_next_run_info,
    send_market_open_gap_alert,
)
from services.push.nifty_ema_pullback_signal import (
    force_test_send as ema_force_test_send,
    get_state_snapshot as ema_get_state_snapshot,
    preview_signal as ema_preview_signal,
    reset_day_state as ema_reset_day_state,
)
from services.push.nifty_orb_signal import (
    force_test_send as orb_force_test_send,
    get_state_snapshot as orb_get_state_snapshot,
    preview_signal as orb_preview_signal,
    reset_day_state as orb_reset_day_state,
)
from services.push.nifty_pdh_pdl_signal import (
    force_test_send as pdh_pdl_force_test_send,
    get_state_snapshot as pdh_pdl_get_state_snapshot,
    preview_signal as pdh_pdl_preview_signal,
    reset_day_state as pdh_pdl_reset_day_state,
)
from services.push.option_contract_resolver import cache_state as option_resolver_state
from services.push.push_service import push_service


router = APIRouter(prefix="/push", tags=["Push"])


class PushRegisterRequest(BaseModel):
    user_id: str = Field(default="default")
    token: str
    platform: str = Field(default="android")
    device_id: Optional[str] = None


class PushUnregisterRequest(BaseModel):
    token: str


class PushTestRequest(BaseModel):
    user_id: str = Field(default="default")
    title: str = Field(default="Test notification")
    body: str = Field(default="If you see this, push is wired end-to-end.")


@router.post("/register")
async def register_device(req: PushRegisterRequest):
    repo = get_push_device_repository()
    ok = repo.upsert(
        user_id=req.user_id,
        token=req.token,
        platform=req.platform,
        device_id=req.device_id,
    )
    return {"ok": ok}


@router.post("/unregister")
async def unregister_device(req: PushUnregisterRequest):
    repo = get_push_device_repository()
    ok = repo.delete_by_token(req.token)
    return {"ok": ok}


@router.get("/devices")
async def list_devices(user_id: str = "default"):
    repo = get_push_device_repository()
    return {"data": repo.list_tokens(user_id=user_id)}


@router.post("/test")
async def send_test(req: PushTestRequest):
    result = await push_service.send_to_user(user_id=req.user_id, title=req.title, body=req.body)
    return {"data": result}


# --- Kite (Zerodha) login weekday reminder (FCM) ---


class KiteReminderGetResponse(BaseModel):
    enabled: bool
    tz: str
    hour: int
    minute: int
    title: str
    body: str
    from_database: bool
    fcm_configured: bool
    next_run: Dict[str, Any]


class KiteReminderUpdateRequest(BaseModel):
    enabled: bool = True
    tz: str = Field(default="Asia/Kolkata", min_length=1, max_length=64)
    hour: int = Field(default=8, ge=0, le=23)
    minute: int = Field(default=0, ge=0, le=59)
    title: str = Field(default="Zerodha (Kite) login", min_length=1, max_length=200)
    body: str = Field(
        default="Log in to Kite Connect in AlgoFeast so trading stays connected today.",
        min_length=1,
        max_length=500,
    )


@router.get("/kite-reminder")
async def get_kite_reminder() -> dict:
    cfg = get_merged_config()
    nxt: Dict[str, Any] = dict(next_run_payload(cfg))
    return {
        "data": KiteReminderGetResponse(
            enabled=cfg.enabled,
            tz=cfg.tz,
            hour=cfg.hour,
            minute=cfg.minute,
            title=cfg.title,
            body=cfg.body,
            from_database=cfg.from_database,
            fcm_configured=push_service.configured(),
            next_run=nxt,
        ).model_dump()
    }


@router.put("/kite-reminder")
async def put_kite_reminder(req: KiteReminderUpdateRequest) -> dict:
    repo = get_kite_push_reminder_repository()
    ok = repo.save(
        enabled=req.enabled,
        tz=req.tz.strip(),
        hour=req.hour,
        minute=req.minute,
        title=req.title.strip(),
        body=req.body.strip(),
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save Kite reminder settings")
    return await get_kite_reminder()


@router.post("/kite-reminder/test")
async def test_kite_reminder() -> dict:
    if not push_service.configured():
        raise HTTPException(status_code=503, detail="FCM is not configured on this server")
    cfg = get_merged_config()
    data = {
        "type": "kite_login_reminder",
        "source": "algofeast_backend",
        "test": "true",
    }
    repo = get_push_device_repository()
    user_ids = repo.list_distinct_user_ids()
    if not user_ids:
        return {
            "data": {
                "sent": 0,
                "failed": 0,
                "user_ids": [],
                "message": "No devices registered. Open the app and register FCM first.",
            }
        }
    results: Dict[str, Any] = {"sent": 0, "failed": 0, "per_user": {}, "user_ids": user_ids}
    for uid in user_ids:
        r = await push_service.send_to_user(
            user_id=uid,
            title=cfg.title,
            body=cfg.body,
            data={k: str(v) for k, v in data.items()},
        )
        results["per_user"][uid] = r
        results["sent"] += r.get("sent", 0)
        results["failed"] += r.get("failed", 0)
    return {"data": results}


# --- Market-open NIFTY gap alert (FCM) ---


@router.get("/market-open-gap")
async def get_market_open_gap_info() -> dict:
    """
    Schedule + readiness info for the market-open gap push alert.
    Useful for surfacing 'next run' in the UI.
    """
    return {"data": market_open_gap_next_run_info()}


@router.get("/market-open-gap/preview")
async def preview_market_open_gap() -> dict:
    """
    Compute (but do NOT send) the current gap alert payload — handy for verifying
    what a fresh push would look like at any time, including outside market hours
    (in which case 'open' may be yesterday's open).
    """
    if not push_service.configured():
        raise HTTPException(status_code=503, detail="FCM is not configured on this server")
    result = await compute_gap_alert(retries=1)
    if result is None:
        return {
            "data": {
                "available": False,
                "reason": "Quote/open price not available yet (likely holiday or pre-open).",
            }
        }
    return {
        "data": {
            "available": True,
            "title": result.title,
            "body": result.body,
            "payload": result.data,
        }
    }


@router.post("/market-open-gap/test")
async def test_market_open_gap() -> dict:
    """
    Trigger a market-open NIFTY gap push immediately to every registered user.
    """
    if not push_service.configured():
        raise HTTPException(status_code=503, detail="FCM is not configured on this server")
    result = await send_market_open_gap_alert(retries=1, is_test=True)
    return {"data": result}


# --- NIFTY 15m Opening-Range-Breakout signal (push-only) ---


class OrbTestRequest(BaseModel):
    direction: str = Field(default="LONG", description="LONG or SHORT")


@router.get("/orb-signal")
async def get_orb_signal_state() -> dict:
    """Today's ORB state + config + readiness info."""
    return {"data": orb_get_state_snapshot()}


@router.get("/orb-signal/preview")
async def preview_orb_signal(direction: str = "LONG") -> dict:
    """
    Compose (but do NOT send) the would-be ORB push using current spot. If OR
    hasn't been built yet, the preview synthesizes a plausible OR around spot
    just so the UI can render the formatting.
    """
    return {"data": orb_preview_signal(direction_override=direction)}


@router.post("/orb-signal/test")
async def test_orb_signal(req: OrbTestRequest) -> dict:
    """
    Compose AND send a synthetic ORB push immediately to every registered user.
    Useful to verify the end-to-end push pipeline.
    """
    if not push_service.configured():
        raise HTTPException(status_code=503, detail="FCM is not configured on this server")
    result = await orb_force_test_send(direction=req.direction)
    return {"data": result}


@router.post("/orb-signal/reset")
async def reset_orb_signal_day() -> dict:
    """
    Manually reset today's ORB state (clears OR, signal-fired flag, etc.). Handy
    while testing — does NOT cancel any past push.
    """
    return {"data": orb_reset_day_state()}


# --- NIFTY 9-EMA Pullback signal (push-only) ---


class StrategyTestRequest(BaseModel):
    direction: str = Field(default="LONG", description="LONG or SHORT")


@router.get("/ema-signal")
async def get_ema_signal_state() -> dict:
    return {"data": ema_get_state_snapshot()}


@router.get("/ema-signal/preview")
async def preview_ema_signal(direction: str = "LONG") -> dict:
    return {"data": ema_preview_signal(direction_override=direction)}


@router.post("/ema-signal/test")
async def test_ema_signal(req: StrategyTestRequest) -> dict:
    if not push_service.configured():
        raise HTTPException(status_code=503, detail="FCM is not configured on this server")
    return {"data": await ema_force_test_send(direction=req.direction)}


@router.post("/ema-signal/reset")
async def reset_ema_signal_day() -> dict:
    return {"data": ema_reset_day_state()}


# --- NIFTY PDH/PDL Breakout signal (push-only) ---


@router.get("/pdh-pdl-signal")
async def get_pdh_pdl_signal_state() -> dict:
    return {"data": pdh_pdl_get_state_snapshot()}


@router.get("/pdh-pdl-signal/preview")
async def preview_pdh_pdl_signal(direction: str = "LONG") -> dict:
    return {"data": pdh_pdl_preview_signal(direction_override=direction)}


@router.post("/pdh-pdl-signal/test")
async def test_pdh_pdl_signal(req: StrategyTestRequest) -> dict:
    if not push_service.configured():
        raise HTTPException(status_code=503, detail="FCM is not configured on this server")
    return {"data": await pdh_pdl_force_test_send(direction=req.direction)}


@router.post("/pdh-pdl-signal/reset")
async def reset_pdh_pdl_signal_day() -> dict:
    return {"data": pdh_pdl_reset_day_state()}


# --- Option-contract resolver introspection ---


@router.get("/option-resolver")
async def get_option_resolver_state() -> dict:
    """
    Returns the current state of the NIFTY ATM CE/PE option-contract resolver
    (expiry being targeted, number of strikes loaded, etc.). Useful for
    debugging stale instruments cache.
    """
    return {"data": option_resolver_state()}


# --- Strategy alert audit / tracking ---


@router.get("/alerts")
async def list_strategy_alerts(
    strategy: Optional[str] = None,
    direction: Optional[str] = None,
    since: Optional[str] = None,
    include_tests: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    Return the most recent push-strategy alerts persisted in the DB.

    Query params:
        strategy: filter by strategy id (`nifty_15m_orb`, `nifty_9ema_pullback`,
                  `nifty_pdh_pdl_break`, `market_open_gap`, ...).
        direction: 'LONG' / 'SHORT' / 'UP' / 'DOWN' / 'FLAT'.
        since: ISO timestamp; only alerts at/after this time.
        include_tests: when False, exclude alerts triggered by the UI test buttons.
        limit, offset: pagination (default 50 / 0).
    """
    repo = get_strategy_alert_repository()
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    return {
        "data": {
            "items": repo.list(
                strategy=strategy,
                direction=direction,
                since=since,
                include_tests=include_tests,
                limit=limit,
                offset=offset,
            ),
            "limit": limit,
            "offset": offset,
        }
    }


@router.get("/alerts/stats")
async def strategy_alert_stats(since: Optional[str] = None) -> dict:
    """Lightweight totals for the UI badge (since timestamp optional)."""
    repo = get_strategy_alert_repository()
    return {"data": repo.stats(since=since)}


@router.get("/alerts/{alert_id}")
async def get_strategy_alert(alert_id: int) -> dict:
    repo = get_strategy_alert_repository()
    row = repo.get(alert_id)
    if not row:
        raise HTTPException(status_code=404, detail="alert not found")
    return {"data": row}


@router.delete("/alerts/{alert_id}")
async def delete_strategy_alert(alert_id: int) -> dict:
    repo = get_strategy_alert_repository()
    ok = repo.delete(alert_id)
    return {"data": {"ok": ok, "id": alert_id}}

