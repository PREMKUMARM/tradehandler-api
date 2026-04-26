from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from database.repositories import get_kite_push_reminder_repository, get_push_device_repository
from services.push.kite_reminder_config import get_merged_config, next_run_payload
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

