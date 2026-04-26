from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from database.repositories import get_push_device_repository
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

