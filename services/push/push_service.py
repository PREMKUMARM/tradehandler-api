import asyncio
from typing import Any, Dict, List, Optional

from utils.logger import log_warning, log_error, log_info
from database.repositories import get_push_device_repository
from services.push.fcm_client import FcmClient


class PushService:
    def __init__(self) -> None:
        self._fcm = FcmClient()

    def configured(self) -> bool:
        return self._fcm.is_configured()

    async def send_to_user(
        self,
        *,
        user_id: str,
        title: str,
        body: str,
        data: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        repo = get_push_device_repository()
        tokens = repo.list_token_strings(user_id=user_id)
        return await self.send_to_tokens(tokens=tokens, title=title, body=body, data=data)

    async def send_to_tokens(
        self,
        *,
        tokens: List[str],
        title: str,
        body: str,
        data: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if not tokens:
            return {"sent": 0, "failed": 0, "errors": []}

        if not self._fcm.is_configured():
            log_warning("PushService: FCM not configured; skipping push send.")
            return {"sent": 0, "failed": len(tokens), "errors": ["FCM not configured"]}

        results = {"sent": 0, "failed": 0, "errors": []}

        async def _send_one(t: str) -> None:
            nonlocal results
            try:
                await self._fcm.send(token=t, title=title, body=body, data=data)
                results["sent"] += 1
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(str(e))

        await asyncio.gather(*[_send_one(t) for t in tokens])
        return results


push_service = PushService()

