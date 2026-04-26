import json
import os
from typing import Any, Dict, Optional

import httpx
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest


class FcmClient:
    """
    Minimal FCM HTTP v1 client.

    Required env:
    - FCM_PROJECT_ID
    - FCM_SERVICE_ACCOUNT_JSON (path to service account json file)
    """

    def __init__(self) -> None:
        self.project_id = os.getenv("FCM_PROJECT_ID", "").strip()
        self.service_account_path = os.getenv("FCM_SERVICE_ACCOUNT_JSON", "").strip()
        self._scopes = ["https://www.googleapis.com/auth/firebase.messaging"]

    def is_configured(self) -> bool:
        return bool(self.project_id and self.service_account_path and os.path.exists(self.service_account_path))

    def _get_access_token(self) -> str:
        if not self.is_configured():
            raise RuntimeError("FCM is not configured. Set FCM_PROJECT_ID and FCM_SERVICE_ACCOUNT_JSON.")
        credentials = service_account.Credentials.from_service_account_file(
            self.service_account_path,
            scopes=self._scopes,
        )
        credentials.refresh(GoogleAuthRequest())
        if not credentials.token:
            raise RuntimeError("Unable to obtain Google access token for FCM.")
        return credentials.token

    async def send(
        self,
        *,
        token: str,
        title: str,
        body: str,
        data: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        access_token = self._get_access_token()
        url = f"https://fcm.googleapis.com/v1/projects/{self.project_id}/messages:send"
        payload: Dict[str, Any] = {
            "message": {
                "token": token,
                "notification": {"title": title, "body": body},
            }
        }
        if data:
            payload["message"]["data"] = data

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                content=json.dumps(payload),
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"FCM send failed ({resp.status_code}): {resp.text}")
            return resp.json()

