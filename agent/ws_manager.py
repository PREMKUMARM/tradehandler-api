from typing import List, Any
import json
from datetime import datetime
from fastapi import WebSocket
from utils.logger import log_agent_activity
from database.repositories import get_log_repository
from database.models import AgentLog
from services.push.push_service import push_service

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] New client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WS] Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected_clients = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                print(f"[WS] Error sending to client: {e}")
                disconnected_clients.append(connection)
        
        for client in disconnected_clients:
            self.disconnect(client)

manager = ConnectionManager()

async def broadcast_agent_update(update_type: str, data: Any):
    await manager.broadcast({
        "type": update_type,
        "timestamp": datetime.now().isoformat(),
        "data": data
    })

    # Mirror key UI notifications to mobile push (best-effort).
    try:
        if update_type == "NEW_APPROVAL":
            title = "New approval required"
            body = str(data.get("action") if isinstance(data, dict) else "Approval event")
            asyncio.create_task(push_service.send_to_user(user_id="default", title=title, body=body))
    except Exception:
        pass

def add_agent_log(message: str, log_type: str = "info", component: str = "agent", metadata: dict = None):
    """
    Unified logging function that saves to database, prints to console, and broadcasts to UI via WebSocket.
    Can be called from anywhere (sync or async).
    """
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%H:%M:%S")

    # Save to database
    try:
        log_repo = get_log_repository()
        log_entry = AgentLog(
            timestamp=timestamp,
            level=log_type.upper(),
            message=message,
            component=component,
            metadata=metadata
        )
        log_repo.save(log_entry)
    except Exception as e:
        print(f"Error saving log to database: {e}")

    # Print to console and log to file
    log_agent_activity(message, log_type)

    # Broadcast via WebSocket (fire and forget)
    update = {
        "timestamp": timestamp_str,
        "message": message,
        "type": log_type
    }
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(broadcast_agent_update("LIVE_LOG", update))
            # Push only higher-signal logs to mobile by default.
            if log_type.lower() in ("error", "warning"):
                loop.create_task(
                    push_service.send_to_user(
                        user_id="default",
                        title=f"{component.upper()} {log_type.upper()}",
                        body=message,
                        data={"type": "LIVE_LOG", "level": log_type.lower()},
                    )
                )
    except Exception as e:
        print(f"Error broadcasting log: {e}")

