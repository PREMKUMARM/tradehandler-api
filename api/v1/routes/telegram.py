"""
Telegram Notification API Routes
Handles Telegram notifications from frontend
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

from services.telegram_service import telegram_service, TelegramNotification
from utils.logger import log_info, log_error

router = APIRouter(prefix="/telegram", tags=["telegram"])


class FrontendNotificationRequest(BaseModel):
    """Frontend notification request"""
    notification: Dict[str, Any] = Field(..., description="Notification data")


@router.post("/notify")
async def send_telegram_notification(request: FrontendNotificationRequest):
    """Send Telegram notification from frontend"""
    
    try:
        # Convert frontend notification to TelegramNotification
        notification_data = request.notification
        
        telegram_notification = TelegramNotification(
            title=notification_data.get("title", "Unknown Event"),
            message=notification_data.get("message", ""),
            priority=notification_data.get("priority", "normal"),
            category=notification_data.get("category", "general"),
            metadata=notification_data.get("metadata"),
        )
        
        # Send notification
        success = await telegram_service.send_notification(telegram_notification)
        
        if success:
            log_info(f"Frontend Telegram notification sent: {telegram_notification.title}")
            return {
                "success": True,
                "message": "Notification sent successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            log_error("Failed to send frontend Telegram notification")
            raise HTTPException(status_code=500, detail="Failed to send notification")
            
    except Exception as e:
        log_error(f"Error processing frontend Telegram notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_telegram_status():
    """Get Telegram service status"""
    
    try:
        status = {
            "enabled": telegram_service.enabled,
            "bot_token_configured": bool(telegram_service.bot_token),
            "chat_id_configured": bool(telegram_service.chat_id),
            "service_available": telegram_service.enabled
        }
        
        return {
            "success": True,
            "data": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Error getting Telegram status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
async def test_telegram_notification():
    """Test Telegram notification"""
    
    try:
        test_notification = TelegramNotification(
            title="Frontend Test",
            message="Telegram notifications are working correctly.",
            priority="normal",
            category="system"
        )
        
        success = await telegram_service.send_notification(test_notification)
        
        if success:
            return {
                "success": True,
                "message": "Test notification sent successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to send test notification")
            
    except Exception as e:
        log_error(f"Error sending test Telegram notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))
