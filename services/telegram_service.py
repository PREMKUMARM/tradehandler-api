"""
Telegram Bot Service for Real-time Notifications
Sends notifications about agent actions, system events, and user interactions
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
import os
from dataclasses import dataclass

from utils.logger import log_info, log_error, log_warning


@dataclass
class TelegramNotification:
    """Telegram notification data structure"""
    title: str
    message: str
    priority: str = "normal"  # low, normal, high, urgent
    category: str = "general"  # agent, system, trading, user_action
    metadata: Optional[Dict[str, Any]] = None


class TelegramService:
    """Telegram Bot Service for sending notifications"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.enabled:
            log_warning("Telegram bot not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
    
    async def send_notification(self, notification: TelegramNotification) -> bool:
        """Send a notification to Telegram"""
        if not self.enabled:
            return False
        
        try:
            cat_label = notification.category.replace("_", " ").title()
            pri_label = notification.priority.replace("_", " ").title()
            ts = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
            body = (notification.message or "").strip()
            formatted_message = (
                f"*{notification.title}*\n\n"
                f"{body}\n\n"
                f"---\n"
                f"_{cat_label} · Priority: {pri_label} · {ts} · AlgoFeast_"
            )
            
            # Send message
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendMessage"
                data = {
                    'chat_id': self.chat_id,
                    'text': formatted_message,
                    'parse_mode': 'Markdown',
                    'disable_web_page_preview': True
                }
                
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        log_info(f"Telegram notification sent: {notification.title}")
                        return True
                    else:
                        error_text = await response.text()
                        log_error(f"Failed to send Telegram notification: {error_text}")
                        return False
                        
        except Exception as e:
            log_error(f"Error sending Telegram notification: {e}")
            return False
    
    async def notify_agent_status(self, agent_id: str, agent_name: str, status: str, details: str = "") -> bool:
        """Notify about agent status changes"""
        notification = TelegramNotification(
            title=f"Agent Status Update: {agent_name}",
            message=f"Agent **{agent_name}** ({agent_id}) status changed to **{status}**" + (f"\n\n{details}" if details else ""),
            priority="normal" if status in ["running", "idle"] else "high",
            category="agent",
        )
        return await self.send_notification(notification)
    
    async def notify_chat_message(self, user_message: str, agent_response: str, agents_used: list, processing_time: float) -> bool:
        """Notify about new chat interactions"""
        notification = TelegramNotification(
            title="Multi-Agent Chat Interaction",
            message=f"**User:** {user_message[:100]}{'...' if len(user_message) > 100 else ''}\n\n"
                   f"**Agents Used:** {', '.join(agents_used)}\n"
                   f"**Processing Time:** {processing_time:.2f}s\n\n"
                   f"**Response:** {agent_response[:200]}{'...' if len(agent_response) > 200 else ''}",
            priority="normal",
            category="chat",
        )
        return await self.send_notification(notification)
    
    async def notify_system_event(self, event_type: str, message: str, priority: str = "normal") -> bool:
        """Notify about system events"""
        notification = TelegramNotification(
            title=f"System Event: {event_type}",
            message=message,
            priority=priority,
            category="system",
        )
        return await self.send_notification(notification)
    
    async def notify_error(self, error_type: str, error_message: str, context: str = "") -> bool:
        """Notify about errors"""
        notification = TelegramNotification(
            title=f"Error: {error_type}",
            message=(
                f"**Error Type:** {error_type}\n\n"
                f"**Message:** {error_message}\n\n"
                f"**Context:** {context if context else '—'}"
            ),
            priority="urgent",
            category="error",
        )
        return await self.send_notification(notification)
    
    async def notify_trading_action(self, action: str, details: Dict[str, Any]) -> bool:
        """Notify about trading actions"""
        notification = TelegramNotification(
            title=f"Trading Action: {action}",
            message=f"**Action:** {action}\n\n"
                   f"**Details:** {self._format_trading_details(details)}",
            priority="high",
            category="trading",
        )
        return await self.send_notification(notification)
    
    def _format_trading_details(self, details: Dict[str, Any]) -> str:
        """Format trading details for message"""
        formatted = []
        for key, value in details.items():
            if isinstance(value, (dict, list)):
                formatted.append(f"**{key}:** {str(value)[:100]}...")
            else:
                formatted.append(f"**{key}:** {value}")
        return "\n".join(formatted)


# Global instance
telegram_service = TelegramService()


# Decorator for automatic notifications
def telegram_notify(category: str = "general", priority: str = "normal"):
    """Decorator to automatically send Telegram notifications for function calls"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                
                # Send success notification
                notification = TelegramNotification(
                    title=f"Function Executed: {func.__name__}",
                    message=f"Function **{func.__name__}** executed successfully.",
                    priority=priority,
                    category=category
                )
                await telegram_service.send_notification(notification)
                
                return result
                
            except Exception as e:
                # Send error notification
                await telegram_service.notify_error(
                    error_type=func.__name__,
                    error_message=str(e),
                    context=f"Arguments: {args}, Kwargs: {kwargs}"
                )
                raise
                
        return wrapper
    return decorator
