"""
WebSocket API endpoints for real-time data
"""
import json
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import List, Optional
from pydantic import BaseModel, Field

from utils.websocket_manager import kite_ws_manager, portfolio_ws_manager
from utils.logger import log_info, log_error

router = APIRouter(prefix="/ws", tags=["WebSocket"])


class SubscribeRequest(BaseModel):
    tokens: List[str] = Field(..., description="List of instrument tokens to subscribe to")


class UnsubscribeRequest(BaseModel):
    tokens: List[str] = Field(..., description="List of instrument tokens to unsubscribe from")


@router.websocket("/market/{client_id}")
async def websocket_market_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time market data"""
    connection_id = f"{client_id}_{uuid.uuid4().hex[:8]}"
    
    try:
        await kite_ws_manager.connect(websocket, connection_id)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "subscribe":
                    tokens = message.get("tokens", [])
                    await kite_ws_manager.subscribe(connection_id, tokens)
                
                elif message_type == "unsubscribe":
                    tokens = message.get("tokens", [])
                    await kite_ws_manager.unsubscribe(connection_id, tokens)
                
                elif message_type == "ping":
                    # Send pong response
                    await kite_ws_manager.send_personal_message({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    }, connection_id)
                
                else:
                    log_error(f"Unknown message type: {message_type}")
                    
            except json.JSONDecodeError:
                log_error(f"Invalid JSON received: {data}")
            except Exception as e:
                log_error(f"Error processing WebSocket message: {e}")
                
    except WebSocketDisconnect:
        log_info(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        log_error(f"WebSocket error: {e}")
    finally:
        kite_ws_manager.disconnect(connection_id)


@router.websocket("/portfolio/{client_id}")
async def websocket_portfolio_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time portfolio data"""
    connection_id = f"portfolio_{client_id}_{uuid.uuid4().hex[:8]}"
    
    try:
        await portfolio_ws_manager.connect(websocket, connection_id)
        
        while True:
            # Keep connection alive and handle ping/pong
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await portfolio_ws_manager.send_personal_message({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    }, connection_id)
                    
            except json.JSONDecodeError:
                log_error(f"Invalid JSON received: {data}")
            except Exception as e:
                log_error(f"Error processing portfolio WebSocket message: {e}")
                
    except WebSocketDisconnect:
        log_info(f"Portfolio WebSocket client disconnected: {connection_id}")
    except Exception as e:
        log_error(f"Portfolio WebSocket error: {e}")
    finally:
        portfolio_ws_manager.disconnect(connection_id)


@router.post("/market/subscribe")
async def subscribe_market_data(request: SubscribeRequest, client_id: str = "default"):
    """Subscribe to market data (REST API fallback)"""
    try:
        connection_id = f"rest_{client_id}"
        
        # This would typically be handled via WebSocket
        # For REST fallback, we can store subscriptions and notify when WebSocket connects
        await kite_ws_manager.subscribe(connection_id, request.tokens)
        
        return {
            "status": "success",
            "message": f"Subscribed to {len(request.tokens)} tokens",
            "tokens": request.tokens
        }
    except Exception as e:
        log_error(f"Error subscribing to market data: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/market/unsubscribe")
async def unsubscribe_market_data(request: UnsubscribeRequest, client_id: str = "default"):
    """Unsubscribe from market data (REST API fallback)"""
    try:
        connection_id = f"rest_{client_id}"
        await kite_ws_manager.unsubscribe(connection_id, request.tokens)
        
        return {
            "status": "success",
            "message": f"Unsubscribed from {len(request.tokens)} tokens",
            "tokens": request.tokens
        }
    except Exception as e:
        log_error(f"Error unsubscribing from market data: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/connections")
async def get_active_connections():
    """Get list of active WebSocket connections"""
    try:
        market_connections = len(kite_ws_manager.active_connections)
        portfolio_connections = len(portfolio_ws_manager.active_connections)
        
        return {
            "data": {
                "market_connections": market_connections,
                "portfolio_connections": portfolio_connections,
                "total_connections": market_connections + portfolio_connections,
                "kite_ws_running": kite_ws_manager.is_running,
                "portfolio_ws_running": portfolio_ws_manager.is_running
            }
        }
    except Exception as e:
        log_error(f"Error getting connections: {e}")
        return {"error": str(e)}
