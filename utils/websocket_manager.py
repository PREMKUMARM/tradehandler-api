"""
WebSocket manager for real-time data streaming
"""
import asyncio
import json
import logging
from typing import Dict, List, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from utils.kite_utils import get_kite_instance, api_key, get_access_token
from kiteconnect.exceptions import KiteException
from utils.logger import log_info, log_error, log_warning
import websockets
from threading import Thread


class KiteWebSocketManager:
    """Manages WebSocket connections for real-time data"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # connection_id -> set of tokens
        self.kite_ws = None
        self.ws_thread = None
        self.is_running = False
        
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.subscriptions[connection_id] = set()
        log_info(f"WebSocket connected: {connection_id}")
        
        # Start Kite WebSocket if not already running
        if not self.is_running:
            self.start_kite_websocket()
    
    def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.subscriptions:
            del self.subscriptions[connection_id]
        log_info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: dict, connection_id: str):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(json.dumps(message))
            except Exception as e:
                log_error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected_clients = []
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                log_error(f"Error broadcasting to {connection_id}: {e}")
                disconnected_clients.append(connection_id)
        
        # Clean up disconnected clients
        for connection_id in disconnected_clients:
            self.disconnect(connection_id)
    
    async def subscribe(self, connection_id: str, tokens: List[str]):
        """Subscribe to tokens for a connection"""
        if connection_id in self.subscriptions:
            self.subscriptions[connection_id].update(tokens)
            log_info(f"Subscribed {connection_id} to tokens: {tokens}")
            
            # Send confirmation
            await self.send_personal_message({
                "type": "subscription_confirm",
                "tokens": list(tokens),
                "timestamp": datetime.now().isoformat()
            }, connection_id)
    
    async def unsubscribe(self, connection_id: str, tokens: List[str]):
        """Unsubscribe from tokens for a connection"""
        if connection_id in self.subscriptions:
            self.subscriptions[connection_id].difference_update(tokens)
            log_info(f"Unsubscribed {connection_id} from tokens: {tokens}")
    
    def start_kite_websocket(self):
        """Start Kite WebSocket in a separate thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.ws_thread = Thread(target=self._run_kite_websocket, daemon=True)
        self.ws_thread.start()
        log_info("Kite WebSocket thread started")
    
    def _run_kite_websocket(self):
        """Run Kite WebSocket connection"""
        try:
            # Import KiteConnect WebSocket
            from kiteconnect import KiteTicker
            
            kite = get_kite_instance()
            access_token = get_access_token()
            
            if not access_token:
                log_error("Access token not found for WebSocket connection")
                return
            
            # Initialize Kite WebSocket
            self.kite_ws = KiteTicker(api_key, access_token)
            
            # Set up event handlers
            self.kite_ws.on_ticks = self._on_ticks
            self.kite_ws.on_connect = self._on_connect
            self.kite_ws.on_close = self._on_close
            self.kite_ws.on_error = self._on_error
            
            # Connect to WebSocket
            self.kite_ws.connect()
            
        except Exception as e:
            log_error(f"Error starting Kite WebSocket: {e}")
            self.is_running = False
    
    def _on_connect(self, ws, response):
        """Called when WebSocket connects"""
        log_info("Kite WebSocket connected")
        
        # Subscribe to all tokens from all connections
        all_tokens = set()
        for tokens in self.subscriptions.values():
            all_tokens.update(tokens)
        
        if all_tokens:
            ws.subscribe(list(all_tokens))
            ws.set_mode(ws.MODE_FULL, list(all_tokens))
            log_info(f"Subscribed to {len(all_tokens)} tokens")
    
    def _on_close(self, ws, code, reason):
        """Called when WebSocket closes"""
        log_warning(f"Kite WebSocket closed: {code} - {reason}")
        self.is_running = False
    
    def _on_error(self, ws, code, reason):
        """Called when WebSocket encounters error"""
        log_error(f"Kite WebSocket error: {code} - {reason}")
    
    def _on_ticks(self, ws, ticks):
        """Called when ticks are received"""
        try:
            # Process ticks and broadcast to relevant clients
            for tick in ticks:
                token = str(tick.get('instrument_token', ''))
                
                # Find connections subscribed to this token
                for connection_id, subscribed_tokens in self.subscriptions.items():
                    if token in subscribed_tokens:
                        # Send tick to this connection
                        message = {
                            "type": "tick",
                            "data": tick,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Use asyncio to send message
                        try:
                            asyncio.create_task(
                                self.send_personal_message(message, connection_id)
                            )
                        except Exception as e:
                            log_error(f"Error sending tick to {connection_id}: {e}")
        
        except Exception as e:
            log_error(f"Error processing ticks: {e}")
    
    def stop_kite_websocket(self):
        """Stop Kite WebSocket connection"""
        if self.kite_ws:
            try:
                self.kite_ws.close()
            except Exception as e:
                log_error(f"Error closing Kite WebSocket: {e}")
        
        self.is_running = False
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
        
        log_info("Kite WebSocket stopped")


class PortfolioWebSocketManager:
    """Manages WebSocket for portfolio data (positions, orders, etc.)"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.update_interval = 2  # seconds
        self.is_running = False
        self.update_task = None
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        log_info(f"Portfolio WebSocket connected: {connection_id}")
        
        # Start updates if not already running
        if not self.is_running:
            self.is_running = True
            self.update_task = asyncio.create_task(self._update_loop())
        
        # Send initial data
        await self._send_portfolio_update(connection_id)
    
    def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        log_info(f"Portfolio WebSocket disconnected: {connection_id}")
        
        # Stop updates if no connections
        if not self.active_connections:
            self.is_running = False
            if self.update_task:
                self.update_task.cancel()
    
    async def _update_loop(self):
        """Main update loop"""
        while self.is_running and self.active_connections:
            try:
                # Send updates to all connected clients
                for connection_id in list(self.active_connections.keys()):
                    await self._send_portfolio_update(connection_id)
                
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(f"Error in portfolio update loop: {e}")
                await asyncio.sleep(5)
    
    async def _send_portfolio_update(self, connection_id: str):
        """Send portfolio update to specific connection"""
        try:
            from utils.kite_utils import get_kite_instance
            
            kite = get_kite_instance()
            
            # Get portfolio data
            margins = kite.margins()
            positions = kite.positions()
            orders = kite.orders()
            
            # Calculate totals
            net_positions = positions.get("net", [])
            total_pnl = sum(pos.get("pnl", 0) for pos in net_positions)
            active_positions = len([pos for pos in net_positions if pos.get("quantity", 0) != 0])
            
            # Prepare update message
            update = {
                "type": "portfolio_update",
                "data": {
                    "margins": margins,
                    "positions": positions,
                    "orders": orders,
                    "totals": {
                        "total_pnl": round(total_pnl, 2),
                        "active_positions": active_positions
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await self.send_personal_message(update, connection_id)
            
        except Exception as e:
            log_error(f"Error sending portfolio update: {e}")
    
    async def send_personal_message(self, message: dict, connection_id: str):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(json.dumps(message))
            except Exception as e:
                log_error(f"Error sending portfolio message to {connection_id}: {e}")
                self.disconnect(connection_id)


# Global instances
kite_ws_manager = KiteWebSocketManager()
portfolio_ws_manager = PortfolioWebSocketManager()
