from typing import Union
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from core.exceptions import (
    AlgoFeastException, ValidationError, AuthenticationError, 
    NotFoundError, BusinessLogicError, ExternalAPIError
)
from utils.logger import log_info, log_error, log_warning, log_debug
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio

import json
from pprint import pprint
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

# Kite utilities
from utils.kite_utils import (
    api_key,
    get_access_token,
    get_kite_instance,
    calculate_trend_and_suggestions
)

# Agent imports
from agent.graph import run_agent, get_agent_instance, get_agent_memory
from agent.approval import get_approval_queue
from agent.safety import get_safety_manager
from agent.config import get_agent_config
from agent.autonomous import start_autonomous_agent
from agent.ws_manager import manager, broadcast_agent_update, add_agent_log
from agent.tools.kite_tools import place_order_tool, cancel_order_tool, place_gtt_tool
from database.connection import init_database
from database.repositories import (
    get_log_repository, get_tool_repository, get_simulation_repository,
    get_config_repository, get_chat_repository
)
from database.models import ChatMessage
import uuid

# Initialize FastAPI app with enterprise-level configuration
from core.config import get_settings
from core.responses import APIResponse

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AlgoFeast - Enterprise AI Trading Agent API with Zerodha Kite Connect integration",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add enterprise middleware (order matters!)
from middleware import RequestIDMiddleware, LoggingMiddleware, ErrorHandlerMiddleware
from middleware.rate_limit import RateLimitMiddleware

app.add_middleware(RequestIDMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)  # Rate limiting before error handling
app.add_middleware(ErrorHandlerMiddleware)

# Include API v1 routes
from api.v1 import api_router
app.include_router(api_router)

# Import utilities and modules
from utils.candle_utils import aggregate_to_tf, analyze_trend
from utils.indicators import (
    calculate_bollinger_bands,
    calculate_bollinger_bands_full,
    calculate_rsi,
    calculate_pivot_points,
    calculate_support_resistance
)
from simulation import (
    simulation_state,
    live_logs,
    get_instrument_history,
    add_sim_order,
    calculate_sim_qty,
    find_option
)
from tasks import live_market_scanner, monitor_order_execution


async def refresh_instruments_cache_daily():
    """Background task to refresh instruments cache daily at 8:30 AM"""
    from utils.instruments_cache import refresh_cache
    from datetime import datetime, time
    
    while True:
        try:
            now = datetime.now()
            # Calculate next 8:30 AM
            target_time = time(8, 30)
            next_refresh = datetime.combine(now.date(), target_time)
            
            # If 8:30 AM has passed today, schedule for tomorrow
            if now.time() > target_time:
                next_refresh += timedelta(days=1)
            
            # Wait until 8:30 AM
            wait_seconds = (next_refresh - now).total_seconds()
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            
            # Refresh cache
            add_agent_log("Refreshing instruments cache (scheduled daily refresh)", "info", "system")
            result = refresh_cache()
            if result.get("success"):
                add_agent_log(f"Instruments cache refreshed successfully. Total instruments: {result.get('total_instruments', 'unknown')}", "info", "system")
            else:
                add_agent_log("Failed to refresh instruments cache", "warning", "system")
            
            # Wait 24 hours before next refresh
            await asyncio.sleep(24 * 60 * 60)
        except Exception as e:
            add_agent_log(f"Error in instruments cache refresh task: {str(e)}", "error", "system")
            # Retry after 1 hour on error
            await asyncio.sleep(60 * 60)
from strategies.runner import run_strategy_on_candles

# Legacy endpoints (maintained for backward compatibility)
# These will be gradually migrated to v1 routes

@app.websocket("/ws/agent")
async def agent_websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages if needed
            data = await websocket.receive_text()
            # For now, we don't need to handle client -> server messages
            # but we need to receive them to keep the connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log_error(f"[WS] Error: {e}")
        manager.disconnect(websocket)

# Helper to send agent updates - Redundant definition removed

# Kite Connect credentials - All loaded from environment variables
# IMPORTANT: The redirect_uri must EXACTLY match what's configured in your Kite Connect app settings
# Global API key handled by utils.kite_utils
# All secrets should be in .env file, never hardcoded
api_secret = os.getenv('KITE_API_SECRET')
redirect_uri = os.getenv('KITE_REDIRECT_URI', 'http://localhost:4200/auth-token')

# Note: api_secret validation is done in utils.kite_utils when actually needed
# This allows the server to start even if Kite credentials aren't configured yet

# CORS configuration from settings (already loaded above)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions moved to respective modules:
# - KiteConnect instance: utils.kite_utils.get_kite_instance()
# - Portfolio endpoints: api/v1/routes/portfolio.py
# - Market scanner: tasks/market_scanner.py

@app.on_event("startup")
async def startup_event():
    # Initialize database
    init_database()
    add_agent_log("Database initialized successfully", "info", "system")

    # Initialize instruments cache
    try:
        from utils.instruments_cache import ensure_cache_valid
        ensure_cache_valid()
        add_agent_log("Instruments cache initialized successfully", "info", "system")
    except Exception as e:
        add_agent_log(f"Instruments cache initialization failed: {str(e)}", "warning", "system")

    # Set approval queue callback for WebSocket broadcasting
    approval_queue = get_approval_queue()
    def broadcast_new_approval(approval):
        asyncio.create_task(broadcast_agent_update("NEW_APPROVAL", approval))

    approval_queue.on_create_callback = broadcast_new_approval

    # Start the background scanner
    asyncio.create_task(live_market_scanner())
    
    # Start Binance VWAP data updates
    from api.v1.routes.market import update_binance_vwap_data
    asyncio.create_task(update_binance_vwap_data())
    # Start the new AI Agent autonomous scanner
    start_autonomous_agent()
    # Start order monitoring task for auto-cancellation
    asyncio.create_task(monitor_order_execution())
    
    # Start instruments cache refresh task (daily at 8:30 AM)
    asyncio.create_task(refresh_instruments_cache_daily())
    
    # Initialize Binance commentary service with historical data
    from utils.binance_commentary_service import initialize_commentary_service
    asyncio.create_task(initialize_commentary_service())
    
    # Start Kite ticker WebSocket listener (market hours aware)
    from utils.kite_websocket_ticker import manage_kite_ticker_market_hours
    asyncio.create_task(manage_kite_ticker_market_hours())

# Simulation state and helpers moved to simulation/ module

# Legacy endpoint removed - use /api/v1/simulation/live-logs instead
def get_access_token_endpoint():
    """Get stored access token and validate it"""
    token = get_access_token()
    if token:
        # Validate token
        token_info = {
            "length": len(token),
            "preview": token[:20] + "..." if len(token) > 20 else token,
            "is_valid_length": len(token) >= 20,
            "status": "unknown"
        }
        
        # Try to validate token if it's long enough
        if len(token) >= 20:
            try:
                from utils.kite_utils import get_kite_api_key
                api_key = get_kite_api_key()
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(token)
                kite.profile()  # Quick validation
                token_info["status"] = "valid"
                token_info["api_key_used"] = api_key[:10] + "..." if api_key and len(api_key) > 10 else "NOT SET"
                return {
                    "access_token": token[:20] + "...",
                    "token_info": token_info,
                    "is_valid": True
                }
            except Exception as e:
                token_info["status"] = "invalid"
                token_info["error"] = str(e)
                token_info["api_key_used"] = api_key[:10] + "..." if api_key and len(api_key) > 10 else "NOT SET"
                return {
                    "access_token": token[:20] + "...",
                    "token_info": token_info,
                    "is_valid": False,
                    "message": f"Token exists but is invalid: {str(e)}"
                }
        else:
            token_info["status"] = "too_short"
            token_info["error"] = "Token is too short to be valid (expected at least 20 chars, got " + str(len(token)) + ")"
            return {
                "access_token": token[:20] + "...",
                "token_info": token_info,
                "is_valid": False,
                "message": "Token is too short. Please regenerate using /auth and /set-token"
            }
    
    return {
        "access_token": None, 
        "message": "No access token found. Please generate one using /auth and /set-token",
        "token_info": None,
        "is_valid": False
    }

# Auth endpoints moved to api/v1/routes/auth.py
def delete_access_token():
    """Delete the stored access token (useful for clearing invalid tokens)"""
    try:
        token_path = Path("config/access_token.txt")
        if token_path.exists():
            token_path.unlink()
            return {"status": "success", "message": "Access token deleted successfully"}
        return {"status": "success", "message": "No access token file found"}
    except Exception as e:
        log_error(f"Error deleting token: {str(e)}")
        raise AlgoFeastException(
            message=f"Error deleting token: {str(e)}",
            status_code=500,
            error_code="INTERNAL_ERROR"
        )

def get_login_url():
    """Get Kite Connect login URL"""
    try:
        # Validate API key is set
        # Get API key from config (may have been updated via UI)
        from agent.config import get_agent_config
        config = get_agent_config()
        current_api_key = config.kite_api_key or api_key
        if current_api_key == 'your_api_key_here' or not current_api_key:
            raise BusinessLogicError(
                message="KITE_API_KEY is not configured. Please set it in the Configuration page or environment variables",
                error_code="CONFIGURATION_ERROR"
            )
        
        kite = KiteConnect(api_key=current_api_key)
        login_url = kite.login_url()
        log_info(f"Generated login URL with redirect_uri: {redirect_uri}")
        return {
            "login_url": login_url,
            "message": "Redirect user to this URL for authentication",
            "redirect_uri": redirect_uri,
            "note": f"Make sure the redirect URI in your Kite Connect app settings matches: {redirect_uri}"
        }
    except AlgoFeastException:
        raise
    except Exception as e:
        log_error(f"Error generating login URL: {str(e)}")
        raise AlgoFeastException(
            message=f"Error generating login URL: {str(e)}",
            status_code=500,
            error_code="INTERNAL_ERROR"
        )

async def set_token(req: Request):
    """Store access token after authentication"""
    try:
        data = await req.json()
        request_token = data.get('request_token')
        access_token_from_request = data.get('access-token')
        
        # Get user_id from request to use user-specific config
        user_id = "default"
        try:
            from core.user_context import get_user_id_from_request
            user_id = get_user_id_from_request(req)
        except:
            pass
        
        log_debug(f"Received request_token: {request_token[:20] if request_token else None}...")
        log_debug(f"Received access-token from request: {access_token_from_request[:20] if access_token_from_request else None}...")
        log_debug(f"User ID: {user_id}")
        log_debug(f"Redirect URI configured: {redirect_uri}")
        
        if not request_token and not access_token_from_request:
            raise ValidationError(
                message="Either request_token or access-token required",
                field="request_token"
            )
        
        # Initialize access_token variable - will be set from generate_session() if request_token provided
        access_token = None
        
        # If request_token is provided, generate access_token (ignore any access-token from request)
        if request_token:
            # Validate API key and secret are set
            # Get credentials from user-specific config (may have been updated via UI)
            from agent.user_config import get_user_config
            config = get_user_config(user_id=user_id)
            current_api_key = config.kite_api_key or api_key
            current_api_secret = config.kite_api_secret or api_secret
            
            log_debug(f"API Key configured: {current_api_key[:10] if current_api_key and len(current_api_key) > 10 else 'NOT SET'}...")
            
            if current_api_key == 'your_api_key_here' or not current_api_key:
                raise BusinessLogicError(
                    message="KITE_API_KEY is not configured. Please set it in the Configuration page or environment variables",
                    error_code="CONFIGURATION_ERROR"
                )
            if current_api_secret == 'your_api_secret_here' or not current_api_secret:
                raise BusinessLogicError(
                    message="KITE_API_SECRET is not configured. Please set it in the Configuration page or environment variables",
                    error_code="CONFIGURATION_ERROR"
                )
            
            kite = KiteConnect(api_key=current_api_key)
            try:
                log_debug(f"Attempting to generate session with request_token...")
                log_debug(f"Request token length: {len(request_token) if request_token else 0}")
                log_debug(f"Request token preview: {request_token[:30] if request_token and len(request_token) > 30 else request_token}...")
                
                data_response = kite.generate_session(request_token, api_secret=current_api_secret)
                log_debug(f"generate_session response type: {type(data_response)}")
                log_debug(f"generate_session response: {data_response}")
                
                # Handle both dict and object responses
                if isinstance(data_response, dict):
                    access_token = data_response.get('access_token')
                    log_debug(f"Response is dict, keys: {list(data_response.keys())}")
                else:
                    # If it's an object, try to get the attribute
                    access_token = getattr(data_response, 'access_token', None) if hasattr(data_response, 'access_token') else None
                    log_debug(f"Response is object, has access_token attr: {hasattr(data_response, 'access_token')}")
                
                log_debug(f"Access token extracted: {access_token is not None}")
                log_debug(f"Access token length: {len(access_token) if access_token else 0}")
                if access_token:
                    log_debug(f"Access token preview: {access_token[:30]}...")
                    # Safety check: access_token should NOT be the same as request_token
                    if access_token == request_token:
                        log_error(f"ERROR: access_token equals request_token! This indicates a problem.")
                        raise ValidationError(
                            message="Token exchange failed: Received request_token instead of access_token. "
                                   "This usually means: 1) API key/secret mismatch, "
                                   "2) Redirect URI mismatch, or 3) Request token expired. "
                                   f"Please check your Kite Connect app settings. Redirect URI should be: {redirect_uri}",
                            field="request_token"
                        )
                
                if not access_token:
                    log_warning(f"WARNING: No access_token in response! Full response: {data_response}")
                    log_warning(f"Response type: {type(data_response)}")
                    if isinstance(data_response, dict):
                        log_warning(f"Available keys: {list(data_response.keys())}")
                        log_warning(f"Full response content: {data_response}")
                    raise ValidationError(
                        message="Failed to get access_token from Kite. The generate_session() call did not return an access_token. "
                               f"Response: {str(data_response)}. "
                               "Please check: 1) API key and secret are correct, "
                               f"2) Redirect URI matches exactly: {redirect_uri}, "
                               "3) Request token is fresh (they expire quickly).",
                        field="request_token"
                    )
            except KiteException as e:
                error_msg = str(e)
                log_error(f"KiteException: {error_msg}")
                # Provide more helpful error messages
                if "invalid" in error_msg.lower() or "expired" in error_msg.lower():
                    raise ValidationError(
                        message=f"Invalid request token: {error_msg}. "
                               f"Please ensure: 1) Redirect URI in Kite Connect app settings matches exactly '{redirect_uri}', "
                               f"2) Request token is used immediately (they expire quickly), "
                               f"3) API key and secret are correct.",
                        field="request_token"
                    )
                raise ExternalAPIError(
                    message=error_msg,
                    service="Kite Connect"
                )
            except Exception as e:
                # Catch any other unexpected errors
                error_msg = str(e)
                log_error(f"Unexpected error during token exchange: {error_msg}")
                log_error(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                raise AlgoFeastException(
                    message=f"Unexpected error during token exchange: {error_msg}. "
                           "Please check server logs for details.",
                    status_code=500,
                    error_code="INTERNAL_ERROR"
                )
        else:
            # No request_token provided, use access-token from request body
            access_token = access_token_from_request
        
        if not access_token:
            raise ValidationError(
                message="Failed to obtain access token",
                field="access_token"
            )
        
        # Safety check: Make sure we're not accidentally saving the request_token
        if request_token and access_token == request_token:
            log_error(f"ERROR: access_token is the same as request_token! This should not happen.")
            log_error(f"Request token: {request_token}")
            log_error(f"Access token: {access_token}")
            raise ValidationError(
                message="Internal error: Access token matches request token. "
                       "The token exchange may have failed. Please try again with a fresh request_token.",
                field="access_token"
            )
        
        # Validate token before saving (Kite access tokens are typically 32+ characters)
        # Note: Kite access tokens can be 32 characters, so we use a lower threshold
        if len(access_token) < 20:
            log_warning(f"WARNING: Access token seems invalid (length: {len(access_token)})")
            log_warning(f"Token value: {access_token}")
            log_warning(f"Request token was: {request_token[:20] if request_token else None}...")
            log_warning(f"Are they the same? {access_token == request_token if request_token else 'N/A'}")
            raise ValidationError(
                message=f"Invalid access token (too short: {len(access_token)} chars). "
                       "Kite access tokens should be at least 20 characters. "
                       "The token you're trying to save appears to be invalid or corrupted. "
                       "This usually means the token exchange failed. Please check: "
                       "1) Your Kite API Key and Secret are correct in the Config page, "
                       "2) The redirect URI matches exactly in your Kite Connect app settings, "
                       "3) The request_token is fresh (they expire quickly). "
                       "Try generating a new request_token and use it immediately.",
                field="access_token"
            )
        
        # Store access token
        config_path = Path("config")
        config_path.mkdir(exist_ok=True)
        
        with open("config/access_token.txt", "w") as f:
            f.write(access_token.strip())
        
        log_info(f"Access token stored successfully in config/access_token.txt (length: {len(access_token)})")
        return {
            "status": "success", 
            "message": "Access token stored successfully",
            "access_token": access_token[:20] + "..." if access_token else None
        }
    except AlgoFeastException:
        raise
    except Exception as e:
        log_error(f"Unexpected error: {str(e)}")
        raise AlgoFeastException(
            message=f"Error setting token: {str(e)}",
            status_code=500,
            error_code="INTERNAL_ERROR"
        )

# Legacy agent endpoints (kept for backward compatibility)
@app.post("/agent/chat")
async def agent_chat(req: Request):
    """Natural language interaction with the AI agent (Legacy - use /api/v1/agent/chat)"""
    try:
        payload = await req.json()
        user_query = payload.get("message", "")
        session_id = payload.get("session_id", "default")

        if not user_query:
            raise ValidationError(message="Message is required", field="message")

        # Save user message to database
        try:
            chat_repo = get_chat_repository()
            user_message = ChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="user",
                content=user_query,
                timestamp=datetime.now()
            )
            chat_repo.save(user_message)
        except Exception as e:
            log_error(f"Error saving user message: {e}")
        
        # Get context (positions, balance, etc.)
        context = {}
        try:
            kite = get_kite_instance()
            positions = kite.positions().get("net", [])
            margins = kite.margins()
            context = {
                "positions": positions,
                "balance": margins.get("equity", {})
            }
        except:
            pass  # Continue without context if auth fails
        
        # Run agent
        result = await run_agent(user_query, context)

        # Save assistant response to database
        try:
            chat_repo = get_chat_repository()
            assistant_message = ChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="assistant",
                content=result.get("response", ""),
                timestamp=datetime.now(),
                metadata={"agent_result": result}
            )
            chat_repo.save(assistant_message)
        except Exception as e:
            log_error(f"Error saving assistant message: {e}")

        return {
            "status": "success",
            "data": result
        }
    except AlgoFeastException:
        raise
    except Exception as e:
        log_error(f"Error in agent chat: {str(e)}")
        raise AlgoFeastException(
            message=f"Error in agent chat: {str(e)}",
            status_code=500,
            error_code="INTERNAL_ERROR"
        )

@app.post("/agent/execute")
async def agent_execute(req: Request):
    """Direct agent execution with approval workflow (Legacy - use /api/v1/agent/chat)"""
    try:
        payload = await req.json()
        user_query = payload.get("message", "")
        auto_approve = payload.get("auto_approve", False)
        
        if not user_query:
            raise ValidationError(message="Message is required", field="message")
        
        # Get context
        context = {}
        try:
            kite = get_kite_instance()
            positions = kite.positions().get("net", [])
            margins = kite.margins()
            context = {
                "positions": positions,
                "balance": margins.get("equity", {})
            }
        except:
            pass
        
        # Run agent
        result = await run_agent(user_query, context)
        
        # If requires approval and not auto-approved, return approval info
        if result.get("requires_approval") and not auto_approve:
            return {
                "status": "pending_approval",
                "data": result,
                "approval_id": result.get("approval_id")
            }
        
        return {
            "status": "success",
            "data": result
        }
    except AlgoFeastException:
        raise
    except Exception as e:
        log_error(f"Error in agent execute: {str(e)}")
        raise AlgoFeastException(
            message=f"Error in agent execute: {str(e)}",
            status_code=500,
            error_code="INTERNAL_ERROR"
        )

# Legacy endpoint removed - use /api/v1/agent/status instead
