"""
Agent-related API endpoints
"""
from fastapi import APIRouter, Request, HTTPException, Depends
from typing import Optional
import asyncio
import uuid
from datetime import datetime

from core.responses import SuccessResponse, ErrorResponse, APIResponse
from core.exceptions import ValidationError, NotFoundError
from core.user_context import get_user_id_from_request
from agent.approval import get_approval_queue
from agent.config import get_agent_config
from agent.user_config import get_user_config, save_user_config
from agent.graph import run_agent
from agent.ws_manager import broadcast_agent_update, add_agent_log
from agent.tools.kite_tools import place_order_tool, cancel_order_tool, place_gtt_tool
from database.repositories import get_chat_repository
from database.models import ChatMessage
from schemas.agent import (
    ChatRequest, ChatResponse, ApprovalRequest, RejectionRequest, ConfigUpdateRequest
)

router = APIRouter(prefix="/agent", tags=["Agent"])


def get_request_id(request: Request) -> str:
    """Get request ID from request state"""
    return getattr(request.state, "request_id", "unknown")


@router.post("/chat", response_model=APIResponse[ChatResponse])
async def agent_chat(
    request: Request,
    chat_request: ChatRequest
):
    """
    Natural language interaction with the AI agent
    
    - **message**: User message
    - **session_id**: Conversation session ID
    """
    request_id = get_request_id(request)
    
    try:
        # Save user message to database
        chat_repo = get_chat_repository()
        user_message = ChatMessage(
            message_id=str(uuid.uuid4()),
            session_id=chat_request.session_id,
            role="user",
            content=chat_request.message,
            timestamp=datetime.now()
        )
        chat_repo.save(user_message)
        
        # Get context (positions, balance, etc.)
        context = chat_request.context or {}
        try:
            from utils.kite_utils import get_kite_instance
            kite = get_kite_instance()
            positions = kite.positions().get("net", [])
            margins = kite.margins()
            context.update({
                "positions": positions,
                "balance": margins.get("equity", {})
            })
        except Exception:
            pass  # Continue without context if auth fails
        
        # Run agent
        result = await run_agent(chat_request.message, context)
        
        # Save assistant response to database
        assistant_message = ChatMessage(
            message_id=str(uuid.uuid4()),
            session_id=chat_request.session_id,
            role="assistant",
            content=result.get("response", ""),
            timestamp=datetime.now(),
            metadata={"tool_calls": result.get("tool_calls", [])}
        )
        chat_repo.save(assistant_message)
        
        return APIResponse(
            status="success",
            data=ChatResponse(
                response=result.get("response", ""),
                session_id=chat_request.session_id,
                metadata=result
            ),
            request_id=request_id
        )
    except Exception as e:
        add_agent_log(f"[{request_id}] Chat error: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=f"Error in agent chat: {str(e)}")


@router.get("/status")
async def get_agent_status(request: Request):
    """Get agent status and health"""
    request_id = get_request_id(request)
    
    try:
        from agent.graph import get_agent_instance
        agent = get_agent_instance()
        config = get_agent_config()
        
        return SuccessResponse(
            data={
                "agent_initialized": agent is not None,
                "autonomous_mode": config.autonomous_mode,
                "auto_trade_enabled": config.is_auto_trade_enabled,
                "trading_capital": config.trading_capital,
                "active_strategies": config.active_strategies
            },
            request_id=request_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent status: {str(e)}")


@router.get("/config")
async def get_agent_config_endpoint(request: Request):
    """Get agent configuration"""
    request_id = get_request_id(request)
    user_id = get_user_id_from_request(request)
    
    try:
        # Load user-specific config from database
        config = get_user_config(user_id=user_id)
        
        # Try to get live funds from Zerodha if possible
        zerodha_funds = 0.0
        try:
            from utils.kite_utils import get_kite_instance
            kite = get_kite_instance()
            margins = kite.margins()
            equity_data = margins.get('equity', {})
            available_value = equity_data.get('available', 0)
            
            if isinstance(available_value, dict):
                zerodha_funds = float(available_value.get('cash', 0))
                if zerodha_funds == 0:
                    zerodha_funds = float(equity_data.get('opening_balance', 0) or equity_data.get('live_balance', 0))
            else:
                zerodha_funds = float(available_value if available_value else equity_data.get('cash', 0) or equity_data.get('opening_balance', 0))
        except Exception:
            pass
        
        # AUTO-SYNC: If Zerodha funds are available, auto-calculate capital protection values
        # This ensures values are always in sync with actual account balance
        auto_trade_threshold = config.auto_trade_threshold
        max_position_size = config.max_position_size
        trading_capital = config.trading_capital
        daily_loss_limit = config.daily_loss_limit
        circuit_breaker_loss_threshold = config.circuit_breaker_loss_threshold
        
        if zerodha_funds > 0:
            # Auto-sync values based on Zerodha balance
            # Only override if current values don't match expected percentages (allowing for manual overrides)
            if trading_capital == 0 or abs(trading_capital - zerodha_funds) > 1000:
                trading_capital = zerodha_funds
            
            if max_position_size == 0 or max_position_size > zerodha_funds:
                max_position_size = zerodha_funds
            
            # Daily Loss Limit = 2.5% of capital (Conservative safety)
            expected_daily_loss = zerodha_funds * 0.025
            if daily_loss_limit == 0 or abs(daily_loss_limit - expected_daily_loss) > 500:
                daily_loss_limit = expected_daily_loss
            
            # Auto-Approve Threshold = 2.5% of Balance
            expected_auto_approve = zerodha_funds * 0.025
            if auto_trade_threshold == 0 or abs(auto_trade_threshold - expected_auto_approve) > 500:
                auto_trade_threshold = expected_auto_approve
            
            # Circuit Breaker = 5% of Capital
            expected_circuit_breaker = zerodha_funds * 0.05
            if circuit_breaker_loss_threshold == 0 or abs(circuit_breaker_loss_threshold - expected_circuit_breaker) > 1000:
                circuit_breaker_loss_threshold = expected_circuit_breaker
        
        config_data = {
            "llm_provider": config.llm_provider.value,
            "openai_api_key": config.openai_api_key or "",
            "anthropic_api_key": config.anthropic_api_key or "",
            "ollama_base_url": config.ollama_base_url,
            "agent_model": config.agent_model,
            "agent_temperature": config.agent_temperature,
            "max_tokens": config.max_tokens,
            "auto_trade_threshold": auto_trade_threshold,
            "max_position_size": max_position_size,
            "trading_capital": trading_capital,
            "daily_loss_limit": daily_loss_limit,
            "max_trades_per_day": config.max_trades_per_day,
            "risk_per_trade_pct": config.risk_per_trade_pct,
            "reward_per_trade_pct": config.reward_per_trade_pct,
            "autonomous_mode": config.autonomous_mode,
            "autonomous_scan_interval_mins": config.autonomous_scan_interval_mins,
            "autonomous_target_group": config.autonomous_target_group,
            "active_strategies": config.active_strategies,
            "is_auto_trade_enabled": config.is_auto_trade_enabled,
            "vwap_proximity_pct": config.vwap_proximity_pct,
            "vwap_group_proximity_pct": config.vwap_group_proximity_pct,
            "rejection_shadow_pct": config.rejection_shadow_pct,
            "prime_session_start": config.prime_session_start,
            "prime_session_end": config.prime_session_end,
            "intraday_square_off_time": config.intraday_square_off_time,
            "trading_start_time": config.trading_start_time,
            "trading_end_time": config.trading_end_time,
            "circuit_breaker_enabled": config.circuit_breaker_enabled,
            "circuit_breaker_loss_threshold": circuit_breaker_loss_threshold,
            "use_gtt_orders": config.use_gtt_orders,
            "gtt_for_intraday": config.gtt_for_intraday,
            "gtt_for_positional": config.gtt_for_positional,
            "kite_api_key": config.kite_api_key or "",
            "kite_api_secret": config.kite_api_secret or "",
            "kite_redirect_uri": config.kite_redirect_uri,
            "zerodha_funds": zerodha_funds
        }
        
        return SuccessResponse(data=config_data, request_id=request_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent config: {str(e)}")


@router.post("/config")
async def update_agent_config(
    request: Request,
    config_update: ConfigUpdateRequest
):
    """Update agent configuration for the logged-in user"""
    request_id = get_request_id(request)
    user_id = get_user_id_from_request(request)
    
    try:
        # Load current user config
        config = get_user_config(user_id=user_id)
        update_dict = config_update.model_dump(exclude_unset=True)
        
        # Update config fields
        if "llm_provider" in update_dict:
            from agent.config import LLMProvider
            config.llm_provider = LLMProvider(update_dict["llm_provider"])
        for field in [
            "openai_api_key", "anthropic_api_key", "ollama_base_url",
            "agent_model", "agent_temperature", "max_tokens", "auto_trade_threshold",
            "max_position_size", "trading_capital", "daily_loss_limit",
            "max_trades_per_day", "risk_per_trade_pct", "reward_per_trade_pct",
            "autonomous_mode", "autonomous_scan_interval_mins", "autonomous_target_group",
            "active_strategies", "is_auto_trade_enabled", "vwap_proximity_pct",
            "vwap_group_proximity_pct", "rejection_shadow_pct", "prime_session_start",
            "prime_session_end",             "intraday_square_off_time", "trading_start_time",
            "trading_end_time", "circuit_breaker_enabled", "circuit_breaker_loss_threshold",
            "use_gtt_orders", "gtt_for_intraday", "gtt_for_positional",
            "kite_api_key", "kite_api_secret", "kite_redirect_uri"
        ]:
            if field in update_dict:
                setattr(config, field, update_dict[field])
        
        # Persist to .env file
        from pathlib import Path
        env_file = Path(".env")
        env_content = []
        if env_file.exists():
            env_content = env_file.read_text().splitlines()
        
        # Update or add config values
        env_dict = {}
        for line in env_content:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                env_dict[key.strip()] = value.strip()
        
        # Update with new values
        env_dict.update({
            "LLM_PROVIDER": config.llm_provider.value,
            "OPENAI_API_KEY": config.openai_api_key or "",
            "ANTHROPIC_API_KEY": config.anthropic_api_key or "",
            "OLLAMA_BASE_URL": config.ollama_base_url,
            "AGENT_MODEL": config.agent_model,
            "AGENT_TEMPERATURE": str(config.agent_temperature),
            "MAX_TOKENS": str(config.max_tokens),
            "AUTO_TRADE_THRESHOLD": str(config.auto_trade_threshold),
            "MAX_POSITION_SIZE": str(config.max_position_size),
            "TRADING_CAPITAL": str(config.trading_capital),
            "DAILY_LOSS_LIMIT": str(config.daily_loss_limit),
            "MAX_TRADES_PER_DAY": str(config.max_trades_per_day),
            "RISK_PER_TRADE_PCT": str(config.risk_per_trade_pct),
            "REWARD_PER_TRADE_PCT": str(config.reward_per_trade_pct),
            "AUTONOMOUS_MODE": str(config.autonomous_mode),
            "AUTONOMOUS_SCAN_INTERVAL_MINS": str(config.autonomous_scan_interval_mins),
            "AUTONOMOUS_TARGET_GROUP": config.autonomous_target_group,
            "ACTIVE_STRATEGIES": config.active_strategies,
            "IS_AUTO_TRADE_ENABLED": str(config.is_auto_trade_enabled),
            "VWAP_PROXIMITY_PCT": str(config.vwap_proximity_pct),
            "VWAP_GROUP_PROXIMITY_PCT": str(config.vwap_group_proximity_pct),
            "REJECTION_SHADOW_PCT": str(config.rejection_shadow_pct),
            "PRIME_SESSION_START": config.prime_session_start,
            "PRIME_SESSION_END": config.prime_session_end,
            "INTRADAY_SQUARE_OFF_TIME": config.intraday_square_off_time,
            "TRADING_START_TIME": config.trading_start_time,
            "TRADING_END_TIME": config.trading_end_time,
            "CIRCUIT_BREAKER_ENABLED": str(config.circuit_breaker_enabled),
            "CIRCUIT_BREAKER_LOSS_THRESHOLD": str(config.circuit_breaker_loss_threshold),
            "USE_GTT_ORDERS": str(config.use_gtt_orders),
            "GTT_FOR_INTRADAY": str(config.gtt_for_intraday),
            "GTT_FOR_POSITIONAL": str(config.gtt_for_positional),
            "KITE_API_KEY": config.kite_api_key or "",
            "KITE_API_SECRET": config.kite_api_secret or "",
            "KITE_REDIRECT_URI": config.kite_redirect_uri,
        })
        
        # Save user-specific config to database
        save_user_config(user_id=user_id, config=config)
        
        # Write back to .env (for backward compatibility - global defaults)
        with open(".env", "w") as f:
            for key, value in env_dict.items():
                f.write(f"{key}={value}\n")
        
        # Broadcast config update
        asyncio.create_task(broadcast_agent_update("CONFIG_UPDATED", update_dict))
        
        return SuccessResponse(
            message="Configuration updated successfully",
            data=update_dict,
            request_id=request_id
        )
    except Exception as e:
        add_agent_log(f"[{request_id}] Config update error: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=f"Error updating agent config: {str(e)}")


@router.get("/approvals")
async def get_approvals(request: Request):
    """Get pending approvals"""
    request_id = get_request_id(request)
    
    try:
        approval_queue = get_approval_queue()
        approvals = approval_queue.list_pending()
        
        return SuccessResponse(
            data={"approvals": approvals},
            request_id=request_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting approvals: {str(e)}")


@router.get("/approved-trades")
async def get_approved_trades(request: Request):
    """Get all approved trades"""
    request_id = get_request_id(request)
    
    try:
        approval_queue = get_approval_queue()
        approved = approval_queue.list_approved()
        
        return SuccessResponse(
            data={"trades": approved},
            request_id=request_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting approved trades: {str(e)}")


@router.post("/approve/{approval_id}")
async def approve_action(
    approval_id: str,
    request: Request,
    approval_request: ApprovalRequest
):
    """Approve a pending action"""
    request_id = get_request_id(request)
    
    try:
        approval_queue = get_approval_queue()
        success = approval_queue.approve(approval_id, approval_request.approved_by)
        
        if not success:
            raise NotFoundError("Approval", approval_id)
        
        approval = approval_queue.get_approval(approval_id)
        if not approval:
            raise NotFoundError("Approval", approval_id)
        
        # Execute trade (existing logic from main.py)
        execution_msg = ""
        action = approval.get("action", "")
        details = approval.get("details", {})
        
        if action.startswith("LIVE_") and not details.get("is_simulated", False):
            try:
                symbol = details.get("symbol")
                transaction_type = details.get("type", "BUY")
                quantity = int(details.get("qty", 1))
                price = float(details.get("price", 0))
                
                add_agent_log(f"[{request_id}] Executing Approved Trade: {transaction_type} {quantity} {symbol}...", "signal")
                
                # Place entry order
                order_result = place_order_tool.invoke({
                    "tradingsymbol": symbol,
                    "transaction_type": transaction_type,
                    "quantity": quantity,
                    "order_type": "MARKET",
                    "product": "MIS",
                    "exchange": "NSE"
                })
                
                if order_result.get("status") == "success":
                    entry_order_id = str(order_result.get('order_id'))
                    execution_msg = f"Entry Order Placed: {entry_order_id}"
                    add_agent_log(f"[{request_id}] SUCCESS: {execution_msg}", "info")
                    
                    approval_queue.repo.update_order_ids(approval_id, entry_order_id=entry_order_id)
                    
                    # Place exit orders (GTT or regular)
                    config = get_agent_config()
                    exit_type = "SELL" if transaction_type == "BUY" else "BUY"
                    sl_price = float(details.get("sl", 0))
                    tp_price = float(details.get("tp", 0))
                    
                    product = details.get("product", "MIS")
                    use_gtt = False
                    if config.use_gtt_orders:
                        if product == "CNC" and config.gtt_for_positional:
                            use_gtt = True
                        elif product == "MIS" and config.gtt_for_intraday:
                            use_gtt = True
                    
                    if use_gtt and sl_price > 0 and tp_price > 0:
                        try:
                            from utils.kite_utils import get_kite_instance
                            kite = get_kite_instance()
                            quote = kite.quote(f"NSE:{symbol}")
                            instrument_key = f"NSE:{symbol}"
                            current_price = quote[instrument_key].get("last_price", price) if instrument_key in quote else price
                            
                            if transaction_type == "BUY":
                                sl_trigger = sl_price * 1.001
                                tp_trigger = tp_price * 0.999
                            else:
                                sl_trigger = sl_price * 0.999
                                tp_trigger = tp_price * 1.001
                            
                            gtt_result = place_gtt_tool.invoke({
                                "tradingsymbol": symbol,
                                "exchange": "NSE",
                                "trigger_type": "two-leg",
                                "trigger_prices": [sl_trigger, tp_trigger],
                                "last_price": current_price,
                                "stop_loss_price": round(sl_price, 1),
                                "target_price": round(tp_price, 1),
                                "quantity": quantity,
                                "transaction_type": exit_type,
                                "product": product
                            })
                            
                            if gtt_result.get("status") == "success":
                                gtt_trigger_id = str(gtt_result.get("trigger_id"))
                                add_agent_log(f"[{request_id}] ✅ GTT OCO Order Placed: {gtt_trigger_id}", "info")
                                approval_queue.repo.update_order_ids(approval_id, sl_order_id=gtt_trigger_id)
                                execution_msg += f" | GTT OCO: {gtt_trigger_id}"
                            else:
                                use_gtt = False
                        except Exception as e:
                            add_agent_log(f"[{request_id}] GTT Error: {e}", "warning")
                            use_gtt = False
                    
                    if not use_gtt:
                        # Place regular SL-M and LIMIT orders
                        if sl_price > 0:
                            sl_result = place_order_tool.invoke({
                                "tradingsymbol": symbol,
                                "transaction_type": exit_type,
                                "quantity": quantity,
                                "order_type": "SL-M",
                                "trigger_price": round(sl_price, 1),
                                "product": product,
                                "exchange": "NSE"
                            })
                            if sl_result.get("status") == "success":
                                sl_order_id = str(sl_result.get('order_id'))
                                approval_queue.repo.update_order_ids(approval_id, sl_order_id=sl_order_id)
                        
                        if tp_price > 0:
                            tp_result = place_order_tool.invoke({
                                "tradingsymbol": symbol,
                                "transaction_type": exit_type,
                                "quantity": quantity,
                                "order_type": "LIMIT",
                                "price": round(tp_price, 1),
                                "product": product,
                                "exchange": "NSE"
                            })
                            if tp_result.get("status") == "success":
                                tp_order_id = str(tp_result.get('order_id'))
                                approval_queue.repo.update_order_ids(approval_id, tp_order_id=tp_order_id)
                else:
                    execution_msg = f"Order Failed: {order_result.get('error')}"
                    
            except Exception as e:
                execution_msg = f"Execution Error: {str(e)}"
                add_agent_log(f"[{request_id}] Execution error: {execution_msg}", "error")
        else:
            execution_msg = "Simulation Approval recorded (No live order placed)"
        
        # Broadcast update
        asyncio.create_task(broadcast_agent_update("APPROVAL_PROCESSED", {
            "approval_id": approval_id,
            "status": "APPROVED",
            "approval": approval,
            "execution_message": execution_msg
        }))
        
        return SuccessResponse(
            data={
                "approval_id": approval_id,
                "approval": approval,
                "message": execution_msg
            },
            request_id=request_id
        )
    except NotFoundError:
        raise
    except Exception as e:
        add_agent_log(f"[{request_id}] Approval error: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=f"Error approving action: {str(e)}")


@router.post("/reject/{approval_id}")
async def reject_action(
    approval_id: str,
    request: Request,
    rejection_request: RejectionRequest
):
    """Reject a pending action"""
    request_id = get_request_id(request)
    
    try:
        approval_queue = get_approval_queue()
        success = approval_queue.reject(
            approval_id,
            rejection_request.reason,
            rejection_request.rejected_by
        )
        
        if not success:
            raise NotFoundError("Approval", approval_id)
        
        approval = approval_queue.get_approval(approval_id)
        
        asyncio.create_task(broadcast_agent_update("APPROVAL_PROCESSED", {
            "approval_id": approval_id,
            "status": "REJECTED",
            "approval": approval
        }))
        
        return SuccessResponse(
            data={
                "approval_id": approval_id,
                "approval": approval,
                "message": "Rejection successful"
            },
            request_id=request_id
        )
    except NotFoundError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rejecting action: {str(e)}")

