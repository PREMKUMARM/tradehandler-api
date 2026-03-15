"""
Monitoring and analytics API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Dict, List
from datetime import datetime, timedelta

from utils.performance_monitor import performance_monitor
from utils.cache_manager import cache_manager, redis_cache
from utils.trade_limits import trade_limits
from utils.strategy_executor import strategy_executor
from utils.order_monitor import order_monitor
from utils.websocket_manager import kite_ws_manager, portfolio_ws_manager
from utils.logger import log_info, log_error

router = APIRouter(prefix="/monitoring", tags=["Monitoring & Analytics"])


@router.get("/system")
async def get_system_metrics():
    """Get system performance metrics"""
    try:
        system_metrics = performance_monitor.get_system_metrics()
        uptime = performance_monitor.get_uptime()
        
        return {
            "data": {
                "system": system_metrics,
                "uptime": uptime,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        log_error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_metrics(minutes: int = 5):
    """Get application performance metrics"""
    try:
        if minutes < 1 or minutes > 1440:  # Max 24 hours
            raise HTTPException(status_code=400, detail="Minutes must be between 1 and 1440")
        
        metrics = await performance_monitor.get_metrics_summary(minutes)
        
        return {
            "data": {
                "metrics": metrics,
                "timeframe_minutes": minutes,
                "timestamp": datetime.now().isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/endpoints")
async def get_endpoint_stats(minutes: int = 60, endpoint: Optional[str] = None):
    """Get endpoint-specific statistics"""
    try:
        if minutes < 1 or minutes > 1440:
            raise HTTPException(status_code=400, detail="Minutes must be between 1 and 1440")
        
        stats = await performance_monitor.get_endpoint_stats(endpoint, minutes)
        
        return {
            "data": {
                "endpoint_stats": stats,
                "timeframe_minutes": minutes,
                "endpoint_filter": endpoint,
                "timestamp": datetime.now().isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error getting endpoint stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache")
async def get_cache_stats():
    """Get cache performance statistics"""
    try:
        memory_cache_stats = cache_manager.get_stats()
        
        redis_stats = {}
        if redis_cache:
            try:
                # Test Redis connection
                test_key = "health_check"
                await redis_cache.set(test_key, "test", ttl=10)
                test_value = await redis_cache.get(test_key)
                await redis_cache.delete(test_key)
                
                redis_stats = {
                    "status": "connected",
                    "test_result": test_value == "test"
                }
            except Exception as e:
                redis_stats = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            redis_stats = {
                "status": "not_available"
            }
        
        return {
            "data": {
                "memory_cache": memory_cache_stats,
                "redis_cache": redis_stats,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        log_error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache(cache_type: str = "memory"):
    """Clear cache (memory or redis)"""
    try:
        if cache_type not in ["memory", "redis", "all"]:
            raise HTTPException(status_code=400, detail="cache_type must be 'memory', 'redis', or 'all'")
        
        cleared = {}
        
        if cache_type in ["memory", "all"]:
            await cache_manager.clear()
            cleared["memory"] = True
        
        if cache_type in ["redis", "all"] and redis_cache:
            await redis_cache.clear()
            cleared["redis"] = True
        elif cache_type in ["redis", "all"]:
            cleared["redis"] = False
        
        return {
            "status": "success",
            "message": f"Cleared cache: {', '.join(cleared.keys())}",
            "cleared": cleared
        }
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trading-system")
async def get_trading_system_status():
    """Get comprehensive trading system status"""
    try:
        # Trade limits status
        trade_limits_status = trade_limits.get_limits_status()
        
        # Strategy executor status
        strategy_status = strategy_executor.get_strategy_status()
        
        # Order monitor status
        order_monitor_status = {
            "is_running": order_monitor.is_running,
            "monitored_orders": len(order_monitor.monitored_orders),
            "monitored_orders_details": [
                {
                    "order_id": order_id,
                    "symbol": order.get("symbol"),
                    "transaction_type": order.get("transaction_type"),
                    "stoploss": order.get("stoploss"),
                    "target": order.get("target"),
                    "created_at": order.get("created_at").isoformat() if order.get("created_at") else None
                }
                for order_id, order in order_monitor.monitored_orders.items()
            ]
        }
        
        # WebSocket status
        websocket_status = {
            "market_websocket": {
                "is_running": kite_ws_manager.is_running,
                "active_connections": len(kite_ws_manager.active_connections),
                "subscriptions": {
                    conn_id: list(tokens) for conn_id, tokens in kite_ws_manager.subscriptions.items()
                }
            },
            "portfolio_websocket": {
                "is_running": portfolio_ws_manager.is_running,
                "active_connections": len(portfolio_ws_manager.active_connections)
            }
        }
        
        return {
            "data": {
                "trade_limits": trade_limits_status,
                "strategy_executor": strategy_status,
                "order_monitor": order_monitor_status,
                "websockets": websocket_status,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        log_error(f"Error getting trading system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        health_status = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check system metrics
        try:
            system_metrics = performance_monitor.get_system_metrics()
            if system_metrics.get("cpu", {}).get("percent", 0) > 90:
                health_status["checks"]["cpu"] = "warning"
            else:
                health_status["checks"]["cpu"] = "healthy"
        except Exception as e:
            health_status["checks"]["cpu"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check memory
        try:
            system_metrics = performance_monitor.get_system_metrics()
            if system_metrics.get("memory", {}).get("percent", 0) > 90:
                health_status["checks"]["memory"] = "warning"
            else:
                health_status["checks"]["memory"] = "healthy"
        except Exception as e:
            health_status["checks"]["memory"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check trading systems
        try:
            if not order_monitor.is_running:
                health_status["checks"]["order_monitor"] = "stopped"
                health_status["status"] = "degraded"
            else:
                health_status["checks"]["order_monitor"] = "healthy"
        except Exception as e:
            health_status["checks"]["order_monitor"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check cache
        try:
            cache_stats = cache_manager.get_stats()
            if cache_stats["size"] > cache_stats["max_size"] * 0.9:
                health_status["checks"]["cache"] = "warning"
            else:
                health_status["checks"]["cache"] = "healthy"
        except Exception as e:
            health_status["checks"]["cache"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        log_error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/alerts")
async def get_alerts(minutes: int = 60):
    """Get system alerts and warnings"""
    try:
        alerts = []
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        # Check for high CPU usage
        system_metrics = performance_monitor.get_system_metrics()
        cpu_percent = system_metrics.get("cpu", {}).get("percent", 0)
        if cpu_percent > 80:
            alerts.append({
                "type": "warning",
                "message": f"High CPU usage: {cpu_percent:.1f}%",
                "timestamp": datetime.now().isoformat(),
                "metric": "cpu",
                "value": cpu_percent,
                "threshold": 80
            })
        
        # Check for high memory usage
        memory_percent = system_metrics.get("memory", {}).get("percent", 0)
        if memory_percent > 85:
            alerts.append({
                "type": "warning",
                "message": f"High memory usage: {memory_percent:.1f}%",
                "timestamp": datetime.now().isoformat(),
                "metric": "memory",
                "value": memory_percent,
                "threshold": 85
            })
        
        # Check for trade limits
        limits_status = trade_limits.get_limits_status()
        if not limits_status.get("can_trade", True):
            alerts.append({
                "type": "error",
                "message": "Daily trade limits reached",
                "timestamp": datetime.now().isoformat(),
                "metric": "trade_limits",
                "value": limits_status
            })
        
        # Check for slow response times
        metrics = await performance_monitor.get_metrics_summary(5)
        if "requests" in metrics:
            avg_response_time = metrics["requests"].get("avg_response_time", 0)
            if avg_response_time > 1000:  # 1 second
                alerts.append({
                    "type": "warning",
                    "message": f"Slow response times: {avg_response_time:.0f}ms average",
                    "timestamp": datetime.now().isoformat(),
                    "metric": "response_time",
                    "value": avg_response_time,
                    "threshold": 1000
                })
        
        return {
            "data": {
                "alerts": alerts,
                "count": len(alerts),
                "timeframe_minutes": minutes,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        log_error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))
