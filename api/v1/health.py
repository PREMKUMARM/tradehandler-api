"""
Health check and monitoring endpoints
"""
import time
from fastapi import APIRouter, Depends
from datetime import datetime

# Optional dependency for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from core.responses import HealthCheckResponse, SuccessResponse
from core.config import get_settings
from utils.kite_utils import get_kite_instance
from database.connection import get_database

router = APIRouter()

# Application start time
_start_time = time.time()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    settings = get_settings()
    uptime = time.time() - _start_time
    
    checks = {
        "database": "unknown",
        "kite_connect": "unknown",
        "memory": "unknown"
    }
    
    # Check database
    try:
        db = get_database()
        db.get_connection()
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
    
    # Check Kite Connect
    try:
        kite = get_kite_instance()
        kite.margins()  # Test connection
        checks["kite_connect"] = "healthy"
    except Exception:
        checks["kite_connect"] = "unavailable"  # Not necessarily unhealthy
    
    # Check memory (if psutil is available)
    if PSUTIL_AVAILABLE:
        try:
            memory = psutil.virtual_memory()
            checks["memory"] = {
                "percent": memory.percent,
                "available_mb": memory.available / (1024 * 1024),
                "total_mb": memory.total / (1024 * 1024)
            }
        except Exception:
            checks["memory"] = "unknown"
    else:
        checks["memory"] = "unavailable (psutil not installed)"
    
    # Determine overall status
    if checks["database"] == "healthy":
        status = "healthy"
    elif checks["database"] == "unavailable":
        status = "degraded"
    else:
        status = "unhealthy"
    
    return HealthCheckResponse(
        status=status,
        version=settings.app_version,
        timestamp=datetime.now(),
        uptime_seconds=uptime,
        checks=checks
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check (for Kubernetes)"""
    try:
        # Check critical dependencies
        db = get_database()
        db.get_connection()
        return SuccessResponse(message="Service is ready")
    except Exception as e:
        from fastapi import status
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "message": f"Service not ready: {str(e)}"}
        )


@router.get("/live")
async def liveness_check():
    """Liveness check (for Kubernetes)"""
    return SuccessResponse(message="Service is alive")

