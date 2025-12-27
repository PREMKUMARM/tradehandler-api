"""
Global error handling middleware
"""
import traceback
from typing import Callable
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from starlette.middleware.base import BaseHTTPMiddleware

from core.exceptions import TradeHandlerException
from core.responses import ErrorResponse
from utils.logger import log_agent_activity


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global exception handler middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except TradeHandlerException as e:
            # Handle custom exceptions
            request_id = getattr(request.state, "request_id", "unknown")
            from datetime import datetime
            error_response = ErrorResponse(
                status="error",
                message=e.message,
                error_code=e.error_code,
                details={**e.details, "request_id": request_id},
                timestamp=datetime.now(),
                request_id=request_id
            )
            log_agent_activity(
                f"[{request_id}] TradeHandlerException: {e.message} (Code: {e.error_code})",
                "error"
            )
            # Convert to dict and ensure datetime is JSON serializable
            content = error_response.model_dump(exclude_none=True)
            return JSONResponse(
                status_code=e.status_code,
                content=jsonable_encoder(content)
            )
        except Exception as e:
            # Handle unexpected errors
            request_id = getattr(request.state, "request_id", "unknown")
            error_trace = traceback.format_exc()
            from datetime import datetime
            
            log_agent_activity(
                f"[{request_id}] Unexpected error: {str(e)}\n{error_trace}",
                "error"
            )
            
            # Determine if we should show debug info
            show_debug = getattr(request.app, "debug", False) or getattr(request.app.state, "debug", False)
            
            error_response = ErrorResponse(
                status="error",
                message="An unexpected error occurred",
                error_code="INTERNAL_SERVER_ERROR",
                details={
                    "request_id": request_id,
                    "exception_type": type(e).__name__,
                    **({"traceback": error_trace} if show_debug else {})
                },
                timestamp=datetime.now(),
                request_id=request_id
            )
            # Convert to dict and ensure datetime is JSON serializable
            content = error_response.model_dump(exclude_none=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=jsonable_encoder(content)
            )

