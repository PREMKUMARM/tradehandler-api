"""
Request/response logging middleware
"""
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from utils.logger import log_agent_activity


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all HTTP requests and responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get request ID
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request
        start_time = time.time()
        log_agent_activity(
            f"[{request_id}] {request.method} {request.url.path}",
            "info"
        )
        
        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            log_agent_activity(
                f"[{request_id}] {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)",
                "info" if response.status_code < 400 else "error"
            )
            
            # Add process time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            log_agent_activity(
                f"[{request_id}] {request.method} {request.url.path} - ERROR: {str(e)} ({process_time:.3f}s)",
                "error"
            )
            raise
