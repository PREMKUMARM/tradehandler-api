"""
Rate limiting middleware
"""
import time
from typing import Callable, Dict
from collections import defaultdict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import get_settings
from core.exceptions import RateLimitError


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting (for production, use Redis)"""
    
    def __init__(self, app, *args, **kwargs):
        super().__init__(app)
        self.requests: Dict[str, list] = defaultdict(list)
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.settings.rate_limit_enabled:
            return await call_next(request)
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/healthz", "/api/health"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.settings.rate_limit_per_minute:
            raise RateLimitError(
                limit=self.settings.rate_limit_per_minute,
                window=60,
                details={"client_id": client_id}
            )
        
        # Record request
        self.requests[client_id].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.settings.rate_limit_per_minute - len(self.requests[client_id]))
        response.headers["X-RateLimit-Limit"] = str(self.settings.rate_limit_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time) + 60)
        
        return response

