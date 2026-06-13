"""
Request/response logging middleware
"""
import time
from typing import Callable, Tuple
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from utils.logger import log_agent_activity

# High-frequency polling endpoints — log successful responses at DEBUG only.
_QUIET_PATH_PREFIXES: Tuple[str, ...] = (
    "/api/v1/execution/exit-trails",
    "/api/v1/execution/paper-trades",
    "/api/v1/execution/live-trades",
    "/api/v1/auth/kite/access-token",
    "/api/v1/auth/refresh-access-token",
    "/api/v1/risk/paper-trading/segments",
    "/api/v1/risk/paper-trading/funds",
    "/api/v1/risk/live-sizing/funds",
    "/api/v1/v2/trade/watch/status",
    "/api/v1/crypto/trade/watch/status",
    "/api/v1/commodity/trade/watch/status",
    "/api/v1/trading/segments/",
    "/api/v1/crypto/trade/config",
)


def _is_quiet_path(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in _QUIET_PATH_PREFIXES)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log HTTP responses (single line per request)."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = getattr(request.state, "request_id", None) or request.headers.get(
            "X-Request-ID", "unknown"
        )
        path = request.url.path
        quiet = _is_quiet_path(path)
        start_time = time.time()

        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            message = (
                f"[{request_id}] {request.method} {path} - "
                f"{response.status_code} ({process_time:.3f}s)"
            )

            if response.status_code >= 400:
                log_agent_activity(message, "error", request_id=request_id)
            elif quiet:
                log_agent_activity(message, "debug", request_id=request_id)
            else:
                log_agent_activity(message, "info", request_id=request_id)

            response.headers["X-Process-Time"] = str(process_time)
            return response
        except Exception as e:
            process_time = time.time() - start_time
            log_agent_activity(
                f"[{request_id}] {request.method} {path} - ERROR: {str(e)} ({process_time:.3f}s)",
                "error",
                request_id=request_id,
            )
            raise
