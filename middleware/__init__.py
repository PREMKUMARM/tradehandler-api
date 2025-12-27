"""
Enterprise middleware components
"""
from .logging import LoggingMiddleware
from .request_id import RequestIDMiddleware
from .error_handler import ErrorHandlerMiddleware

# Optional middleware (if they exist)
try:
    from .security import SecurityHeadersMiddleware
except ImportError:
    SecurityHeadersMiddleware = None

try:
    from .rate_limit import RateLimitMiddleware
except ImportError:
    RateLimitMiddleware = None

__all__ = [
    "LoggingMiddleware",
    "RequestIDMiddleware",
    "ErrorHandlerMiddleware",
]

if SecurityHeadersMiddleware:
    __all__.append("SecurityHeadersMiddleware")
if RateLimitMiddleware:
    __all__.append("RateLimitMiddleware")

