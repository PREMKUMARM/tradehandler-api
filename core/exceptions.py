"""
Custom exception classes for enterprise-level error handling
"""
from typing import Optional, Dict, Any


class AlgoFeastException(Exception):
    """Base exception for all AlgoFeast errors"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or "INTERNAL_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(AlgoFeastException):
    """Validation error for invalid input"""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details={"field": field, **(details or {})}
        )
        self.field = field


class AuthenticationError(AlgoFeastException):
    """Authentication/authorization error"""
    
    def __init__(self, message: str = "Authentication required", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR",
            details=details or {}
        )


class NotFoundError(AlgoFeastException):
    """Resource not found error"""
    
    def __init__(self, resource: str, identifier: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"
        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND",
            details={"resource": resource, "identifier": identifier, **(details or {})}
        )
        self.resource = resource
        self.identifier = identifier


class BusinessLogicError(AlgoFeastException):
    """Business logic violation error"""
    
    def __init__(self, message: str, error_code: str = "BUSINESS_LOGIC_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=422,
            error_code=error_code,
            details=details or {}
        )


class ExternalAPIError(AlgoFeastException):
    """Error from external API (e.g., Kite Connect)"""
    
    def __init__(self, message: str, service: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"{service} API error: {message}",
            status_code=502,
            error_code="EXTERNAL_API_ERROR",
            details={"service": service, **(details or {})}
        )
        self.service = service

