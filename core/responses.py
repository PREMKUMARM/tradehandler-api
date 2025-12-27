"""
Standardized API response models
"""
from typing import Optional, Any, Dict, List, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar('T')


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper"""
    status: str = Field(..., description="Response status: 'success' or 'error'")
    message: Optional[str] = Field(None, description="Human-readable message")
    data: Optional[T] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class SuccessResponse(APIResponse[T]):
    """Success response"""
    status: str = "success"
    message: Optional[str] = None


class ErrorResponse(APIResponse[None]):
    """Error response"""
    status: str = "error"
    error_code: Optional[str] = Field(None, description="Error code for programmatic handling")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response"""
    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status: 'healthy', 'degraded', or 'unhealthy'")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now)
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    checks: Dict[str, Any] = Field(default_factory=dict, description="Component health checks")

