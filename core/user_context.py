"""
User context utilities for extracting and managing user identification
"""
from typing import Optional
from fastapi import Request, Header
from core.jwt_utils import get_user_id_from_token_or_header


def get_user_id_from_request(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
) -> str:
    """
    Extract user ID from request.
    Priority:
    1. JWT token (Authorization header)
    2. X-User-ID header
    3. user_id query parameter
    4. Default to 'default' for backward compatibility
    """
    return get_user_id_from_token_or_header(request, authorization, x_user_id)


def get_user_id_from_header(x_user_id: Optional[str] = None) -> str:
    """Extract user ID from X-User-ID header"""
    if x_user_id:
        return x_user_id.strip()
    return "default"

