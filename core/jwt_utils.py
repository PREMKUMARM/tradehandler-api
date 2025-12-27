"""
JWT token utilities for extracting user information
"""
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, Header

# Try to import jwt - make it optional
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None


def extract_user_id_from_jwt(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization")
) -> Optional[str]:
    """
    Extract user ID from JWT token in Authorization header.
    Format: "Bearer <token>"
    
    Returns None if token is invalid or missing.
    """
    if not JWT_AVAILABLE or not jwt:
        return None
    
    if not authorization:
        return None
    
    try:
        # Extract token from "Bearer <token>" format
        if not authorization.startswith("Bearer "):
            return None
        
        token = authorization.replace("Bearer ", "").strip()
        if not token:
            return None
        
        # Decode JWT token (without verification for now - adjust based on your JWT secret)
        # In production, you should verify the token signature
        try:
            # Try to decode without verification first (for development)
            # In production, use: jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            decoded = jwt.decode(token, options={"verify_signature": False})
            
            # Extract user_id from token payload
            # Common JWT claims: sub, user_id, id, email
            user_id = (
                decoded.get("user_id") or
                decoded.get("sub") or
                decoded.get("id") or
                decoded.get("email") or
                None
            )
            
            return str(user_id) if user_id else None
            
        except jwt.DecodeError:
            # Token is invalid
            return None
        except Exception as e:
            # Other errors
            return None
            
    except Exception:
        return None


def get_user_id_from_token_or_header(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
) -> str:
    """
    Get user ID from JWT token (priority 1) or X-User-ID header (priority 2).
    Falls back to 'default' if neither is available.
    """
    # Try JWT token first
    if authorization:
        # Handle Header object - extract string value if it's a Header object
        auth_str = authorization if isinstance(authorization, str) else str(authorization) if authorization else None
        if auth_str:
            user_id = extract_user_id_from_jwt(request, auth_str)
            if user_id:
                return user_id
    
    # Try X-User-ID header
    if x_user_id:
        # Handle Header object - extract string value if it's a Header object
        user_id_str = x_user_id if isinstance(x_user_id, str) else str(x_user_id) if x_user_id else None
        if user_id_str:
            return user_id_str.strip()
    
    # Try query parameter
    user_id = request.query_params.get("user_id")
    if user_id:
        return user_id.strip()
    
    # Default fallback
    return "default"

