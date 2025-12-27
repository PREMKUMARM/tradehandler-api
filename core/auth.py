"""
Authentication utilities for JWT token generation and validation
"""
import jwt
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from core.config import get_settings

settings = get_settings()

# JWT secret key - use from environment or generate a default
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 7 days


def generate_jwt_token(user_id: str, email: str, name: Optional[str] = None) -> str:
    """
    Generate JWT token for a user
    
    Args:
        user_id: Unique user identifier
        email: User email address
        name: User display name (optional)
    
    Returns:
        JWT token string
    """
    payload = {
        "user_id": user_id,
        "email": email,
        "name": name,
        "sub": user_id,  # Standard JWT claim
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode JWT token
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_user_id_from_token(token: str) -> Optional[str]:
    """
    Extract user_id from JWT token
    
    Args:
        token: JWT token string
    
    Returns:
        user_id or None if token is invalid
    """
    payload = verify_jwt_token(token)
    if payload:
        return payload.get("user_id") or payload.get("sub")
    return None

