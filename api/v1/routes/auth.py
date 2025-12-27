"""
Authentication API endpoints for Google OAuth
"""
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from typing import Optional
import os
import uuid
from datetime import datetime
from urllib.parse import urlencode

from core.responses import SuccessResponse, ErrorResponse
from core.auth import generate_jwt_token
from core.config import get_settings
from database.user_repository import get_user_repository
from database.models import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

settings = get_settings()

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:4200/auth/callback")

# Google OAuth endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


@router.get("/google/login")
async def google_login(request: Request):
    """
    Initiate Google OAuth login flow
    Redirects user to Google OAuth consent screen
    """
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth is not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET"
        )
    
    # Build OAuth URL
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent"
    }
    
    auth_url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url=auth_url)


@router.get("/google/callback")
async def google_callback(request: Request, code: Optional[str] = None, error: Optional[str] = None):
    """
    Handle Google OAuth callback
    Exchange authorization code for access token and get user info
    """
    if error:
        # User denied access
        return RedirectResponse(
            url=f"http://localhost:4200/auth/login?error={error}"
        )
    
    if not code:
        return RedirectResponse(
            url="http://localhost:4200/auth/login?error=no_code"
        )
    
    try:
        import httpx
        
        # Exchange code for access token
        token_response = httpx.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code"
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if token_response.status_code != 200:
            return RedirectResponse(
                url="http://localhost:4200/auth/login?error=token_exchange_failed"
            )
        
        token_data = token_response.json()
        access_token = token_data.get("access_token")
        
        if not access_token:
            return RedirectResponse(
                url="http://localhost:4200/auth/login?error=no_access_token"
            )
        
        # Get user info from Google
        userinfo_response = httpx.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if userinfo_response.status_code != 200:
            return RedirectResponse(
                url="http://localhost:4200/auth/login?error=userinfo_failed"
            )
        
        userinfo = userinfo_response.json()
        
        # Extract user information
        google_id = userinfo.get("id")
        email = userinfo.get("email")
        name = userinfo.get("name")
        picture = userinfo.get("picture")
        
        if not email:
            return RedirectResponse(
                url="http://localhost:4200/auth/login?error=no_email"
            )
        
        # Check if user exists
        user_repo = get_user_repository()
        user = user_repo.get_by_email(email) or user_repo.get_by_google_id(google_id)
        
        if not user:
            # Create new user
            user_id = str(uuid.uuid4())
            user = User(
                user_id=user_id,
                email=email,
                name=name,
                picture=picture,
                google_id=google_id,
                created_at=datetime.now(),
                last_login=datetime.now(),
                is_active=True
            )
            user_repo.save(user)
        else:
            # Update existing user
            user.last_login = datetime.now()
            user.name = name or user.name
            user.picture = picture or user.picture
            user.google_id = google_id or user.google_id
            user_repo.save(user)
            user_repo.update_last_login(user.user_id)
        
        # Generate JWT token
        jwt_token = generate_jwt_token(
            user_id=user.user_id,
            email=user.email,
            name=user.name
        )
        
        # Redirect to frontend with token
        redirect_url = f"http://localhost:4200/auth/callback?token={jwt_token}&user_id={user.user_id}"
        return RedirectResponse(url=redirect_url)
        
    except Exception as e:
        print(f"Error in Google OAuth callback: {e}")
        return RedirectResponse(
            url=f"http://localhost:4200/auth/login?error=server_error"
        )


@router.get("/me", response_model=SuccessResponse)
async def get_current_user(request: Request):
    """
    Get current authenticated user information
    """
    from core.jwt_utils import extract_user_id_from_jwt
    from core.user_context import get_user_id_from_request
    
    # Try to get user from JWT token
    authorization = request.headers.get("Authorization")
    user_id = None
    
    if authorization:
        user_id = extract_user_id_from_jwt(request, authorization)
    
    if not user_id:
        user_id = get_user_id_from_request(request)
    
    if not user_id or user_id == "default":
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user_repo = get_user_repository()
    user = user_repo.get_by_user_id(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return SuccessResponse(data={
        "user_id": user.user_id,
        "email": user.email,
        "name": user.name,
        "picture": user.picture,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None
    })


@router.post("/logout")
async def logout():
    """
    Logout endpoint (client-side token removal)
    """
    return SuccessResponse(data={"message": "Logged out successfully"})


@router.get("/verify")
async def verify_token(request: Request):
    """
    Verify JWT token validity
    """
    from core.jwt_utils import extract_user_id_from_jwt
    from core.auth import verify_jwt_token
    
    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No token provided")
    
    token = authorization.replace("Bearer ", "").strip()
    payload = verify_jwt_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return SuccessResponse(data={
        "valid": True,
        "user_id": payload.get("user_id"),
        "email": payload.get("email")
    })

