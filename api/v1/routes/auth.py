"""
Authentication API endpoints for Google OAuth and Kite Connect
"""
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from typing import Optional
import os
import uuid
from datetime import datetime
from urllib.parse import urlencode
from pathlib import Path

from core.responses import SuccessResponse, ErrorResponse
from core.auth import generate_jwt_token
from core.config import get_settings
from core.user_context import get_user_id_from_request
from database.user_repository import get_user_repository
from database.models import User
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException
from utils.kite_utils import get_kite_api_key, get_access_token

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


# ============================================================================
# Kite Connect Authentication Endpoints
# ============================================================================

# Get Kite Connect configuration
def get_kite_redirect_uri():
    """Get Kite Connect redirect URI from environment or default"""
    return os.getenv('KITE_REDIRECT_URI', 'http://localhost:4200/auth-token')

def get_kite_api_secret(user_id: str = "default"):
    """Get Kite API secret from user config or environment"""
    try:
        from agent.user_config import get_user_config
        config = get_user_config(user_id=user_id)
        if config.kite_api_secret:
            return config.kite_api_secret
    except:
        pass
    try:
        from agent.config import get_agent_config
        config = get_agent_config()
        if config.kite_api_secret:
            return config.kite_api_secret
    except:
        pass
    return os.getenv('KITE_API_SECRET', '')


@router.get("/kite/login")
async def kite_login(request: Request):
    """
    Get Kite Connect login URL
    """
    try:
        user_id = get_user_id_from_request(request)
        current_api_key = get_kite_api_key(user_id)
        redirect_uri = get_kite_redirect_uri()
        
        if current_api_key == 'your_api_key_here' or not current_api_key:
            raise HTTPException(
                status_code=500, 
                detail="KITE_API_KEY is not configured. Please set it in the Configuration page or environment variables"
            )
        
        kite = KiteConnect(api_key=current_api_key)
        login_url = kite.login_url()
        print(f"Generated login URL with redirect_uri: {redirect_uri}")
        return {
            "login_url": login_url,
            "message": "Redirect user to this URL for authentication",
            "redirect_uri": redirect_uri,
            "note": f"Make sure the redirect URI in your Kite Connect app settings matches: {redirect_uri}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating login URL: {str(e)}")


@router.post("/kite/set-token")
async def kite_set_token(request: Request):
    """
    Store Kite Connect access token after authentication
    """
    try:
        data = await request.json()
        request_token = data.get('request_token')
        access_token_from_request = data.get('access-token')
        
        user_id = get_user_id_from_request(request)
        redirect_uri = get_kite_redirect_uri()
        
        print(f"Received request_token: {request_token[:20] if request_token else None}...")
        print(f"Received access-token from request: {access_token_from_request[:20] if access_token_from_request else None}...")
        print(f"User ID: {user_id}")
        print(f"Redirect URI configured: {redirect_uri}")
        
        if not request_token and not access_token_from_request:
            raise HTTPException(status_code=400, detail="Either request_token or access-token required")
        
        access_token = None
        
        if request_token:
            current_api_key = get_kite_api_key(user_id)
            current_api_secret = get_kite_api_secret(user_id)
            
            print(f"API Key configured: {current_api_key[:10] if current_api_key and len(current_api_key) > 10 else 'NOT SET'}...")
            
            if current_api_key == 'your_api_key_here' or not current_api_key:
                raise HTTPException(
                    status_code=500, 
                    detail="KITE_API_KEY is not configured. Please set it in the Configuration page or environment variables"
                )
            if current_api_secret == 'your_api_secret_here' or not current_api_secret:
                raise HTTPException(
                    status_code=500, 
                    detail="KITE_API_SECRET is not configured. Please set it in the Configuration page or environment variables"
                )
            
            kite = KiteConnect(api_key=current_api_key)
            try:
                print(f"Attempting to generate session with request_token...")
                data_response = kite.generate_session(request_token, api_secret=current_api_secret)
                
                if isinstance(data_response, dict):
                    access_token = data_response.get('access_token')
                else:
                    access_token = getattr(data_response, 'access_token', None) if hasattr(data_response, 'access_token') else None
                
                if access_token and access_token == request_token:
                    raise HTTPException(
                        status_code=400,
                        detail="Token exchange failed: Received request_token instead of access_token. "
                               "This usually means: 1) API key/secret mismatch, "
                               "2) Redirect URI mismatch, or 3) Request token expired. "
                               f"Please check your Kite Connect app settings. Redirect URI should be: {redirect_uri}"
                    )
                
                if not access_token:
                    raise HTTPException(
                        status_code=400, 
                        detail="Failed to get access_token from Kite. "
                               "Please check: 1) API key and secret are correct, "
                               f"2) Redirect URI matches exactly: {redirect_uri}, "
                               "3) Request token is fresh (they expire quickly)."
                    )
            except KiteException as e:
                error_msg = str(e)
                if "invalid" in error_msg.lower() or "expired" in error_msg.lower():
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid request token: {error_msg}. "
                               f"Please ensure: 1) Redirect URI in Kite Connect app settings matches exactly '{redirect_uri}', "
                               f"2) Request token is used immediately (they expire quickly), "
                               f"3) API key and secret are correct."
                    )
                raise HTTPException(status_code=400, detail=f"Kite API error: {error_msg}")
        else:
            access_token = access_token_from_request
        
        if not access_token:
            raise HTTPException(status_code=400, detail="Failed to obtain access token")
        
        if request_token and access_token == request_token:
            raise HTTPException(
                status_code=400,
                detail="Internal error: Access token matches request token. "
                       "The token exchange may have failed. Please try again with a fresh request_token."
            )
        
        if len(access_token) < 20:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid access token (too short: {len(access_token)} chars). "
                       "Kite access tokens should be at least 20 characters. "
                       "Please check: 1) Your Kite API Key and Secret are correct, "
                       f"2) The redirect URI matches exactly: {redirect_uri}, "
                       "3) The request_token is fresh (they expire quickly)."
            )
        
        # Store access token
        config_path = Path("config")
        config_path.mkdir(exist_ok=True)
        
        with open("config/access_token.txt", "w") as f:
            f.write(access_token.strip())
        
        print(f"Access token stored successfully (length: {len(access_token)})")
        return {
            "status": "success", 
            "message": "Access token stored successfully",
            "access_token": access_token[:20] + "..." if access_token else None
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting token: {str(e)}")


@router.get("/kite/access-token")
async def kite_get_access_token(request: Request):
    """
    Get stored Kite Connect access token and validate it
    """
    token = get_access_token()
    if token:
        token_info = {
            "length": len(token),
            "preview": token[:20] + "..." if len(token) > 20 else token,
            "is_valid_length": len(token) >= 20,
            "status": "unknown"
        }
        
        if len(token) >= 20:
            try:
                user_id = get_user_id_from_request(request)
                api_key = get_kite_api_key(user_id)
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(token)
                kite.profile()
                token_info["status"] = "valid"
                token_info["api_key_used"] = api_key[:10] + "..." if api_key and len(api_key) > 10 else "NOT SET"
                return {
                    "access_token": token[:20] + "...",
                    "token_info": token_info,
                    "is_valid": True
                }
            except Exception as e:
                token_info["status"] = "invalid"
                token_info["error"] = str(e)
                return {
                    "access_token": token[:20] + "...",
                    "token_info": token_info,
                    "is_valid": False,
                    "message": f"Token exists but is invalid: {str(e)}"
                }
        else:
            token_info["status"] = "too_short"
            return {
                "access_token": token[:20] + "...",
                "token_info": token_info,
                "is_valid": False,
                "message": "Token is too short. Please regenerate using /api/v1/auth/kite/login and /api/v1/auth/kite/set-token"
            }
    
    return {
        "access_token": None, 
        "message": "No access token found. Please generate one using /api/v1/auth/kite/login and /api/v1/auth/kite/set-token",
        "token_info": None,
        "is_valid": False
    }


@router.delete("/kite/access-token")
async def kite_delete_access_token():
    """
    Delete the stored Kite Connect access token
    """
    try:
        token_path = Path("config/access_token.txt")
        if token_path.exists():
            token_path.unlink()
            return {"status": "success", "message": "Access token deleted successfully"}
        return {"status": "success", "message": "No access token file found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting token: {str(e)}")

