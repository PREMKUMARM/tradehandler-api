"""
User management API endpoints
"""
from fastapi import APIRouter, Request, HTTPException, Header
from typing import Optional, List, Dict, Any
from core.responses import SuccessResponse, ErrorResponse
from core.user_context import get_user_id_from_request
from core.jwt_utils import extract_user_id_from_jwt
from database.repositories import get_config_repository
from database.connection import get_database
import sqlite3

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/current", response_model=SuccessResponse)
async def get_current_user(request: Request):
    """Get current logged-in user information"""
    try:
        # Extract user ID from JWT token or header
        user_id = get_user_id_from_request(request)
        
        # Try to get additional user info from database if available
        # For now, just return the user_id
        # In a full implementation, you'd have a users table
        
        user_info = {
            "user_id": user_id,
            "username": user_id,  # Default to user_id as username
            "email": None,
            "created_at": None
        }
        
        # Try to get user details from a users table if it exists
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT username, email, created_at FROM users WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                user_info["username"] = row.get("username") or user_id
                user_info["email"] = row.get("email")
                user_info["created_at"] = row.get("created_at")
        except (sqlite3.OperationalError, AttributeError):
            # Users table doesn't exist yet - that's okay
            pass
        
        return SuccessResponse(data=user_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting current user: {str(e)}")


@router.post("/switch", response_model=SuccessResponse)
async def switch_user(
    request: Request,
    user_id: str
):
    """
    Switch to a different user.
    This is mainly for frontend use - the backend will use the user_id from the request.
    """
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required")
    
    user_id = user_id.strip()
    
    # Verify user exists (check if they have any configs)
    config_repo = get_config_repository()
    user_configs = config_repo.get_all(user_id=user_id)
    
    user_info = {
        "user_id": user_id,
        "username": user_id,
        "has_config": len(user_configs) > 0,
        "config_count": len(user_configs)
    }
    
    return SuccessResponse(
        data=user_info,
        message=f"Switched to user: {user_id}"
    )


@router.get("/list", response_model=SuccessResponse)
async def list_users(request: Request):
    """
    List all users who have configurations.
    Returns list of user IDs that have saved configs.
    """
    try:
        db = get_database()
        
        # Get distinct user_ids from agent_config table
        cursor = db.execute_query(
            "SELECT DISTINCT user_id, COUNT(*) as config_count, MAX(updated_at) as last_updated "
            "FROM agent_config "
            "GROUP BY user_id "
            "ORDER BY last_updated DESC"
        )
        
        users = []
        for row in cursor.fetchall():
            users.append({
                "user_id": row["user_id"],
                "config_count": row["config_count"],
                "last_updated": row["last_updated"]
            })
        
        return SuccessResponse(data={"users": users, "total": len(users)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing users: {str(e)}")


@router.get("/{user_id}/info", response_model=SuccessResponse)
async def get_user_info(user_id: str, request: Request):
    """Get information about a specific user"""
    try:
        config_repo = get_config_repository()
        user_configs = config_repo.get_all(user_id=user_id)
        
        # Count configs by category
        categories = {}
        for config in user_configs:
            cat = config.category
            categories[cat] = categories.get(cat, 0) + 1
        
        user_info = {
            "user_id": user_id,
            "username": user_id,  # Default
            "config_count": len(user_configs),
            "categories": categories,
            "has_config": len(user_configs) > 0
        }
        
        # Try to get additional info from users table
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT username, email, created_at FROM users WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                user_info["username"] = row.get("username") or user_id
                user_info["email"] = row.get("email")
                user_info["created_at"] = row.get("created_at")
        except (sqlite3.OperationalError, AttributeError):
            pass
        
        return SuccessResponse(data=user_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting user info: {str(e)}")

