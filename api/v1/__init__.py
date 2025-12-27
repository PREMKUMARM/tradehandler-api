"""
API v1 routes
"""
from fastapi import APIRouter

# Create main API router
api_router = APIRouter(prefix="/api/v1", tags=["v1"])

# Import and register route modules
from .routes import agent, users, auth
from . import health

# Register routers
api_router.include_router(agent.router)
api_router.include_router(users.router)
api_router.include_router(auth.router)
api_router.include_router(health.router, prefix="/health", tags=["Health"])

