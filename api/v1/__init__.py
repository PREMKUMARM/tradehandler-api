"""
API v1 routes
"""
from fastapi import APIRouter

# Create main API router
api_router = APIRouter(prefix="/api/v1", tags=["v1"])

# Import and register route modules
from .routes import agent
from . import health

# Register routers
api_router.include_router(agent.router)
api_router.include_router(health.router, prefix="/health", tags=["Health"])

