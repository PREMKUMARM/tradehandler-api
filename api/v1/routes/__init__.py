"""
API v1 routes
"""
from fastapi import APIRouter
from api.v1.routes import agent, users, auth

api_router = APIRouter()

api_router.include_router(agent.router)
api_router.include_router(users.router)
api_router.include_router(auth.router)
