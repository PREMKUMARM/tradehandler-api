"""
API v1 routes
"""
from fastapi import APIRouter
from api.v1.routes import agent, users, auth, strategies, market, orders, portfolio, simulation, stocks, trading, websocket, trade_limits, monitoring, multi_agent, agent_multi, telegram, telegram_scheduler

api_router = APIRouter()

api_router.include_router(agent.router)
api_router.include_router(users.router)
api_router.include_router(auth.router)
api_router.include_router(stocks.router)
api_router.include_router(multi_agent.router)
api_router.include_router(agent_multi.router)
api_router.include_router(telegram.router)
api_router.include_router(telegram_scheduler.router)
