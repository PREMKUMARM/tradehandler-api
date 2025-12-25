from typing import Union
import os
from pathlib import Path

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import json
from pprint import pprint
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

# Kite utilities
from utils.kite_utils import (
    api_key,
    get_access_token,
    get_kite_instance,
    calculate_trend_and_suggestions
)

# Agent imports
from agent.graph import run_agent, get_agent_instance, get_agent_memory
from agent.approval import get_approval_queue
from agent.safety import get_safety_manager
from agent.config import get_agent_config

# Initialize FastAPI app
app = FastAPI()

# Kite Connect credentials - Update these with your actual API key and secret
# IMPORTANT: The redirect_uri must EXACTLY match what's configured in your Kite Connect app settings
# api_secret = os.getenv('KITE_API_SECRET', 'your_secret_here')
api_secret = os.getenv('KITE_API_SECRET', 'vmrsky50fsozxonx2v5wwjwdmm6jcjtk')
# For local development, use: http://localhost:4200/auth-token
# For production, use: https://www.tradehandler.com/auth-token
redirect_uri = os.getenv('KITE_REDIRECT_URI', 'http://localhost:4200/auth-token')

origins = [
    "https://www.tradehandler.com",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/live-positions")
def get_live_positions():
    """Fetch current open positions from Zerodha Kite"""
    try:
        kite = get_kite_instance()
        positions = kite.positions()
        
        # Calculate live MTM and totals
        net_positions = positions.get("net", [])
        total_pnl = 0
        active_count = 0
        
        for pos in net_positions:
            total_pnl += pos.get("pnl", 0)
            if pos.get("quantity", 0) != 0:
                active_count += 1
                
        return {
            "data": {
                "positions": net_positions,
                "total_pnl": round(total_pnl, 2),
                "active_count": active_count
            }
        }
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching positions: {str(e)}")

# ... rest of main.py continues here ...

