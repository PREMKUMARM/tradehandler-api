import os
import httpx
from dotenv import load_dotenv

load_dotenv()
BASE_URL = "https://fapi.binance.com"  # Mainnet futures API

async def fetch_klines(symbol: str, interval: str, limit: int = 100):
    url = f"{BASE_URL}/fapi/v1/klines"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params={"symbol": symbol, "interval": interval, "limit": limit})
        data = resp.json()
        closes = [float(k[4]) for k in data]
        return closes
