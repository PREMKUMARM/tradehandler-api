from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import asyncio
from binance_client import fetch_klines
from rsi_calculator import compute_rsi

app = FastAPI()
SYMBOLS = ["1000PEPEUSDT","BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "MATICUSDT"]
TIMEFRAMES = ["1m", "5m", "15m", "30m"]
latest_rsi_data = {}

async def update_rsi_data():
    while True:
        for symbol in SYMBOLS:
            latest_rsi_data[symbol] = {}
            for tf in TIMEFRAMES:
                try:
                    closes = await fetch_klines(symbol, tf)
                    rsi = compute_rsi(closes)
                    latest_rsi_data[symbol][tf] = rsi
                except Exception as e:
                    print(f"Error fetching data for {symbol} {tf}: {e}")
                    latest_rsi_data[symbol][tf] = None
        await asyncio.sleep(5)  # update every 10s

@app.on_event("startup")
async def start_bg_tasks():
    asyncio.create_task(update_rsi_data())

@app.get("/rsi/{symbol}")
async def get_rsi(symbol: str):
    symbol = symbol.upper()
    if symbol in latest_rsi_data:
        return latest_rsi_data[symbol]
    return JSONResponse(content={"error": "Symbol not found"}, status_code=404)

@app.websocket("/ws/rsi")
async def rsi_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(latest_rsi_data)
            #await print("Sent RSI data to WebSocket clients", latest_rsi_data)
            await asyncio.sleep(5)
    except Exception:
        await websocket.close()
