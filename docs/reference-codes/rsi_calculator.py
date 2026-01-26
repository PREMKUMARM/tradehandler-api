import pandas as pd
from ta.momentum import RSIIndicator

def compute_rsi(closes: list, window: int = 14) -> float:
    df = pd.DataFrame({"close": closes})
    rsi = RSIIndicator(df["close"], window=window).rsi()
    return round(rsi.iloc[-1], 2)