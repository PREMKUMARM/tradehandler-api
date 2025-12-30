"""
Instrument name resolution utilities
"""
from typing import Optional, Dict, Any, Union, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.kite_utils import get_kite_instance
from kiteconnect.exceptions import KiteException


# Predefined groups of stocks
TOP_10_NIFTY50 = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "ITC", 
    "LT", "TCS", "AXISBANK", "KOTAKBANK", "BHARTIARTL"
]

def get_selected_stocks_group() -> List[str]:
    """Get list of selected stocks from database"""
    try:
        from database.stocks_repository import get_stocks_repository
        repo = get_stocks_repository()
        stocks = repo.get_all(active_only=True)
        return [stock.tradingsymbol for stock in stocks]
    except Exception as e:
        print(f"[Instrument Resolver] Error loading selected stocks: {e}")
        return []


# Initialize instrument groups with selected stocks
_selected_stocks_cache: Optional[List[str]] = None
_selected_stocks_cache_time: Optional[float] = None
CACHE_TTL = 300  # 5 minutes cache

def get_selected_stocks_cached() -> List[str]:
    """Get selected stocks with caching"""
    global _selected_stocks_cache, _selected_stocks_cache_time
    import time
    
    current_time = time.time()
    if _selected_stocks_cache is None or _selected_stocks_cache_time is None or (current_time - _selected_stocks_cache_time) > CACHE_TTL:
        _selected_stocks_cache = get_selected_stocks_group()
        _selected_stocks_cache_time = current_time
    
    return _selected_stocks_cache


INSTRUMENT_GROUPS = {
    "top 10 nifty50 stocks": TOP_10_NIFTY50,
    "top 10 nifty50": TOP_10_NIFTY50,
    "top 10 nifty": TOP_10_NIFTY50,
    "nifty 10": TOP_10_NIFTY50,
    "nifty top 10": TOP_10_NIFTY50,
    "nifty top 10 stocks": TOP_10_NIFTY50,
    # Dynamic groups from selected stocks
    "selected stocks": lambda: get_selected_stocks_cached(),
    "my stocks": lambda: get_selected_stocks_cached(),
    "watchlist": lambda: get_selected_stocks_cached(),
    "selected": lambda: get_selected_stocks_cached(),
}

# Common instrument name mappings
INSTRUMENT_ALIASES = {
    "reliance": "RELIANCE",
    "reliance industries": "RELIANCE",
    "ril": "RELIANCE",
    "nifty": "NIFTY 50",
    "nifty50": "NIFTY 50",
    "nifty 50": "NIFTY 50",
    "banknifty": "NIFTY BANK",
    "nifty bank": "NIFTY BANK",
    "tcs": "TCS",
    "infy": "INFY",
    "infosys": "INFY",
    "hdfc": "HDFC",
    "hdfc bank": "HDFC",
    "icici": "ICICIBANK",
    "icici bank": "ICICIBANK",
    "sbi": "SBIN",
    "state bank": "SBIN",
    "wipro": "WIPRO",
    "lt": "LT",
    "larsen": "LT",
    "ltim": "LTIM",
    "lti": "LTIM",
    "bharti": "BHARTIARTL",
    "airtel": "BHARTIARTL",
    "hcl": "HCLTECH",
    "hcl tech": "HCLTECH",
    "asian paints": "ASIANPAINT",
    "asianpaints": "ASIANPAINT",
    "maruti": "MARUTI",
    "maruti suzuki": "MARUTI",
    "titan": "TITAN",
    "ultracemco": "ULTRACEMCO",
    "ultra tech": "ULTRACEMCO",
    "nestle": "NESTLEIND",
    "nestle india": "NESTLEIND",
}


# Global cache for instruments to avoid rate limits
_instruments_cache: Dict[str, List[Dict[str, Any]]] = {}

def resolve_instrument_name(instrument_name: str, exchange: str = "NSE", return_multiple: bool = False) -> Union[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Resolve instrument name to instrument token and details.
    """
    global _instruments_cache
    try:
        kite = get_kite_instance()
        
        # Normalize instrument name
        name_upper = instrument_name.upper().strip()
        
        # Check aliases first
        if name_upper.lower() in INSTRUMENT_ALIASES:
            name_upper = INSTRUMENT_ALIASES[name_upper.lower()]
        
        # USE CACHE: Fetch instruments only if not already cached for this exchange
        if exchange not in _instruments_cache:
            print(f"[DEBUG] Fetching full instrument list for {exchange} from Kite (caching for performance)...")
            _instruments_cache[exchange] = kite.instruments(exchange)
        
        instruments = _instruments_cache[exchange]
        
        matches = []
        
        # Try exact match first
        for inst in instruments:
            symbol = inst.get("tradingsymbol", "").upper()
            name = inst.get("name", "").upper()
            
            if symbol == name_upper or name == name_upper:
                match = {
                    "instrument_token": inst.get("instrument_token"),
                    "tradingsymbol": inst.get("tradingsymbol"),
                    "exchange": inst.get("exchange", exchange),
                    "name": inst.get("name"),
                    "instrument_type": inst.get("instrument_type"),
                }
                if not return_multiple:
                    return match
                matches.append(match)
        
        # Try partial match if no exact match found or if we want multiple
        if not matches or return_multiple:
            for inst in instruments:
                symbol = inst.get("tradingsymbol", "").upper()
                name = inst.get("name", "").upper()
                
                # If we already added this as an exact match, skip it
                if any(m["tradingsymbol"] == inst.get("tradingsymbol") for m in matches):
                    continue
                    
                if name_upper in symbol or name_upper in name:
                    match = {
                        "instrument_token": inst.get("instrument_token"),
                        "tradingsymbol": inst.get("tradingsymbol"),
                        "exchange": inst.get("exchange", exchange),
                        "name": inst.get("name"),
                        "instrument_type": inst.get("instrument_type"),
                    }
                    if not return_multiple:
                        return match
                    matches.append(match)
        
        if return_multiple:
            # Sort matches to prioritize:
            # 1. Exact symbol match
            # 2. Symbol starts with name_upper
            # 3. Shorter symbol length
            def sort_key(x):
                sym = x["tradingsymbol"].upper()
                if sym == name_upper:
                    return (0, len(sym))
                if sym.startswith(name_upper):
                    return (1, len(sym))
                return (2, len(sym))
                
            matches.sort(key=sort_key)
            return matches
            
        return matches[0] if matches else None
        
    except Exception as e:
        print(f"Error resolving instrument {instrument_name}: {e}")
        return [] if return_multiple else None


def get_instrument_token(instrument_name: str, exchange: str = "NSE") -> Optional[int]:
    """
    Get instrument token for a given instrument name.

    Args:
        instrument_name: Instrument name
        exchange: Exchange

    Returns:
        Instrument token or None
    """
    result = resolve_instrument_name(instrument_name, exchange)
    return result.get("instrument_token") if result else None


def bulk_resolve_instruments(instrument_names: List[str], exchange: str = "NSE") -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Bulk resolve multiple instrument names at once to avoid rate limits.

    Args:
        instrument_names: List of instrument names to resolve
        exchange: Exchange to search in

    Returns:
        Dictionary mapping instrument names to resolved data (or None if not found)
    """
    global _instruments_cache
    try:
        kite = get_kite_instance()

        # USE CACHE: Fetch instruments only if not already cached for this exchange
        if exchange not in _instruments_cache:
            print(f"[DEBUG] Fetching full instrument list for {exchange} from Kite (caching for performance)...")
            _instruments_cache[exchange] = kite.instruments(exchange)

        instruments = _instruments_cache[exchange]
        results = {}

        # Process all instruments at once
        for inst_name in instrument_names:
            name_upper = inst_name.upper().strip()

            # Check aliases first
            if name_upper.lower() in INSTRUMENT_ALIASES:
                name_upper = INSTRUMENT_ALIASES[name_upper.lower()]

            # Search for this instrument
            for inst in instruments:
                symbol = inst.get("tradingsymbol", "").upper()
                name = inst.get("name", "").upper()

                if symbol == name_upper or name == name_upper:
                    results[inst_name] = {
                        "instrument_token": inst.get("instrument_token"),
                        "tradingsymbol": inst.get("tradingsymbol"),
                        "exchange": inst.get("exchange", exchange),
                        "name": inst.get("name"),
                        "instrument_type": inst.get("instrument_type"),
                    }
                    break
            else:
                # No exact match found
                results[inst_name] = None

        return results

    except Exception as e:
        print(f"Error bulk resolving instruments: {e}")
        # Return None for all instruments on error
        return {name: None for name in instrument_names}

