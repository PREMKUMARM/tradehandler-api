"""
Instruments cache manager - Downloads and caches Kite instruments CSV
"""
import os
import gzip
import csv
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import threading

from utils.kite_utils import get_kite_instance


# Cache directory
CACHE_DIR = Path("data/instruments")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILE = CACHE_DIR / "instruments.csv"
LAST_UPDATE_FILE = CACHE_DIR / "last_update.txt"
LOCK = threading.Lock()

# Cache expiry (24 hours)
CACHE_EXPIRY_HOURS = 24


def get_instruments_csv_path() -> Path:
    """Get the path to the cached instruments CSV"""
    return CSV_FILE


def get_last_update_time() -> Optional[datetime]:
    """Get the last update time from cache"""
    if LAST_UPDATE_FILE.exists():
        try:
            with open(LAST_UPDATE_FILE, 'r') as f:
                timestamp_str = f.read().strip()
                return datetime.fromisoformat(timestamp_str)
        except Exception:
            return None
    return None


def is_cache_valid() -> bool:
    """Check if cache is still valid (not expired)"""
    last_update = get_last_update_time()
    if not last_update:
        return False
    
    if not CSV_FILE.exists():
        return False
    
    # Check if cache is expired
    age = datetime.now() - last_update
    return age < timedelta(hours=CACHE_EXPIRY_HOURS)


def download_instruments_csv() -> bool:
    """
    Download instruments CSV from Kite API and save to cache
    
    Returns:
        True if successful, False otherwise
    """
    try:
        kite = get_kite_instance()
        
        # Get instruments for all exchanges
        # Kite API returns a list, but we'll use the CSV endpoint if available
        # For now, we'll use the instruments() method which returns a list
        # and convert it to CSV format
        
        print("[Instruments Cache] Downloading instruments from Kite API...")
        
        all_instruments = []
        exchanges = ["NSE", "NFO", "BSE", "MCX"]
        
        for exchange in exchanges:
            try:
                instruments = kite.instruments(exchange)
                all_instruments.extend(instruments)
                print(f"[Instruments Cache] Downloaded {len(instruments)} instruments from {exchange}")
            except Exception as e:
                print(f"[Instruments Cache] Error downloading from {exchange}: {e}")
                continue
        
        if not all_instruments:
            print("[Instruments Cache] No instruments downloaded")
            return False
        
        # Write to CSV file
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            if all_instruments:
                # Get fieldnames from first instrument
                fieldnames = list(all_instruments[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_instruments)
        
        # Update last update time
        with open(LAST_UPDATE_FILE, 'w') as f:
            f.write(datetime.now().isoformat())
        
        print(f"[Instruments Cache] Successfully cached {len(all_instruments)} instruments to {CSV_FILE}")
        return True
        
    except Exception as e:
        print(f"[Instruments Cache] Error downloading instruments: {e}")
        return False


def ensure_cache_valid() -> bool:
    """
    Ensure cache is valid, download if needed
    
    Returns:
        True if cache is valid or successfully downloaded
    """
    with LOCK:
        if is_cache_valid():
            return True
        
        # Cache expired or missing, download fresh
        return download_instruments_csv()


def load_instruments_from_csv(exchange: Optional[str] = None) -> List[Dict]:
    """
    Load instruments from cached CSV file
    
    Args:
        exchange: Optional exchange filter (NSE, NFO, BSE, MCX)
        
    Returns:
        List of instrument dictionaries
    """
    if not CSV_FILE.exists():
        # Try to download if file doesn't exist
        ensure_cache_valid()
    
    if not CSV_FILE.exists():
        return []
    
    instruments = []
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if exchange:
                    # Filter by exchange if specified
                    if row.get('exchange', '').upper() == exchange.upper():
                        instruments.append(row)
                else:
                    instruments.append(row)
    except Exception as e:
        print(f"[Instruments Cache] Error reading CSV: {e}")
        return []
    
    return instruments


def search_instruments(query: str, exchange: str = "NSE", limit: int = 20) -> List[Dict]:
    """
    Search instruments from cached CSV
    
    Args:
        query: Search query (symbol or name)
        exchange: Exchange to search in
        limit: Maximum number of results
        
    Returns:
        List of matching instruments
    """
    # Ensure cache is valid
    ensure_cache_valid()
    
    # Load instruments
    instruments = load_instruments_from_csv(exchange)
    
    if not instruments:
        return []
    
    query_upper = query.upper().strip()
    matches = []
    
    for inst in instruments:
        symbol = inst.get("tradingsymbol", "").upper()
        name = inst.get("name", "").upper()
        
        if query_upper in symbol or query_upper in name:
            # Format response similar to API
            match = {
                "tradingsymbol": inst.get("tradingsymbol"),
                "exchange": inst.get("exchange", exchange),
                "instrument_token": int(inst.get("instrument_token", 0)) if inst.get("instrument_token") else 0,
                "instrument_key": f"{inst.get('exchange', exchange)}:{inst.get('tradingsymbol')}",
                "name": inst.get("name"),
                "instrument_type": inst.get("instrument_type")
            }
            matches.append(match)
            
            if len(matches) >= limit:
                break
    
    return matches


def refresh_cache() -> Dict[str, any]:
    """
    Manually refresh the instruments cache
    
    Returns:
        Dictionary with refresh status
    """
    with LOCK:
        success = download_instruments_csv()
        last_update = get_last_update_time()
        
        return {
            "success": success,
            "last_update": last_update.isoformat() if last_update else None,
            "cache_file": str(CSV_FILE),
            "cache_exists": CSV_FILE.exists()
        }


def get_cache_info() -> Dict[str, any]:
    """
    Get information about the cache
    
    Returns:
        Dictionary with cache information
    """
    last_update = get_last_update_time()
    cache_valid = is_cache_valid()
    cache_exists = CSV_FILE.exists()
    
    info = {
        "cache_file": str(CSV_FILE),
        "cache_exists": cache_exists,
        "cache_valid": cache_valid,
        "last_update": last_update.isoformat() if last_update else None
    }
    
    if cache_exists:
        try:
            instruments = load_instruments_from_csv()
            info["total_instruments"] = len(instruments)
            # Count by exchange
            exchange_counts = {}
            for inst in instruments:
                exch = inst.get("exchange", "UNKNOWN")
                exchange_counts[exch] = exchange_counts.get(exch, 0) + 1
            info["instruments_by_exchange"] = exchange_counts
        except Exception as e:
            info["error"] = str(e)
    
    return info

