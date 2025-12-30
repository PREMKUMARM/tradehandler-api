"""
API routes for selected stocks management
"""
from fastapi import APIRouter, HTTPException, Request
from typing import List, Optional
from datetime import datetime
from core.responses import SuccessResponse, ErrorResponse
from core.dependencies import get_request_id
from database.stocks_repository import get_stocks_repository, SelectedStock
from agent.tools.instrument_resolver import resolve_instrument_name

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("")
async def get_selected_stocks(request: Request, active_only: bool = False):
    """Get all selected stocks"""
    request_id = get_request_id(request)
    
    try:
        repo = get_stocks_repository()
        stocks = repo.get_all(active_only=active_only)
        
        return SuccessResponse(
            data={
                "stocks": [stock.to_dict() for stock in stocks],
                "total": len(stocks)
            },
            request_id=request_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stocks: {str(e)}")


@router.post("")
async def add_stock(request: Request, stock_data: dict):
    """Add a new stock to selected stocks"""
    request_id = get_request_id(request)
    
    try:
        # Validate required fields
        tradingsymbol = stock_data.get("tradingsymbol")
        exchange = stock_data.get("exchange", "NSE")
        
        if not tradingsymbol:
            raise HTTPException(status_code=400, detail="tradingsymbol is required")
        
        # Resolve instrument if not provided
        instrument_token = stock_data.get("instrument_token")
        instrument_key = stock_data.get("instrument_key")
        name = stock_data.get("name")
        instrument_type = stock_data.get("instrument_type")
        
        if not instrument_key or not instrument_token:
            # Resolve instrument name
            instrument_info = resolve_instrument_name(tradingsymbol, exchange)
            if not instrument_info:
                raise HTTPException(
                    status_code=404,
                    detail=f"Instrument '{tradingsymbol}' not found in {exchange}"
                )
            
            instrument_token = instrument_info.get("instrument_token")
            instrument_key = f"{instrument_info.get('exchange', exchange)}:{instrument_info.get('tradingsymbol', tradingsymbol)}"
            name = instrument_info.get("name", tradingsymbol)
            instrument_type = instrument_info.get("instrument_type")
        
        # Create stock object
        stock = SelectedStock(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            instrument_token=instrument_token,
            instrument_key=instrument_key,
            name=name,
            instrument_type=instrument_type,
            is_active=stock_data.get("is_active", True),
            notes=stock_data.get("notes")
        )
        
        repo = get_stocks_repository()
        
        # Check if stock already exists before saving
        existing = repo.get_by_instrument_key(stock.instrument_key)
        if not existing:
            existing = repo.get_by_tradingsymbol_exchange(stock.tradingsymbol, stock.exchange)
        
        if existing:
            # Stock already exists, update it
            existing.tradingsymbol = stock.tradingsymbol
            existing.exchange = stock.exchange
            existing.instrument_token = stock.instrument_token
            existing.instrument_key = stock.instrument_key
            existing.name = stock.name
            existing.instrument_type = stock.instrument_type
            existing.is_active = stock.is_active
            existing.notes = stock.notes
            existing.updated_at = datetime.now()
            
            if repo.save(existing):
                return SuccessResponse(
                    data={"stock": existing.to_dict()},
                    message="Stock already exists and was updated",
                    request_id=request_id
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to update existing stock")
        
        # New stock - save it
        if repo.save(stock):
            return SuccessResponse(
                data={"stock": stock.to_dict()},
                message="Stock added successfully",
                request_id=request_id
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save stock. Please check server logs for details.")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding stock: {str(e)}")


@router.post("/bulk")
async def add_stocks_bulk(request: Request, stocks_data: List[dict]):
    """Add multiple stocks at once"""
    request_id = get_request_id(request)
    
    try:
        repo = get_stocks_repository()
        stocks = []
        
        for stock_data in stocks_data:
            tradingsymbol = stock_data.get("tradingsymbol")
            exchange = stock_data.get("exchange", "NSE")
            
            if not tradingsymbol:
                continue
            
            # Resolve instrument if needed
            instrument_token = stock_data.get("instrument_token")
            instrument_key = stock_data.get("instrument_key")
            name = stock_data.get("name")
            instrument_type = stock_data.get("instrument_type")
            
            if not instrument_key or not instrument_token:
                instrument_info = resolve_instrument_name(tradingsymbol, exchange)
                if instrument_info:
                    instrument_token = instrument_info.get("instrument_token")
                    instrument_key = f"{instrument_info.get('exchange', exchange)}:{instrument_info.get('tradingsymbol', tradingsymbol)}"
                    name = instrument_info.get("name", tradingsymbol)
                    instrument_type = instrument_info.get("instrument_type")
                else:
                    continue  # Skip if not found
            
            stock = SelectedStock(
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                instrument_token=instrument_token,
                instrument_key=instrument_key,
                name=name,
                instrument_type=instrument_type,
                is_active=stock_data.get("is_active", True),
                notes=stock_data.get("notes")
            )
            stocks.append(stock)
        
        saved_count = repo.bulk_save(stocks)
        
        return SuccessResponse(
            data={
                "saved": saved_count,
                "total": len(stocks_data)
            },
            message=f"Added {saved_count} stocks successfully",
            request_id=request_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding stocks: {str(e)}")


@router.delete("/{instrument_key:path}")
async def remove_stock(instrument_key: str, request: Request):
    """Remove a stock from selected stocks"""
    request_id = get_request_id(request)
    
    try:
        repo = get_stocks_repository()
        if repo.delete(instrument_key):
            return SuccessResponse(
                message="Stock removed successfully",
                request_id=request_id
            )
        else:
            raise HTTPException(status_code=404, detail="Stock not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing stock: {str(e)}")


@router.patch("/{instrument_key:path}/toggle")
async def toggle_stock_active(instrument_key: str, request: Request, is_active: bool):
    """Toggle active status of a stock"""
    request_id = get_request_id(request)
    
    try:
        repo = get_stocks_repository()
        if repo.toggle_active(instrument_key, is_active):
            return SuccessResponse(
                message=f"Stock {'activated' if is_active else 'deactivated'} successfully",
                request_id=request_id
            )
        else:
            raise HTTPException(status_code=404, detail="Stock not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error toggling stock status: {str(e)}")


@router.get("/search")
async def search_instruments(request: Request, query: str, exchange: str = "NSE", limit: int = 20):
    """Search for instruments by name or symbol (uses cached CSV)"""
    request_id = get_request_id(request)
    
    try:
        from utils.instruments_cache import search_instruments
        
        matches = search_instruments(query, exchange, limit)
        
        return SuccessResponse(
            data={
                "instruments": matches,
                "total": len(matches)
            },
            request_id=request_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching instruments: {str(e)}")


@router.post("/cache/refresh")
async def refresh_instruments_cache(request: Request):
    """Manually refresh the instruments cache"""
    request_id = get_request_id(request)
    
    try:
        from utils.instruments_cache import refresh_cache
        
        result = refresh_cache()
        
        return SuccessResponse(
            data=result,
            message="Instruments cache refreshed successfully" if result.get("success") else "Failed to refresh cache",
            request_id=request_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing cache: {str(e)}")


@router.get("/cache/info")
async def get_cache_info(request: Request):
    """Get information about the instruments cache"""
    request_id = get_request_id(request)
    
    try:
        from utils.instruments_cache import get_cache_info
        
        info = get_cache_info()
        
        return SuccessResponse(
            data=info,
            request_id=request_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache info: {str(e)}")

