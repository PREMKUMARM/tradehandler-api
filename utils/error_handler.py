"""
Comprehensive error handling and validation utilities
"""
import traceback
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from core.exceptions import (
    AlgoFeastException, ValidationError as AlgoValidationError,
    AuthenticationError, NotFoundError, BusinessLogicError, ExternalAPIError
)
from utils.logger import log_error, log_warning, log_info


class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_details: List[Dict] = []
        self.max_error_history = 1000
    
    def handle_exception(self, exception: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle and format exceptions consistently"""
        error_id = f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(exception)}"
        
        # Count error types
        error_type = type(exception).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Create error detail
        error_detail = {
            "error_id": error_id,
            "error_type": error_type,
            "message": str(exception),
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
            "traceback": traceback.format_exc() if isinstance(exception, Exception) else None
        }
        
        # Store error detail
        self.error_details.append(error_detail)
        if len(self.error_details) > self.max_error_history:
            self.error_details = self.error_details[-self.max_error_history:]
        
        # Log the error
        self._log_error(exception, error_detail)
        
        # Return formatted error response
        return self._format_error_response(exception, error_id)
    
    def _log_error(self, exception: Exception, error_detail: Dict):
        """Log error with appropriate level"""
        if isinstance(exception, (AuthenticationError, AlgoValidationError)):
            log_warning(f"Validation/Auth Error: {error_detail}")
        elif isinstance(exception, ExternalAPIError):
            log_error(f"External API Error: {error_detail}")
        elif isinstance(exception, BusinessLogicError):
            log_warning(f"Business Logic Error: {error_detail}")
        elif isinstance(exception, AlgoFeastException):
            log_error(f"AlgoFeast Error: {error_detail}")
        else:
            log_error(f"Unexpected Error: {error_detail}")
    
    def _format_error_response(self, exception: Exception, error_id: str) -> Dict[str, Any]:
        """Format error response based on exception type"""
        if isinstance(exception, AuthenticationError):
            return {
                "error": "authentication_error",
                "message": exception.message,
                "error_id": error_id,
                "status_code": status.HTTP_401_UNAUTHORIZED
            }
        
        elif isinstance(exception, AlgoValidationError):
            return {
                "error": "validation_error",
                "message": exception.message,
                "field": getattr(exception, 'field', None),
                "error_id": error_id,
                "status_code": status.HTTP_400_BAD_REQUEST
            }
        
        elif isinstance(exception, NotFoundError):
            return {
                "error": "not_found",
                "message": exception.message,
                "error_id": error_id,
                "status_code": status.HTTP_404_NOT_FOUND
            }
        
        elif isinstance(exception, BusinessLogicError):
            return {
                "error": "business_logic_error",
                "message": exception.message,
                "error_code": getattr(exception, 'error_code', None),
                "error_id": error_id,
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY
            }
        
        elif isinstance(exception, ExternalAPIError):
            return {
                "error": "external_api_error",
                "message": exception.message,
                "service": getattr(exception, 'service', None),
                "error_id": error_id,
                "status_code": status.HTTP_502_BAD_GATEWAY
            }
        
        elif isinstance(exception, ValidationError):
            return {
                "error": "validation_error",
                "message": "Validation failed",
                "details": exception.errors(),
                "error_id": error_id,
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY
            }
        
        elif isinstance(exception, HTTPException):
            return {
                "error": "http_error",
                "message": exception.detail,
                "error_id": error_id,
                "status_code": exception.status_code
            }
        
        else:
            return {
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "error_id": error_id,
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR
            }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "error_counts": self.error_counts,
            "total_errors": sum(self.error_counts.values()),
            "recent_errors": len([e for e in self.error_details 
                               if (datetime.now() - datetime.fromisoformat(e['timestamp'])).seconds < 3600]),
            "error_rate": len(self.error_details) / max(1, len(self.error_details))
        }
    
    def get_recent_errors(self, limit: int = 50) -> List[Dict]:
        """Get recent errors"""
        return self.error_details[-limit:]


# Global error handler instance
error_handler = ErrorHandler()


def validate_trading_symbol(symbol: str) -> bool:
    """Validate trading symbol format"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation - should be alphanumeric with possible special characters
    import re
    pattern = r'^[A-Z0-9\-&\s]+$'
    return bool(re.match(pattern, symbol.upper()))


def validate_quantity(quantity: int, min_qty: int = 1, max_qty: int = 10000) -> bool:
    """Validate order quantity"""
    return isinstance(quantity, int) and min_qty <= quantity <= max_qty


def validate_price(price: float, min_price: float = 0.01) -> bool:
    """Validate price"""
    return isinstance(price, (int, float)) and price >= min_price


def validate_order_type(order_type: str) -> bool:
    """Validate order type"""
    valid_types = ['MARKET', 'LIMIT', 'SL', 'SL-M']
    return order_type.upper() in valid_types


def validate_product_type(product: str) -> bool:
    """Validate product type"""
    valid_products = ['MIS', 'CNC', 'NRML']
    return product.upper() in valid_products


def validate_exchange(exchange: str) -> bool:
    """Validate exchange"""
    valid_exchanges = ['NSE', 'BSE', 'NFO', 'MCX', 'CDS']
    return exchange.upper() in valid_exchanges


def validate_transaction_type(transaction_type: str) -> bool:
    """Validate transaction type"""
    return transaction_type.upper() in ['BUY', 'SELL']


class TradingValidator:
    """Comprehensive trading validation"""
    
    @staticmethod
    def validate_order_request(order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete order request"""
        errors = []
        
        # Required fields
        required_fields = ['tradingsymbol', 'transaction_type', 'quantity', 'order_type', 'product']
        for field in required_fields:
            if field not in order_data or order_data[field] is None:
                errors.append(f"{field} is required")
        
        if errors:
            return {"valid": False, "errors": errors}
        
        # Validate individual fields
        symbol = order_data.get('tradingsymbol', '')
        if not validate_trading_symbol(symbol):
            errors.append("Invalid trading symbol format")
        
        transaction_type = order_data.get('transaction_type', '')
        if not validate_transaction_type(transaction_type):
            errors.append("Invalid transaction type")
        
        quantity = order_data.get('quantity', 0)
        if not validate_quantity(quantity):
            errors.append("Invalid quantity")
        
        order_type = order_data.get('order_type', '')
        if not validate_order_type(order_type):
            errors.append("Invalid order type")
        
        product = order_data.get('product', '')
        if not validate_product_type(product):
            errors.append("Invalid product type")
        
        exchange = order_data.get('exchange', 'NSE')
        if not validate_exchange(exchange):
            errors.append("Invalid exchange")
        
        # Conditional validations
        if order_type.upper() == 'LIMIT':
            price = order_data.get('price', 0)
            if not validate_price(price):
                errors.append("Price is required for LIMIT orders")
        
        if order_type.upper() in ['SL', 'SL-M']:
            trigger_price = order_data.get('trigger_price', 0)
            if not validate_price(trigger_price):
                errors.append("Trigger price is required for SL orders")
        
        # Stoploss and target validation
        stoploss = order_data.get('stoploss')
        target = order_data.get('target')
        price = order_data.get('price') or order_data.get('trigger_price', 0)
        
        if stoploss is not None:
            if not validate_price(stoploss):
                errors.append("Invalid stoploss price")
            elif price > 0:
                # Validate stoploss logic
                if transaction_type.upper() == 'BUY' and stoploss >= price:
                    errors.append("Stoploss must be below entry price for BUY orders")
                elif transaction_type.upper() == 'SELL' and stoploss <= price:
                    errors.append("Stoploss must be above entry price for SELL orders")
        
        if target is not None:
            if not validate_price(target):
                errors.append("Invalid target price")
            elif price > 0:
                # Validate target logic
                if transaction_type.upper() == 'BUY' and target <= price:
                    errors.append("Target must be above entry price for BUY orders")
                elif transaction_type.upper() == 'SELL' and target >= price:
                    errors.append("Target must be below entry price for SELL orders")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    @staticmethod
    def validate_strategy_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate strategy configuration"""
        errors = []
        
        # Required fields
        required_fields = ['strategy_type', 'symbol', 'quantity', 'product']
        for field in required_fields:
            if field not in config or config[field] is None:
                errors.append(f"{field} is required")
        
        if errors:
            return {"valid": False, "errors": errors}
        
        # Validate strategy type
        valid_strategies = [
            '915_candle_break', 'mean_reversion', 'momentum_breakout',
            'support_resistance', 'rsi_reversal', 'macd_crossover', 'ema_cross'
        ]
        if config.get('strategy_type') not in valid_strategies:
            errors.append(f"Invalid strategy type. Valid types: {valid_strategies}")
        
        # Validate trading hours
        active_hours_start = config.get('active_hours_start')
        active_hours_end = config.get('active_hours_end')
        
        if active_hours_start and active_hours_end:
            try:
                from datetime import time
                start = time.fromisoformat(active_hours_start)
                end = time.fromisoformat(active_hours_end)
                
                if start >= end:
                    errors.append("Active hours start must be before end")
            except ValueError:
                errors.append("Invalid time format. Use HH:MM format")
        
        # Validate percentages
        stoploss_pct = config.get('stoploss_pct', 0)
        target_pct = config.get('target_pct', 0)
        
        if stoploss_pct <= 0 or stoploss_pct > 1:
            errors.append("Stoploss percentage must be between 0 and 1")
        
        if target_pct <= 0 or target_pct > 1:
            errors.append("Target percentage must be between 0 and 1")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }


# Global validator instance
trading_validator = TradingValidator()


def setup_error_handlers(app):
    """Setup FastAPI error handlers"""
    
    @app.exception_handler(AlgoFeastException)
    async def algofeast_exception_handler(request: Request, exc: AlgoFeastException):
        error_response = error_handler.handle_exception(exc, {
            "path": request.url.path,
            "method": request.method,
            "client_ip": request.client.host if request.client else None
        })
        return JSONResponse(
            status_code=error_response["status_code"],
            content=error_response
        )
    
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        error_response = error_handler.handle_exception(exc, {
            "path": request.url.path,
            "method": request.method,
            "client_ip": request.client.host if request.client else None
        })
        return JSONResponse(
            status_code=error_response["status_code"],
            content=error_response
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        error_response = error_handler.handle_exception(exc, {
            "path": request.url.path,
            "method": request.method,
            "client_ip": request.client.host if request.client else None
        })
        return JSONResponse(
            status_code=error_response["status_code"],
            content=error_response
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        error_response = error_handler.handle_exception(exc, {
            "path": request.url.path,
            "method": request.method,
            "client_ip": request.client.host if request.client else None
        })
        return JSONResponse(
            status_code=error_response["status_code"],
            content=error_response
        )
