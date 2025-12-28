"""
Order-related request/response schemas
"""
from typing import Optional
from pydantic import BaseModel, Field, validator


class PlaceOrderRequest(BaseModel):
    """Request schema for placing an order"""
    tradingsymbol: str = Field(..., min_length=1, description="Trading symbol")
    exchange: str = Field(default="NSE", description="Exchange (NSE, BSE, etc.)")
    transaction_type: str = Field(..., pattern="^(BUY|SELL)$", description="BUY or SELL")
    quantity: int = Field(..., gt=0, description="Order quantity")
    order_type: str = Field(default="MARKET", description="Order type (MARKET, LIMIT, SL, SL-M)")
    product: str = Field(default="MIS", description="Product type (MIS, CNC, NRML)")
    price: Optional[float] = Field(None, gt=0, description="Price for LIMIT orders")
    trigger_price: Optional[float] = Field(None, gt=0, description="Trigger price for SL orders")
    validity: Optional[str] = Field(default="DAY", description="Order validity")
    disclosed_quantity: Optional[int] = Field(None, ge=0, description="Disclosed quantity")
    tag: Optional[str] = Field(default="algofeast", description="Order tag")

    @validator("transaction_type")
    def validate_transaction_type(cls, v):
        if v.upper() not in ["BUY", "SELL"]:
            raise ValueError("transaction_type must be BUY or SELL")
        return v.upper()

    @validator("order_type")
    def validate_order_type(cls, v):
        valid_types = ["MARKET", "LIMIT", "SL", "SL-M"]
        if v.upper() not in valid_types:
            raise ValueError(f"order_type must be one of {valid_types}")
        return v.upper()

    @validator("product")
    def validate_product(cls, v):
        valid_products = ["MIS", "CNC", "NRML"]
        if v.upper() not in valid_products:
            raise ValueError(f"product must be one of {valid_products}")
        return v.upper()


class ModifyOrderRequest(BaseModel):
    """Request schema for modifying an order"""
    order_id: str = Field(..., description="Order ID to modify")
    order_type: Optional[str] = Field(None, description="New order type")
    quantity: Optional[int] = Field(None, gt=0, description="New quantity")
    price: Optional[float] = Field(None, gt=0, description="New price")
    trigger_price: Optional[float] = Field(None, gt=0, description="New trigger price")
    validity: Optional[str] = Field(None, description="New validity")


class CancelOrderRequest(BaseModel):
    """Request schema for canceling an order"""
    order_id: str = Field(..., description="Order ID to cancel")
    variety: Optional[str] = Field(default="regular", description="Order variety")


class OrderResponse(BaseModel):
    """Response schema for order operations"""
    order_id: str = Field(..., description="Order ID")
    status: str = Field(..., description="Order status")
    message: Optional[str] = Field(None, description="Status message")

