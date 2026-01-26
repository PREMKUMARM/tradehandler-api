"""
Trading utilities and validation endpoints
"""
from fastapi import APIRouter, Request
from schemas.trading import TradeValidationRequest, TradeValidationResponse
from core.exceptions import ValidationError
from utils.logger import log_info, log_error

router = APIRouter(prefix="/trading", tags=["Trading"])


@router.post("/validate-trade")
async def validate_trade(request: Request, validation_request: TradeValidationRequest):
    """
    Validate a trade based on risk/reward parameters
    
    This endpoint validates whether a trade meets the configured risk/reward criteria:
    - Risk should be <= max risk % of capital
    - Reward should be >= (Risk Amount × Reward% / Risk%)
    
    Returns detailed validation results including whether the trade is "good" or "bad".
    """
    try:
        # Extract parameters
        entry_price = validation_request.entry_price
        stoploss = validation_request.stoploss
        target_price = validation_request.target_price
        quantity = validation_request.quantity
        capital = validation_request.capital
        risk_percentage = validation_request.risk_percentage
        reward_percentage = validation_request.reward_percentage
        
        # Validate input
        if entry_price <= stoploss:
            raise ValidationError(
                message="Entry price must be greater than stoploss",
                field="entry_price"
            )
        
        if target_price <= entry_price:
            raise ValidationError(
                message="Target price must be greater than entry price",
                field="target_price"
            )
        
        # Calculate actual risk and reward amounts
        current_risk_amount = (entry_price - stoploss) * quantity
        current_reward_amount = (target_price - entry_price) * quantity
        
        # Calculate maximum allowed risk (capital × risk%)
        max_risk_amount = (capital * risk_percentage) / 100
        
        # Calculate reward ratio (reward% / risk%)
        reward_ratio = reward_percentage / risk_percentage
        
        # Calculate minimum required reward based on reward ratio relative to actual risk
        # Reward should be >= (Risk Amount × Reward% / Risk%)
        min_required_reward = current_risk_amount * reward_ratio
        
        # Validate risk: should be <= max risk % of capital
        risk_within_limit = current_risk_amount <= max_risk_amount
        
        # Validate reward: should be >= (Risk × Reward Ratio)
        reward_meets_requirement = current_reward_amount >= min_required_reward
        
        # Overall validation: both conditions must be met
        is_good_trade = risk_within_limit and reward_meets_requirement
        
        # Prepare validation details
        validation_details = {
            "risk_percentage_of_capital": round((current_risk_amount / capital * 100), 2) if capital > 0 else 0,
            "reward_percentage_of_capital": round((current_reward_amount / capital * 100), 2) if capital > 0 else 0,
            "risk_percentage_used": round((current_risk_amount / max_risk_amount * 100), 2) if max_risk_amount > 0 else 0,
            "reward_ratio_achieved": round((current_reward_amount / current_risk_amount), 2) if current_risk_amount > 0 else 0,
            "required_reward_ratio": round(reward_ratio, 2)
        }
        
        log_info(
            f"Trade validation: is_good={is_good_trade}, "
            f"risk=₹{current_risk_amount:.2f}, reward=₹{current_reward_amount:.2f}, "
            f"min_required=₹{min_required_reward:.2f}"
        )
        
        response_data = TradeValidationResponse(
            is_good_trade=is_good_trade,
            risk_amount=round(current_risk_amount, 2),
            reward_amount=round(current_reward_amount, 2),
            max_risk_amount=round(max_risk_amount, 2),
            min_required_reward=round(min_required_reward, 2),
            reward_ratio=round(reward_ratio, 2),
            risk_within_limit=risk_within_limit,
            reward_meets_requirement=reward_meets_requirement,
            validation_details=validation_details
        )
        
        # Return in standard API response format
        from core.responses import SuccessResponse
        return SuccessResponse(
            data=response_data.dict(),
            message="Trade validation completed"
        )
        
    except ValidationError:
        raise
    except Exception as e:
        log_error(f"Error validating trade: {str(e)}")
        raise ValidationError(
            message=f"Error validating trade: {str(e)}",
            field="validation"
        )

