"""
Input validation utilities
"""
from typing import Any, Optional
from pydantic import BaseModel, ValidationError
from core.exceptions import ValidationError as AppValidationError


def validate_request(model: type[BaseModel], data: dict[str, Any]) -> BaseModel:
    """
    Validate request data against a Pydantic model
    
    Args:
        model: Pydantic model class
        data: Request data dictionary
        
    Returns:
        Validated model instance
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        return model(**data)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            errors.append(f"{field}: {error['msg']}")
        raise AppValidationError(
            message="Validation failed",
            details={"errors": errors, "raw_errors": e.errors()}
        )


def validate_time_format(time_str: str) -> bool:
    """Validate time format (HH:MM)"""
    try:
        from datetime import datetime
        datetime.strptime(time_str, "%H:%M")
        return True
    except ValueError:
        return False


def validate_percentage(value: float, min_val: float = 0.0, max_val: float = 100.0) -> bool:
    """Validate percentage value"""
    return min_val <= value <= max_val

