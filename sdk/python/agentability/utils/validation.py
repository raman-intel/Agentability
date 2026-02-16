"""Input validation utilities."""

from typing import Any, Optional
from uuid import UUID


def validate_uuid(value: Any) -> UUID:
    """Validate and convert value to UUID.
    
    Args:
        value: Value to validate (str or UUID)
        
    Returns:
        UUID object
        
    Raises:
        ValueError: If value is not a valid UUID
    """
    if isinstance(value, UUID):
        return value
    
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError) as e:
        raise ValueError(f"Invalid UUID: {value}") from e


def validate_float_range(
    value: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    name: str = "value",
) -> float:
    """Validate that a float is within a specified range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        name: Name of the value (for error messages)
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If value is out of range
    """
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    
    return value


def validate_positive_int(value: int, name: str = "value") -> int:
    """Validate that an integer is positive.
    
    Args:
        value: Value to validate
        name: Name of the value (for error messages)
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If value is not positive
    """
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    
    return value


def validate_non_empty_string(value: str, name: str = "value") -> str:
    """Validate that a string is non-empty.
    
    Args:
        value: Value to validate
        name: Name of the value (for error messages)
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If string is empty
    """
    if not value or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    
    return value
