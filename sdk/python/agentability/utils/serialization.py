"""Data serialization utilities.

Handles conversion between Python objects and JSON/database formats.
"""

import json
from datetime import datetime
from typing import Any, Dict
from uuid import UUID


def serialize_data(data: Any) -> str:
    """Serialize data to JSON string.
    
    Handles special types like datetime, UUID, etc.
    
    Args:
        data: Data to serialize
        
    Returns:
        JSON string
    """
    return json.dumps(data, cls=AgentabilityJSONEncoder)


def deserialize_data(json_str: str) -> Any:
    """Deserialize JSON string to Python object.
    
    Args:
        json_str: JSON string
        
    Returns:
        Deserialized Python object
    """
    return json.loads(json_str)


class AgentabilityJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Agentability types."""
    
    def default(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format.
        
        Args:
            obj: Object to encode
            
        Returns:
            JSON-serializable representation
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, UUID):
            return str(obj)
        elif hasattr(obj, "dict"):
            # Pydantic model
            return obj.dict()
        elif hasattr(obj, "__dict__"):
            # Generic object with __dict__
            return obj.__dict__
        
        return super().default(obj)


def safe_json_dumps(data: Dict[str, Any]) -> str:
    """Safely dump data to JSON, handling errors gracefully.
    
    Args:
        data: Dictionary to serialize
        
    Returns:
        JSON string, or error message if serialization fails
    """
    try:
        return serialize_data(data)
    except (TypeError, ValueError) as e:
        return json.dumps({
            "error": "Serialization failed",
            "message": str(e),
            "data_type": str(type(data)),
        })
