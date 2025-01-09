from pydantic._internal._model_construction import ModelMetaclass
from typing import Union, Dict, Any

def is_pydantic_model( schema: Any) -> bool:
    """Check if the schema is a Pydantic model class or instance.
    
    Args:
        schema: The schema to check
        
    Returns:
        bool: True if the schema is a Pydantic model, False otherwise
    """
    # Check if it's a Pydantic model class (ModelMetaclass)
    if isinstance(schema, ModelMetaclass):
        return True
    return False