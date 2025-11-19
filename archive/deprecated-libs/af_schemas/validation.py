"""
AF Schemas Validation
"""

from typing import Dict, Any

class ValidationError(Exception):
    """Validation error"""
    pass

def validate_task(task_data: Dict[str, Any]) -> bool:
    """Validate task data"""
    required_fields = ['id', 'description']
    for field in required_fields:
        if field not in task_data:
            raise ValidationError(f"Missing required field: {field}")
    return True

def validate_agent_contract(contract_data: Dict[str, Any]) -> bool:
    """Validate agent contract data"""
    required_fields = ['name', 'capabilities']
    for field in required_fields:
        if field not in contract_data:
            raise ValidationError(f"Missing required field: {field}")
    return True



