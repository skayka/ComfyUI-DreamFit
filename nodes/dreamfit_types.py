"""
Custom type definitions for DreamFit nodes
ComfyUI doesn't support custom types directly, so we wrap them in dictionaries
"""

# Register custom types with ComfyUI
# These will be treated as generic types that can be passed between nodes
DREAMFIT_TYPES = [
    "DREAMFIT_MODEL",
    "DREAMFIT_ENCODER", 
    "DREAMFIT_CONFIG",
    "DREAMFIT_CONDITIONING"
]

# Helper functions to wrap/unwrap custom types
def wrap_dreamfit_data(data, type_name):
    """Wrap data in a dictionary with type info for ComfyUI"""
    return {
        "_dreamfit_type": type_name,
        "data": data
    }

def unwrap_dreamfit_data(wrapped_data):
    """Extract data from wrapped dictionary"""
    if isinstance(wrapped_data, dict) and "_dreamfit_type" in wrapped_data:
        return wrapped_data["data"]
    return wrapped_data