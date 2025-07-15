"""
ComfyUI-DreamFit
Garment-centric human generation nodes for ComfyUI using DreamFit with Flux
Based on: https://github.com/bytedance/DreamFit
"""

# Version info
__version__ = "1.0.0"
__author__ = "ComfyUI-DreamFit"

# Only import these when ComfyUI loads the nodes
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def init():
    """Initialize the nodes - called by ComfyUI"""
    try:
        from .nodes import NODE_CLASS_MAPPINGS as node_mappings
        from .nodes import NODE_DISPLAY_NAME_MAPPINGS as display_mappings
        
        NODE_CLASS_MAPPINGS.update(node_mappings)
        NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)
        
        return True
    except Exception as e:
        print(f"[ComfyUI-DreamFit] Failed to load nodes: {e}")
        return False

# Try to initialize if we're being loaded by ComfyUI
try:
    init()
except:
    # This is fine - it means we're being imported outside of ComfyUI
    pass

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']