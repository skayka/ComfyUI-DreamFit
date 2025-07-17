"""
ComfyUI-DreamFit
Garment-centric human generation nodes for ComfyUI using DreamFit with Flux
Based on: https://github.com/bytedance/DreamFit
"""

try:
    from .nodes.custom_garmentfit_v3 import CustomGarmentFitLoaderV3, ApplyCustomGarmentFitV3
    
    NODE_CLASS_MAPPINGS = {
        "CustomGarmentFitLoaderV3": CustomGarmentFitLoaderV3,
        "ApplyCustomGarmentFitV3": ApplyCustomGarmentFitV3,
    }
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        "CustomGarmentFitLoaderV3": "Load GarmentFit Model",
        "ApplyCustomGarmentFitV3": "Apply GarmentFit",
    }
    
except ImportError as e:
    print(f"[DreamFit] Failed to load nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']