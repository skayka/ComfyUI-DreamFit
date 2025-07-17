"""
ComfyUI-DreamFit
Garment-centric human generation nodes for ComfyUI
"""

# Try to import the full version first
try:
    from .nodes.custom_garmentfit_v3 import (
        GarmentCLIPVisionLoader, 
        CustomGarmentFitLoaderV3, 
        ApplyCustomGarmentFitV3
    )
    
    NODE_CLASS_MAPPINGS = {
        "GarmentCLIPVisionLoader": GarmentCLIPVisionLoader,
        "CustomGarmentFitLoaderV3": CustomGarmentFitLoaderV3,
        "ApplyCustomGarmentFitV3": ApplyCustomGarmentFitV3,
    }
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        "GarmentCLIPVisionLoader": "Load Garment CLIP Vision",
        "CustomGarmentFitLoaderV3": "Load GarmentFit Model",
        "ApplyCustomGarmentFitV3": "Apply GarmentFit",
    }
    print("[DreamFit] Loaded full version successfully")
    
except ImportError as e:
    print(f"[DreamFit] Failed to load full version: {e}")
    
    # Fallback to simple version
    try:
        from .nodes.garmentfit_simple import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("[DreamFit] Loaded simple version as fallback")
    except ImportError as e2:
        print(f"[DreamFit] Failed to load simple version: {e2}")
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']