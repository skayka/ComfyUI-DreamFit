"""
Custom types for DreamFit nodes in ComfyUI
"""

from typing import Dict, Optional, Any
import torch


class DreamFitFeatures:
    """
    Custom type for passing DreamFit features between nodes
    Contains extracted garment features and metadata
    """
    def __init__(self, data: Dict[str, Any]):
        self.garment_token = data.get("garment_token")
        self.pooled_features = data.get("pooled_features")
        self.patch_features = data.get("patch_features")
        self.features = data.get("features", {})
        self.attention_weights = data.get("attention_weights", {})
        self.encoder_config = data.get("encoder_config", {})
        self.pose_features = data.get("pose_features")
        self._data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary format"""
        return self._data
    
    def get_injection_config(self) -> Dict[str, Any]:
        """Get feature injection configuration"""
        return {
            "garment_token": self.garment_token,
            "pooled_features": self.pooled_features,
            "patch_features": self.patch_features,
            "pose_features": self.pose_features,
        }
    
    def has_pose(self) -> bool:
        """Check if pose features are available"""
        return self.pose_features is not None
    
    def get_feature_dim(self) -> int:
        """Get the feature dimension"""
        if self.garment_token is not None:
            return self.garment_token.shape[-1]
        return 0


# Register the custom type in ComfyUI
def register_dreamfit_types():
    """Register custom DreamFit types in ComfyUI"""
    try:
        # Try to register with ComfyUI's type system
        import sys
        if "comfy.nodes" in sys.modules:
            # Add to ComfyUI's type system
            sys.modules["comfy.nodes"].DREAMFIT_FEATURES = DreamFitFeatures
            
            # Also register type name
            if hasattr(sys.modules["comfy.nodes"], "CUSTOM_TYPES"):
                sys.modules["comfy.nodes"].CUSTOM_TYPES["DREAMFIT_FEATURES"] = DreamFitFeatures
            else:
                sys.modules["comfy.nodes"].CUSTOM_TYPES = {"DREAMFIT_FEATURES": DreamFitFeatures}
                
        print("DreamFit custom types registered successfully")
    except Exception as e:
        print(f"Warning: Could not register DreamFit types: {e}")


# Auto-register when imported
register_dreamfit_types()