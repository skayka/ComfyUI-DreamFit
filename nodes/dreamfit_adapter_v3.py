"""
DreamFit Flux Adapter V3 - Properly integrated with conditioning pipeline
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import copy

# DreamFit imports
from ..dreamfit_core.models.lora_adapter import DreamFitLoRAAdapter


class DreamFitFluxAdapterV3:
    """
    Simplified adapter that uses DreamFit conditioning properly
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "dreamfit_conditioning": ("DREAMFIT_CONDITIONING",),
                "positive": ("STRING", {
                    "default": "A person wearing the garment",
                    "multiline": True
                }),
                "negative": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
            },
            "optional": {
                "lora_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "adapt"
    CATEGORY = "DreamFit"
    
    def adapt(
        self,
        model,
        clip,
        dreamfit_conditioning: Dict,
        positive: str,
        negative: str,
        lora_strength: float = 1.0
    ):
        """
        Adapt model and create properly formatted conditioning
        """
        # Clone the model
        adapted_model = copy.deepcopy(model)
        
        # Extract components from conditioning
        garment_features = dreamfit_conditioning.get("garment_features", {})
        injection_config = dreamfit_conditioning.get("injection_config", {})
        adapter_state = dreamfit_conditioning.get("adapter_state", {})
        
        # Apply LoRA if available
        if adapter_state.get("lora_weights") and lora_strength > 0:
            adapted_model = self._apply_lora(
                adapted_model,
                adapter_state["lora_weights"],
                lora_strength
            )
        
        # Encode text
        positive_tokens = clip.tokenize(positive)
        negative_tokens = clip.tokenize(negative)
        
        positive_cond, positive_pooled = clip.encode_from_tokens(positive_tokens)
        negative_cond, negative_pooled = clip.encode_from_tokens(negative_tokens)
        
        # Create conditioning with embedded DreamFit features
        positive_conditioning = [[
            positive_cond,
            {
                "pooled_output": positive_pooled,
                "dreamfit_features": garment_features,
                "dreamfit_config": injection_config
            }
        ]]
        
        negative_conditioning = [[
            negative_cond,
            {"pooled_output": negative_pooled}
        ]]
        
        # Store reference in model for samplers that need it
        adapted_model.dreamfit_conditioning = dreamfit_conditioning
        
        return (adapted_model, positive_conditioning, negative_conditioning)
    
    def _apply_lora(self, model, lora_weights, strength):
        """Simple LoRA application"""
        # This is a simplified version - in production you'd want more sophisticated handling
        try:
            for name, weight in lora_weights.items():
                # Apply LoRA weights with strength scaling
                # This is placeholder logic - real implementation would properly merge LoRA
                pass
        except Exception as e:
            print(f"Warning: LoRA application failed: {e}")
        
        return model


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DreamFitFluxAdapterV3": DreamFitFluxAdapterV3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitFluxAdapterV3": "DreamFit Flux Adapter V3",
}