"""
Redesigned DreamFit Flux Adapter Node
Properly integrates with ComfyUI's conditioning system
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import copy

# DreamFit imports
from ..dreamfit_core.models import AdaptiveAttentionInjector
from ..dreamfit_core.models.adaptive_attention import InjectionConfig
from ..dreamfit_core.models.lora_adapter import apply_dreamfit_lora, DreamFitLoRAAdapter


class DreamFitFluxAdapterV2:
    """
    Adapts a Flux model with DreamFit features and handles conditioning
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "dreamfit_model": ("DREAMFIT_MODEL",),
                "dreamfit_encoder": ("DREAMFIT_ENCODER",),
                "garment_image": ("IMAGE",),
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
                "model_image": ("IMAGE",),  # Optional pose reference
                "lora_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "garment_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "adapt_and_encode"
    CATEGORY = "DreamFit"
    
    def adapt_and_encode(
        self,
        model,
        clip,
        dreamfit_model: Dict,
        dreamfit_encoder,
        garment_image: torch.Tensor,
        positive: str,
        negative: str,
        model_image: Optional[torch.Tensor] = None,
        lora_strength: float = 1.0,
        garment_strength: float = 0.5
    ):
        """
        Adapt model with DreamFit and create conditioning
        """
        # Clone the model to avoid modifying the original
        adapted_model = copy.deepcopy(model)
        
        # Apply DreamFit LoRA adaptation
        if dreamfit_model.get("lora_weights"):
            adapted_model = self._apply_lora_adaptation(
                adapted_model,
                dreamfit_model["lora_weights"],
                lora_strength
            )
        
        # Encode text with CLIP
        positive_tokens = clip.tokenize(positive)
        negative_tokens = clip.tokenize(negative)
        
        positive_cond, positive_pooled = clip.encode_from_tokens(positive_tokens)
        negative_cond, negative_pooled = clip.encode_from_tokens(negative_tokens)
        
        # Process garment features
        with torch.no_grad():
            # Ensure garment image is in the right format
            if garment_image.dim() == 4:  # B, H, W, C
                garment_image = garment_image.permute(0, 3, 1, 2)  # B, C, H, W
            
            # Extract garment features using the encoder
            garment_features = dreamfit_encoder.encode_garment(garment_image)
            
            # If model image is provided, encode it too
            model_features = None
            if model_image is not None:
                if model_image.dim() == 4:
                    model_image = model_image.permute(0, 3, 1, 2)
                model_features = dreamfit_encoder.encode_pose(model_image)
        
        # Inject garment features into the model
        self._setup_attention_injection(
            adapted_model,
            garment_features,
            model_features,
            garment_strength
        )
        
        # Store DreamFit components in the model for the sampler
        adapted_model.dreamfit_components = {
            "garment_features": garment_features,
            "model_features": model_features,
            "garment_strength": garment_strength,
            "encoder": dreamfit_encoder
        }
        
        return (adapted_model, [[positive_cond, {"pooled_output": positive_pooled}]], [[negative_cond, {"pooled_output": negative_pooled}]])
    
    def _apply_lora_adaptation(self, model, lora_weights, strength):
        """Apply LoRA weights to the model"""
        try:
            # Apply LoRA adaptation
            lora_adapter = DreamFitLoRAAdapter(
                rank=16,  # Standard LoRA rank
                alpha=strength * 16,
                target_modules=["to_q", "to_v", "to_k", "to_out"]
            )
            
            # Apply weights
            for name, weight in lora_weights.items():
                if hasattr(model.model, name):
                    module = getattr(model.model, name)
                    if isinstance(module, nn.Linear):
                        # Apply LoRA to linear layers
                        adapted = lora_adapter.adapt_module(module, weight)
                        setattr(model.model, name, adapted)
        except Exception as e:
            print(f"Warning: Failed to apply LoRA adaptation: {e}")
        
        return model
    
    def _setup_attention_injection(self, model, garment_features, model_features, strength):
        """Setup attention injection for garment features"""
        try:
            # Create injection configuration
            config = InjectionConfig(
                injection_strength=strength,
                injection_mode="adaptive",
                target_layers=["cross_attention", "self_attention"],
                start_step=0,
                end_step=1000
            )
            
            # Create attention injector
            injector = AdaptiveAttentionInjector(config)
            
            # Setup hooks for injection
            def create_hook(features):
                def hook(module, input, output):
                    # Inject features into attention
                    if isinstance(output, tuple):
                        attn_output = output[0]
                        # Add garment features to attention
                        injected = injector.inject(attn_output, features)
                        return (injected,) + output[1:]
                    return output
                return hook
            
            # Register hooks on attention layers
            for name, module in model.model.named_modules():
                if "attention" in name.lower():
                    module.register_forward_hook(create_hook(garment_features))
                    
        except Exception as e:
            print(f"Warning: Failed to setup attention injection: {e}")


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DreamFitFluxAdapterV2": DreamFitFluxAdapterV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitFluxAdapterV2": "DreamFit Flux Adapter V2",
}