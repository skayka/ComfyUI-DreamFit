"""
DreamFit Flux Adapter Node
Applies DreamFit adaptation to Flux model
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import copy

# DreamFit imports
from ..dreamfit_core.models import AdaptiveAttentionInjector
from ..dreamfit_core.models.adaptive_attention import InjectionConfig
from ..dreamfit_core.models.lora_adapter import apply_dreamfit_lora, DreamFitLoRAAdapter


class DreamFitFluxAdapter:
    """
    Apply DreamFit adaptation to Flux model
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_model": ("MODEL",),
                "dreamfit_model": ("DREAMFIT_MODEL",),
                "conditioning": ("DREAMFIT_CONDITIONING",),
            },
            "optional": {
                "lora_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "merge_lora": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "DREAMFIT_ADAPTER")
    RETURN_NAMES = ("adapted_model", "adapter_info")
    FUNCTION = "adapt_model"
    CATEGORY = "DreamFit"
    
    def __init__(self):
        self.injection_hooks = []
        self.original_forwards = {}
        
    def adapt_model(
        self,
        flux_model,
        dreamfit_model: Dict,
        conditioning: Dict,
        lora_strength: float = 1.0,
        merge_lora: bool = False
    ) -> Tuple[Any, Dict]:
        """
        Apply DreamFit adaptation to Flux model
        
        Args:
            flux_model: The Flux model from ComfyUI
            dreamfit_model: The loaded DreamFit model dict
            conditioning: DreamFit conditioning from encode node
            lora_strength: Strength of LoRA adaptation
            merge_lora: Whether to merge LoRA weights
            
        Returns:
            Tuple of (adapted_model, adapter_info)
        """
        # Clone the model to avoid modifying the original
        adapted_model = copy.deepcopy(flux_model)
        
        # Get the actual model from ComfyUI wrapper
        if hasattr(adapted_model, 'model'):
            base_model = adapted_model.model
        else:
            base_model = adapted_model
        
        adapter_info = {
            "type": "dreamfit_flux_adapter",
            "lora_applied": False,
            "injection_configured": False,
            "adapter_modules": [],
            "injection_layers": []
        }
        
        # Apply LoRA adaptation if weights are available
        lora_adapter = None
        if "lora_weights" in dreamfit_model.get("weights", {}):
            try:
                # Scale LoRA weights by strength
                lora_weights = dreamfit_model["weights"]["lora_weights"]
                if lora_strength != 1.0:
                    scaled_weights = {}
                    for key, value in lora_weights.items():
                        if isinstance(value, torch.Tensor):
                            scaled_weights[key] = value * lora_strength
                        else:
                            scaled_weights[key] = value
                    lora_weights = scaled_weights
                
                # Apply LoRA
                _, lora_adapter = apply_dreamfit_lora(
                    base_model,
                    {"lora_weights": lora_weights},
                    rank=dreamfit_model["config"].get("lora_rank", 16),
                    alpha=dreamfit_model["config"].get("lora_alpha", 16.0),
                    merge=merge_lora
                )
                
                adapter_info["lora_applied"] = True
                adapter_info["lora_merged"] = merge_lora
                adapter_info["lora_strength"] = lora_strength
                adapter_info["adapter_modules"] = list(lora_adapter.lora_layers.keys())
                
            except Exception as e:
                print(f"Warning: Failed to apply LoRA adaptation: {e}")
        
        # Create adaptive attention injector
        injection_config = InjectionConfig(
            injection_layers=conditioning["injection_config"]["injection_layers"],
            injection_strength=conditioning["injection_config"]["injection_strength"],
            injection_mode=conditioning["injection_config"]["injection_mode"],
            cross_attention_dim=1024,  # From encoder hidden_dim
            num_heads=16,
            dropout=0.0
        )
        
        attention_injector = AdaptiveAttentionInjector(injection_config)
        
        # Configure injection hooks
        self._setup_injection_hooks(
            base_model,
            attention_injector,
            conditioning["garment_features"],
            conditioning["injection_config"]["injection_layers"]
        )
        
        adapter_info["injection_configured"] = True
        adapter_info["injection_layers"] = conditioning["injection_config"]["injection_layers"]
        adapter_info["injection_mode"] = conditioning["injection_config"]["injection_mode"]
        
        # Store adapter components for later use
        if hasattr(adapted_model, 'dreamfit_components'):
            adapted_model.dreamfit_components = {
                "lora_adapter": lora_adapter,
                "attention_injector": attention_injector,
                "conditioning": conditioning,
                "hooks": self.injection_hooks
            }
        
        return (adapted_model, adapter_info)
    
    def _setup_injection_hooks(
        self,
        model: nn.Module,
        injector: AdaptiveAttentionInjector,
        garment_features: Dict,
        injection_layers: List[int]
    ):
        """
        Set up hooks for feature injection
        
        Args:
            model: The Flux model
            injector: The attention injector
            garment_features: Pre-computed garment features
            injection_layers: Which layers to inject into
        """
        # Clear any existing hooks
        for hook in self.injection_hooks:
            hook.remove()
        self.injection_hooks = []
        
        # Find transformer blocks in Flux model
        transformer_blocks = []
        for name, module in model.named_modules():
            # Look for transformer blocks - adjust based on actual Flux structure
            if "transformer" in name and "block" in name:
                transformer_blocks.append((name, module))
        
        # Register hooks for specified layers
        for layer_idx in injection_layers:
            if layer_idx < len(transformer_blocks):
                block_name, block_module = transformer_blocks[layer_idx]
                
                # Create hook function
                def make_injection_hook(layer_key, features):
                    def hook(module, input, output):
                        # This is a simplified hook - actual implementation
                        # would need to properly intercept and modify features
                        # based on Flux model structure
                        
                        # For now, we'll store features for use in sampling
                        if hasattr(module, '_dreamfit_features'):
                            module._dreamfit_features = features
                        
                        return output
                    return hook
                
                # Register hook
                hook = block_module.register_forward_hook(
                    make_injection_hook(f"layer_{layer_idx}", garment_features)
                )
                self.injection_hooks.append(hook)
    
    def remove_adaptation(self, model):
        """
        Remove DreamFit adaptation from model
        
        Args:
            model: The adapted model
        """
        # Remove injection hooks
        for hook in self.injection_hooks:
            hook.remove()
        self.injection_hooks = []
        
        # Remove LoRA if present
        if hasattr(model, 'dreamfit_components'):
            if model.dreamfit_components.get('lora_adapter'):
                model.dreamfit_components['lora_adapter'].remove_lora_from_model(
                    model.model if hasattr(model, 'model') else model
                )
            delattr(model, 'dreamfit_components')


class DreamFitAdapterInfo:
    """
    Display information about DreamFit adapter
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter_info": ("DREAMFIT_ADAPTER",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_info"
    CATEGORY = "DreamFit/Utils"
    
    def get_info(self, adapter_info: Dict) -> Tuple[str]:
        """Get adapter information"""
        info_lines = [
            "DreamFit Adapter Info:",
            f"Type: {adapter_info.get('type', 'unknown')}",
            f"LoRA Applied: {adapter_info.get('lora_applied', False)}",
        ]
        
        if adapter_info.get('lora_applied'):
            info_lines.extend([
                f"  - LoRA Merged: {adapter_info.get('lora_merged', False)}",
                f"  - LoRA Strength: {adapter_info.get('lora_strength', 1.0)}",
                f"  - Adapter Modules: {len(adapter_info.get('adapter_modules', []))}",
            ])
        
        info_lines.extend([
            f"Injection Configured: {adapter_info.get('injection_configured', False)}",
            f"  - Injection Layers: {adapter_info.get('injection_layers', [])}",
            f"  - Injection Mode: {adapter_info.get('injection_mode', 'N/A')}",
        ])
        
        return ("\n".join(info_lines),)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DreamFitFluxAdapter": DreamFitFluxAdapter,
    "DreamFitAdapterInfo": DreamFitAdapterInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitFluxAdapter": "DreamFit Flux Adapter",
    "DreamFitAdapterInfo": "DreamFit Adapter Info",
}