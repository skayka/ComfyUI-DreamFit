"""
LoRA Adapter for DreamFit
Applies LoRA weights to Flux model for garment-centric generation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import math


class LoRALayer(nn.Module):
    """
    LoRA layer implementation
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation"""
        # x: [*, in_features]
        result = x @ self.lora_A.T  # [*, rank]
        result = self.dropout(result)
        result = result @ self.lora_B.T  # [*, out_features]
        return result * self.scaling


class DreamFitLoRAAdapter:
    """
    Manages LoRA adaptation for DreamFit
    """
    
    # Target modules in Flux model for LoRA adaptation
    TARGET_MODULES = [
        "attn.to_q",
        "attn.to_k", 
        "attn.to_v",
        "attn.to_out",
        "mlp.fc1",
        "mlp.fc2"
    ]
    
    def __init__(self, rank: int = 16, alpha: float = 16.0, dropout: float = 0.0):
        """
        Initialize LoRA adapter
        
        Args:
            rank: LoRA rank
            alpha: LoRA alpha scaling factor
            dropout: Dropout rate for LoRA
        """
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lora_layers = {}
        self.original_forwards = {}
        
    def create_lora_layers(self, model: nn.Module, lora_weights: Optional[Dict] = None) -> Dict[str, LoRALayer]:
        """
        Create LoRA layers for target modules in model
        
        Args:
            model: The Flux model to adapt
            lora_weights: Optional pre-trained LoRA weights
            
        Returns:
            Dictionary of created LoRA layers
        """
        lora_layers = {}
        
        for name, module in model.named_modules():
            if any(target in name for target in self.TARGET_MODULES):
                if isinstance(module, nn.Linear):
                    # Create LoRA layer
                    lora_layer = LoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout
                    )
                    
                    # Load pre-trained weights if available
                    if lora_weights and name in lora_weights:
                        weight_dict = lora_weights[name]
                        if "lora_A" in weight_dict:
                            lora_layer.lora_A.data = weight_dict["lora_A"]
                        if "lora_B" in weight_dict:
                            lora_layer.lora_B.data = weight_dict["lora_B"]
                    
                    lora_layers[name] = lora_layer
        
        return lora_layers
    
    def apply_lora_to_model(self, model: nn.Module, lora_weights: Optional[Dict] = None) -> nn.Module:
        """
        Apply LoRA adaptation to model
        
        Args:
            model: The Flux model to adapt
            lora_weights: Optional pre-trained LoRA weights
            
        Returns:
            Adapted model with LoRA layers
        """
        # Create LoRA layers
        self.lora_layers = self.create_lora_layers(model, lora_weights)
        
        # Hook into model forward passes
        for name, module in model.named_modules():
            if name in self.lora_layers:
                if isinstance(module, nn.Linear):
                    # Store original forward
                    self.original_forwards[name] = module.forward
                    
                    # Create new forward with LoRA
                    lora_layer = self.lora_layers[name]
                    
                    def make_lora_forward(linear_module, lora):
                        def forward(x):
                            # Original linear transformation
                            result = F.linear(x, linear_module.weight, linear_module.bias)
                            # Add LoRA
                            result = result + lora(x)
                            return result
                        return forward
                    
                    # Replace forward method
                    module.forward = make_lora_forward(module, lora_layer)
        
        return model
    
    def remove_lora_from_model(self, model: nn.Module):
        """
        Remove LoRA adaptation from model
        
        Args:
            model: The adapted model
        """
        for name, module in model.named_modules():
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]
        
        self.original_forwards.clear()
        self.lora_layers.clear()
    
    def merge_lora_weights(self, model: nn.Module):
        """
        Merge LoRA weights into the base model weights
        
        Args:
            model: The adapted model
        """
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in self.lora_layers and isinstance(module, nn.Linear):
                    lora_layer = self.lora_layers[name]
                    # Compute LoRA weight update
                    lora_weight = lora_layer.lora_B @ lora_layer.lora_A * lora_layer.scaling
                    # Add to original weight
                    module.weight.data += lora_weight
        
        # Remove LoRA layers after merging
        self.remove_lora_from_model(model)
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get state dict of LoRA parameters
        
        Returns:
            Dictionary of LoRA parameters
        """
        state_dict = {}
        for name, lora_layer in self.lora_layers.items():
            state_dict[f"{name}.lora_A"] = lora_layer.lora_A
            state_dict[f"{name}.lora_B"] = lora_layer.lora_B
        return state_dict
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load LoRA parameters from state dict
        
        Args:
            state_dict: Dictionary of LoRA parameters
        """
        for key, value in state_dict.items():
            if key.endswith(".lora_A"):
                module_name = key[:-7]  # Remove .lora_A
                if module_name in self.lora_layers:
                    self.lora_layers[module_name].lora_A.data = value
            elif key.endswith(".lora_B"):
                module_name = key[:-7]  # Remove .lora_B
                if module_name in self.lora_layers:
                    self.lora_layers[module_name].lora_B.data = value


def apply_dreamfit_lora(
    model: nn.Module,
    dreamfit_weights: Dict,
    rank: int = 16,
    alpha: float = 16.0,
    merge: bool = False
) -> Tuple[nn.Module, DreamFitLoRAAdapter]:
    """
    Convenience function to apply DreamFit LoRA to a model
    
    Args:
        model: The Flux model
        dreamfit_weights: DreamFit weights containing LoRA parameters
        rank: LoRA rank
        alpha: LoRA alpha
        merge: Whether to merge LoRA weights into base model
        
    Returns:
        Tuple of (adapted_model, adapter)
    """
    # Extract LoRA weights from DreamFit weights
    lora_weights = {}
    for key, value in dreamfit_weights.items():
        if "lora" in key.lower():
            # Parse module name from key
            # Expected format: "flux.transformer.blocks.0.attn.to_q.lora_A"
            parts = key.split(".")
            if len(parts) > 2 and parts[-1] in ["lora_A", "lora_B"]:
                module_name = ".".join(parts[:-1])
                if module_name not in lora_weights:
                    lora_weights[module_name] = {}
                lora_weights[module_name][parts[-1]] = value
    
    # Create adapter
    adapter = DreamFitLoRAAdapter(rank=rank, alpha=alpha)
    
    # Apply LoRA
    adapted_model = adapter.apply_lora_to_model(model, lora_weights)
    
    # Merge if requested
    if merge:
        adapter.merge_lora_weights(adapted_model)
    
    return adapted_model, adapter


# Import F for the forward functions
import torch.nn.functional as F